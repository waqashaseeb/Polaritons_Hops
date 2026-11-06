"""Run the required stages inside one allocation; return nonzero on any required failure."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED
from research_hpc import ROOT, read, dump, stamp, campaign_identity, configure_workspace
from run_support import inside, memory_mib, task_inventory, worker_capacity


def run_step(command, log_base):
    print("SRUN:", " ".join(str(v) for v in command), flush=True)
    with log_base.with_suffix(".out").open("a", encoding="utf-8") as out, log_base.with_suffix(".err").open("a", encoding="utf-8") as err:
        return subprocess.run(command, cwd=ROOT, stdout=out, stderr=err).returncode


def srun_prefix(tasks=1, memory="8G"):
    return ["srun", "--nodes=1", "--ntasks="+str(tasks), "--cpus-per-task=1", "--exclusive", "--exact",
            "--mem="+str(memory), "--kill-on-bad-exit=0", "--unbuffered"]


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["hops", "references"], default="hops")
    parser.add_argument("--job-directory", type=Path, required=True)
    args = parser.parse_args()
    job = inside(ROOT, args.job_directory)
    job.mkdir(parents=True, exist_ok=True)
    config = inside(ROOT, args.config)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("Run this entrypoint in a Slurm allocation through Batch")
    logs = ROOT/"logs"/job_id
    logs.mkdir(parents=True, exist_ok=True)
    summary = {"job_id": job_id, "job_name": os.environ.get("SLURM_JOB_NAME"), "project_root": str(ROOT),
               "configuration": str(config), "mode": args.mode, "started_utc": stamp(), "status": "RUNNING", "exit_code": 1,
               "HPC_execution_verified_locally": False, "synthetic": False, "stages": [], "paper_reproduction_verified": False}
    dump(job/"run_record.json", summary)
    folder, meta, lock_stream = None, None, None
    failed = False
    try:
        if args.mode=="references":
            command = srun_prefix()+[sys.executable, "-u", str(ROOT/"research_hpc.py"), "paper-reference-grid", "--config", str(ROOT/"configs/paper_reference_grid.json")]
            code = run_step(command, logs/"references")
            summary["stages"].append({"stage": "references", "exit_code": code})
            failed = code!=0
        else:
            folder, cfg, hashes, signature = campaign_identity(config)
            folder.mkdir(parents=True, exist_ok=True)
            summary["campaign"] = str(folder)
            summary["signature"] = signature
            dump(job/"run_record.json", summary)
            lock_stream = (folder/"allocation.lock").open("a+")
            try: fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another Batch owns this campaign; no existing job will be cancelled")
            command = srun_prefix()+[sys.executable, "-u", str(ROOT/"research_hpc.py"), "prepare", "--config", str(config), "--campaign", str(folder)]
            code = run_step(command, logs/"prepare")
            summary["stages"].append({"stage": "prepare", "exit_code": code})
            if code: raise RuntimeError("Scientific preparation failed; inspect prepare.err")
            meta = read(folder/"campaign.json")
            cpu_slots = int(os.environ.get("SLURM_NTASKS", "1"))
            memory_budget = int(os.environ.get("SLURM_MEM_PER_NODE", "491520"))
            if memory_budget<=0: raise ValueError("A finite node memory budget is required")
            reserve = 24*1024
            for stage in cfg["default_sequence"]:
                if (job/"stop_requested").exists() or int(os.environ["HOPS_DEADLINE_EPOCH"])-time.time()<360:
                    raise TimeoutError("Stopping before the allocation walltime; resume this campaign")
                selected = [task for task in meta["tasks"] if task["stage"]==stage]
                inventory = task_inventory(folder, selected)
                pending = sum(row["state"]!="COMPLETE" for row in inventory)
                if not pending:
                    summary["stages"].append({"stage": stage, "status": "PRESERVED_COMPLETE", "exit_code": 0, "active_workers": 0})
                    continue
                spec = cfg["stages"][stage]
                workers = worker_capacity(cpu_slots, memory_budget, reserve, memory_mib(spec["resources"]["memory"]), pending,
                                          int(spec.get("maximum_parallel", cpu_slots)))
                print(f"STAGE {stage}: remaining={pending}, active_workers={workers}, allocated_CPUs={cpu_slots}", flush=True)
                summary["current_stage"] = stage
                dump(job/"run_record.json", summary)
                command = srun_prefix(workers, str(memory_budget-reserve)+"M")
                command += ["--output="+str(logs/(stage+"_rank_%t.out")), "--error="+str(logs/(stage+"_rank_%t.err")),
                            sys.executable, "-u", str(ROOT/"worker_rank.py"), "--campaign", str(folder), "--stage", stage, "--job-directory", str(job)]
                code = run_step(command, logs/(stage+"_srun"))
                remaining = sum(row["state"]!="COMPLETE" for row in task_inventory(folder, selected))
                summary["stages"].append({"stage": stage, "exit_code": code, "active_workers": workers, "unfinished": remaining})
                dump(job/"run_record.json", summary)
                if code or remaining:
                    failed = True
                    break
    except BaseException as exc:
        failed = True
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        print("REQUIRED WORKFLOW FAILED:", exc, flush=True)
    finally:
        if folder and (folder/"campaign.json").is_file():
            meta = read(folder/"campaign.json")
            required_stages = set(meta["config"]["default_sequence"])
            rows = task_inventory(folder, [t for t in meta["tasks"] if t["stage"] in required_stages])
            summary["completed_experiments"] = sum(row["state"]=="COMPLETE" for row in rows)
            summary["planned_experiments"] = len(rows)
            summary["failed_or_unfinished_experiments"] = [row for row in rows if row["state"]!="COMPLETE"]
            failed = failed or summary["completed_experiments"]!=len(rows)
            try:
                command = srun_prefix()+[sys.executable, "-u", str(ROOT/"research_hpc.py"), "analyse", "--campaign", str(folder), "--collect"]
                analysis_code = run_step(command, logs/"analysis")
                summary["analysis_exit_code"] = analysis_code
                failed = failed or analysis_code!=0
                figures = ROOT/"figures"/folder.name
                figures.mkdir(parents=True, exist_ok=True)
                for image in (folder/"analysis").glob("*.png"):
                    shutil.copyfile(image, figures/image.name)
            except BaseException as exc:
                summary["analysis_error"] = str(exc)
                failed = True
        states = []
        for file in (job/"workers").glob("*/*.json"):
            try: states.append(read(file))
            except (ValueError, OSError): pass
        summary["completed_worker_shards"] = sum(state.get("state")=="COMPLETE" for state in states)
        summary["worker_shards_recorded"] = len(states)
        summary["status"] = "FAILED_OR_INCOMPLETE" if failed else "COMPLETE"
        summary["exit_code"] = int(failed)
        summary["finished_utc"] = stamp()
        summary["reports"] = {"json": str(ROOT/"reports/latest_summary.json"), "markdown": str(ROOT/"reports/latest_report.md")}
        archive = ROOT/"results/archives"/("job_"+job_id+".zip")
        summary["job_result_archive"] = str(archive)
        dump(job/"summary.json", summary)
        dump(ROOT/"reports/latest_summary.json", summary)
        report = ["# Polariton HOPS job report", "", f"Job: {job_id} ({summary.get('job_name')})", f"Status: {summary['status']}",
                  f"Project: {ROOT}", f"Configuration: {config}", f"Campaign: {summary.get('campaign', 'reference job')}",
                  f"Completed experiments: {summary.get('completed_experiments', 0)}/{summary.get('planned_experiments', 0)}",
                  f"Worker shards completed: {summary['completed_worker_shards']}/{summary['worker_shards_recorded']}", "",
                  "## Interpretation", "This report records execution and completion only. It does not certify HOPS convergence or paper reproduction.",
                  "Partial results remain saved. Failed or unfinished trajectories must retain their original seeds.", "", "## Stages"]
        report += ["- "+json.dumps(stage) for stage in summary["stages"]]
        if summary.get("error"): report += ["", "## Failure", summary["error"]]
        text = "\n".join(report)+"\n"
        (job/"report.md").write_text(text, encoding="utf-8")
        (ROOT/"reports/latest_report.md").write_text(text, encoding="utf-8")
        try:
            archive.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive, "w", ZIP_DEFLATED) as z:
                for base in (job, logs):
                    for file in sorted(base.rglob("*")):
                        if file.is_file(): z.write(file, file.relative_to(ROOT))
                for suffix in ("out", "err"):
                    main_log = ROOT/"logs"/(str(summary.get("job_name"))+"_"+job_id+"."+suffix)
                    if main_log.is_file(): z.write(main_log, main_log.relative_to(ROOT))
                if folder:
                    candidates = sorted((ROOT/"results/archives").glob("hops_hpc_results_"+folder.name+"_*.zip"))
                    if candidates: z.write(candidates[-1], "scientific_results/"+candidates[-1].name)
        except Exception as exc:
            failed = True
            summary.update(status="FAILED_OR_INCOMPLETE", exit_code=1, archive_error=str(exc))
            dump(job/"summary.json", summary)
            dump(ROOT/"reports/latest_summary.json", summary)
            with (ROOT/"reports/latest_report.md").open("a", encoding="utf-8") as stream:
                stream.write("\nArchive failed: "+str(exc)+"\n")
        if lock_stream is not None: lock_stream.close()
        print("FINAL REPORT:", ROOT/"reports/latest_report.md", "EXIT:", int(failed), flush=True)
    return int(failed)


if __name__=="__main__":
    raise SystemExit(main())
