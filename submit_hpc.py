"""Submit CPU trajectory arrays. Uses only the standard library on login nodes.

No submission is made without --submit. Does not execute scientific code locally.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
from research_hpc import ROOT, read, dump, stamp, campaign_identity, configure_workspace


def accounting(folder):
    folder = Path(folder).resolve()
    records = []
    for file in folder.glob("submission_*.json"):
        records.extend(read(file).get("jobs", []))
    ids = sorted({str(r["job_id"]) for r in records if r.get("job_id") and str(r["job_id"]).isdigit()})
    if not ids:
        return
    command = ["sacct", "-P", "-j", ",".join(ids), "--format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList"]
    result = subprocess.run(command, text=True, capture_output=True)
    target = folder/"analysis"
    target.mkdir(exist_ok=True)
    (target/"slurm_accounting.psv").write_text(result.stdout, encoding="utf-8")
    dump(target/"slurm_accounting_status.json", {"command": command, "returncode": result.returncode, "stderr": result.stderr,
         "MaxRSS_note": "Inspect the worker job-step rows; some sites do not collect accounting or report RSS for the parent array row."})


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/validation.json")
    parser.add_argument("--profile", type=Path, default=ROOT/"configs/cluster.json")
    parser.add_argument("--sequence", choices=["validation"])
    parser.add_argument("--stage", help="Submit a single configured stage; prerequisites must already be complete or supplied through --after")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--references", action="store_true", help="Submit conditional Pauli/static reference grid, not HOPS")
    parser.add_argument("--reference-config", type=Path, default=ROOT/"configs/paper_reference_grid.json")
    parser.add_argument("--after", help="Existing successful preparation/stage job ID(s), separated by colon")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--accounting", type=Path)
    args = parser.parse_args()
    if args.accounting:
        accounting(args.accounting)
        return
    profile = read(args.profile)
    executable = Path(profile["python"])
    if not executable.is_absolute(): executable = ROOT/executable
    # Preserve the venv interpreter path: resolving its symlink would bypass
    # the venv and launch the base interpreter without the installed packages.
    executable = Path(os.path.abspath(str(executable)))
    if args.submit and not executable.is_file():
        raise FileNotFoundError("Python environment missing; run bash bootstrap.sh first: " + str(executable))
    if sum(bool(v) for v in [args.sequence, args.stage, args.prepare_only, args.references]) != 1:
        parser.error("Choose exactly one of --sequence validation, --stage NAME, --prepare-only, --references")
    folder, cfg, hashes, signature = campaign_identity(args.config.resolve())
    folder.mkdir(parents=True, exist_ok=True)
    logs = folder/"slurm_logs"
    logs.mkdir(exist_ok=True)
    jobs, commands = [], []
    mode = "SUBMITTED" if args.submit else "DRY_RUN_NO_JOBS_SUBMITTED"
    def enqueue(name, resources, script, script_args, dependency=None, array=None):
        command = ["sbatch", "--parsable", "--job-name=hops_"+name, "--chdir="+str(ROOT),
                   "--nodes=1", "--ntasks=1", "--cpus-per-task="+str(resources.get("cpus", 1)),
                   "--mem="+resources["memory"], "--time="+resources["time"],
                   "--output="+str(logs/(name+"-%A_%a.log")), "--open-mode=append"]
        for key, flag in [("partition", "--partition="), ("account", "--account="), ("qos", "--qos="), ("exclude_nodes", "--exclude=")]:
            if profile.get(key): command.append(flag+str(profile[key]))
        if dependency:
            command += ["--dependency="+dependency, "--kill-on-invalid-dep=yes"]
        if array:
            command += ["--array="+array]
        command += [str(ROOT/"slurm"/script)] + [str(v) for v in script_args]
        commands.append(command)
        print(shlex.join(command), flush=True)
        job_id = str(900000+len(jobs))
        if args.submit:
            response = subprocess.run(command, text=True, capture_output=True)
            if response.returncode:
                raise RuntimeError("sbatch rejected this job:\n"+response.stderr+"\nEarlier accepted jobs remain recorded and are not cancelled.")
            job_id = response.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError("Unexpected sbatch --parsable output: " + response.stdout)
        jobs.append({"name": name, "job_id": job_id if args.submit else None, "dependency": dependency, "command": command})
        # Persist every accepted job immediately, including partially submitted workflows.
        dump(record_path, {"mode": mode, "campaign": str(folder), "signature": signature, "jobs": jobs, "commands": commands})
        return job_id
    record_path = folder/("submission_"+stamp()+".json")
    prepare_resources = profile.get("prepare_resources", {"cpus": 1, "memory": "8G", "time": "01:00:00"})
    if args.references:
        enqueue("references", profile.get("reference_resources", {"cpus": 1, "memory": "8G", "time": "03:00:00"}),
                "references.sbatch", [ROOT, args.reference_config.resolve(), executable])
    elif args.prepare_only:
        enqueue("prepare", prepare_resources, "prepare.sbatch", [ROOT, args.config.resolve(), folder, executable],
                "afterok:"+args.after if args.after else None)
    else:
        if args.sequence:
            preparation_id = enqueue("prepare", prepare_resources, "prepare.sbatch", [ROOT, args.config.resolve(), folder, executable],
                                     "afterok:"+args.after if args.after else None)
            selected = cfg["default_sequence"]
        else:
            preparation_id = None
            selected = [args.stage]
            if not (folder/"campaign.json").is_file() and not args.after:
                raise ValueError("No prepared campaign: first use --prepare-only --submit, then --stage NAME --after PREPARE_JOB --submit")
        submitted_stages = {}
        for stage in selected:
            if stage not in cfg["stages"]:
                raise ValueError("Unknown configured stage: " + stage)
            spec = cfg["stages"][stage]
            dependencies = [preparation_id] if preparation_id else []
            dependencies += [submitted_stages[s] for s in cfg.get("prerequisites", {}).get(stage, []) if s in submitted_stages]
            if args.after and not args.sequence: dependencies += args.after.split(":")
            dependencies = list(dict.fromkeys(dependencies))
            count = len(cfg["cases"])*int(spec["trajectories"])
            if count<1: raise ValueError("Empty stage")
            maximum_parallel = int(spec.get("maximum_parallel", profile.get("maximum_parallel", 8)))
            if maximum_parallel<1: raise ValueError("maximum_parallel must be positive")
            array = f"0-{count-1}%{maximum_parallel}"
            stage_resources = {**spec["resources"], **profile.get("stage_resource_overrides", {}).get(stage, {})}
            submitted_stages[stage] = enqueue(stage, stage_resources, "worker.sbatch", [ROOT, folder, stage, executable],
                                              "afterok:"+":".join(dependencies) if dependencies else None, array)
        all_ids = ([preparation_id] if preparation_id else []) + list(submitted_stages.values())
        enqueue("analysis", profile.get("analysis_resources", {"cpus": 1, "memory": "8G", "time": "01:00:00"}),
                "analyse.sbatch", [ROOT, folder, executable], "afterany:"+":".join(all_ids))
    print("\nMode:", mode, "\nCampaign:", folder, "\nSubmission record:", record_path, flush=True)
    if args.submit:
        print("Monitor: squeue -u $USER\nSuccessful trajectories are preserved on repeat submission. Failed/interrupted trajectories restart with their original seed.", flush=True)


if __name__ == "__main__":
    main()
