"""One srun rank claims and executes independent trajectory tasks, one at a time."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from research_hpc import ROOT, read, dump, configure_workspace
from run_support import claim_once, inside, time_seconds


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job-directory", type=Path, required=True)
    args = parser.parse_args()
    folder = inside(ROOT, args.campaign)
    job = inside(ROOT, args.job_directory)
    meta = read(folder/"campaign.json")
    selected = [t for t in meta["tasks"] if t["stage"]==args.stage]
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    for key, name in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"), ("TMPDIR", "tmp")):
        value = ROOT/"runtime"/(name+"_"+os.environ["SLURM_JOB_ID"]+"_rank_"+str(rank))
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    stage_dir = job/"workers"/args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    status_path = stage_dir/("rank_%03d.json" % rank)
    claim_dir = job/"claims"/args.stage
    logs = ROOT/"logs"/os.environ["SLURM_JOB_ID"]/args.stage
    logs.mkdir(parents=True, exist_ok=True)
    deadline = int(os.environ["HOPS_DEADLINE_EPOCH"])
    spec = meta["config"]["stages"][args.stage]
    task_limit = time_seconds(spec["resources"]["time"])
    state = {"rank": rank, "stage": args.stage, "state": "RUNNING", "completed_tasks": [], "failed_tasks": [],
             "current_task_id": None, "synthetic": False}
    dump(status_path, state)
    child = [None]
    stopped = [False]
    def terminate(signum, frame):
        stopped[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].terminate()  # Only the child created by this rank, never another job.
    signal.signal(signal.SIGTERM, terminate)
    try:
        for position, task in enumerate(selected):
            target = folder/"trajectories"/("task_%06d.npz" % task["id"])
            if target.is_file():
                continue
            remaining = deadline-time.time()-240
            if stopped[0] or (job/"stop_requested").exists() or remaining<120:
                state["state"] = "WALLTIME_STOPPED"
                break
            if not claim_once(claim_dir, task["id"], rank):
                continue
            state["current_task_id"] = task["id"]
            dump(status_path, state)
            log_stem = logs/("task_%06d_seed_%d" % (task["id"], task["seed"]))
            command = [sys.executable, "-u", str(ROOT/"research_hpc.py"), "worker", "--campaign", str(folder),
                       "--stage", args.stage, "--position", str(position)]
            print(f"START rank={rank} stage={args.stage} task={task['id']} seed={task['seed']}", flush=True)
            with log_stem.with_suffix(".out").open("a", encoding="utf-8") as out, log_stem.with_suffix(".err").open("a", encoding="utf-8") as err:
                child[0] = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
                try:
                    code = child[0].wait(timeout=min(remaining, task_limit or remaining))
                except subprocess.TimeoutExpired:
                    child[0].terminate()
                    try: child[0].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child[0].kill(); child[0].wait()
                    code = 124
                    err.write("Required task exceeded its runtime budget; retain this seed and resume after diagnosis.\n")
            child[0] = None
            state["current_task_id"] = None
            if code==0 and target.is_file():
                state["completed_tasks"].append(task["id"])
                print(f"COMPLETE rank={rank} task={task['id']}", flush=True)
            else:
                state["failed_tasks"].append({"task_id": task["id"], "exit_code": code, "stderr": str(log_stem.with_suffix('.err'))})
                print(f"FAILED rank={rank} task={task['id']} exit={code}", flush=True)
            dump(status_path, state)
        if state["state"]=="RUNNING":
            state["state"] = "FAILED" if state["failed_tasks"] else "COMPLETE"
    finally:
        dump(status_path, state)
    return 0 if state["state"]=="COMPLETE" and not state["failed_tasks"] else 1


if __name__=="__main__":
    raise SystemExit(main())
