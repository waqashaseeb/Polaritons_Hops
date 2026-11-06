"""Inspect queue MaxTime and submit Batch; requires the actual cluster's Slurm."""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from research_hpc import ROOT, configure_workspace, read, dump, stamp
from queue_policy import CPU_QUEUES, query_partition, choose_queue, slurm_time


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validation.json")
    parser.add_argument("--hours", type=float, default=23.)
    parser.add_argument("--expected-hours", type=float, help="Known runtime budget: choose the shortest inspected CPU queue that fits it")
    parser.add_argument("--name", default="polHOPS_N4_d2d3")
    parser.add_argument("--mode", choices=["hops", "references"], default="hops")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name): raise ValueError("Use a simple job name")
    configuration = (ROOT/args.config).resolve()
    if not configuration.is_relative_to(ROOT) or not configuration.is_file(): raise ValueError("Configuration must be inside this project")
    if not read(configuration).get("enabled", False): raise ValueError("This configuration is a disabled template")
    hours = args.expected_hours if args.expected_hours is not None else args.hours
    if hours<=0: raise ValueError("Walltime must be positive")
    seconds = int(hours*3600)
    partitions, query_errors = {}, {}
    for name in CPU_QUEUES:
        try: partitions[name] = query_partition(name)
        except (ValueError, RuntimeError) as exc: query_errors[name] = str(exc)
    chosen = choose_queue(partitions, seconds, uncertain=args.expected_hours is None)
    dump(ROOT/"outputs/queue_limits.json", {"partitions": partitions, "query_errors": query_errors, "verified_on_cluster": True})
    flags = ["sbatch", "--parsable", "--partition="+chosen, "--time="+slurm_time(seconds), "--job-name="+args.name,
             "--export=ALL,HOPS_CONFIG="+str(configuration.relative_to(ROOT))+",HOPS_MODE="+args.mode+",HOPS_ALLOCATION_SECONDS="+str(seconds)]
    if args.mode=="references": flags += ["--ntasks=1", "--mem=8G"]
    command = flags+[str(ROOT/"Batch")]
    print(shlex.join(command), flush=True)
    if args.dry_run: return
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit(): raise RuntimeError("Unexpected sbatch response: "+result.stdout)
    dump(ROOT/"outputs/submissions"/(job_id+".json"), {"job_id": job_id, "name": args.name, "submitted_utc": stamp(), "command": command})
    print("SLURM JOB ID:", job_id)
    print("squeue -u tariq.qayyum")
    print("sacct -j "+job_id+" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%30")
    print("tail -f "+str(ROOT/"logs"/(args.name+"_"+job_id+".out")))
    print("tail -f "+str(ROOT/"logs"/(args.name+"_"+job_id+".err")))


if __name__=="__main__": main()
