"""Clear stale locks only for an explicitly identified, ended job of this project."""
import argparse
import subprocess
from pathlib import Path
from research_hpc import ROOT, configure_workspace, read
from run_support import inside


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not args.job.isdigit(): raise ValueError("Numeric job ID required")
    job_directory = ROOT/"outputs/jobs"/args.job
    record_path = job_directory/"summary.json" if (job_directory/"summary.json").is_file() else job_directory/"run_record.json"
    record = read(record_path)
    if record.get("job_id")!=args.job or Path(record.get("project_root", "")).resolve()!=ROOT:
        raise ValueError("Job ownership is not established for this project")
    live = subprocess.run(["squeue", "-h", "-j", args.job, "-o", "%T"], capture_output=True, text=True)
    if live.stdout.strip(): raise ValueError("This job still has an active scheduler state; no lock was changed")
    accounting = subprocess.run(["sacct", "-n", "-X", "-j", args.job, "--format=JobIDRaw,State", "-P"], capture_output=True, text=True)
    states = []
    for line in accounting.stdout.splitlines():
        values = line.split("|")
        if len(values)>1 and values[0].strip()==args.job: states.append(values[1].strip().split()[0].rstrip("+"))
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    if accounting.returncode or not states or any(s not in terminal for s in states):
        raise ValueError("Cannot verify an ended job with sacct. No lock was changed; inspect accounting manually")
    if not record.get("campaign"):
        print("No campaign locks: failure occurred before campaign preparation")
        return
    folder = inside(ROOT, record["campaign"])
    locks = list((folder/"trajectories").glob("*.lock"))+[folder/"prepare.lock"]
    changed = []
    for path in locks:
        if not path.is_file(): continue
        data = read(path)
        if str(data.get("slurm_job_id"))==args.job:
            path.unlink(); changed.append(str(path))
    print("Cleared locks belonging to ended project job:", *changed, sep="\n")
    print("Completed trajectories were preserved. Resubmit the same Batch/configuration.")


if __name__=="__main__": main()
