"""Read saved progress; no numerical propagation or remote job cancellation."""
from pathlib import Path
import argparse
import json
from research_hpc import ROOT, configure_workspace, read
from run_support import task_inventory


def main():
    configure_workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not args.job.isdigit(): raise ValueError("Use the numeric job ID")
    directory = ROOT/"outputs/jobs"/args.job
    source = directory/"summary.json" if (directory/"summary.json").is_file() else directory/"run_record.json"
    record = read(source)
    report = {"job_id": args.job, "status": record.get("status"), "current_stage": record.get("current_stage"),
              "completed_worker_shards": 0, "failed_worker_shards": [], "workers": []}
    for file in (directory/"workers").glob("*/*.json"):
        state = read(file)
        report["workers"].append({k: state.get(k) for k in ["rank", "stage", "state", "current_task_id"]})
        if state.get("state")=="COMPLETE": report["completed_worker_shards"] += 1
        if state.get("failed_tasks"): report["failed_worker_shards"].append({"state_file": str(file), "failures": state["failed_tasks"]})
    if record.get("campaign"):
        folder = Path(record["campaign"]).resolve()
        if not folder.is_relative_to(ROOT): raise ValueError("Recorded campaign escapes project")
        if (folder/"campaign.json").is_file():
            meta = read(folder/"campaign.json")
            inventory = task_inventory(folder, meta["tasks"])
            report["completed_experiments"] = sum(row["state"]=="COMPLETE" for row in inventory)
            report["planned_experiments"] = len(inventory)
            report["failed_or_locked_experiments"] = [row for row in inventory if row["state"] in ("FAILED", "LOCKED")]
            report["partial_results"] = str(folder/"trajectories")
    print(json.dumps(report, indent=2))


if __name__=="__main__": main()
