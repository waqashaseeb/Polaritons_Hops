"""Private-cluster checks: real queue limits and sbatch --test-only, not a submission."""
import json
import subprocess
from research_hpc import ROOT, configure_workspace, dump
from queue_policy import CPU_QUEUES, query_partition, choose_queue
from run_support import memory_mib


def main():
    configure_workspace()
    partitions, errors = {}, []
    for queue in CPU_QUEUES:
        try: partitions[queue] = query_partition(queue)
        except (ValueError, RuntimeError) as exc: errors.append(str(exc))
    chosen = choose_queue(partitions, 23*3600, uncertain=True)
    maximum_memory = partitions[chosen].get("MaxMemPerNode")
    if maximum_memory and maximum_memory.isdigit() and int(maximum_memory)>0 and int(maximum_memory)<memory_mib("480G"):
        raise ValueError("longq's configured MaxMemPerNode is below 480G; adapt the request and concurrency before submission")
    test = subprocess.run(["sbatch", "--test-only", str(ROOT/"Batch")], capture_output=True, text=True)
    result = {"status": "PASS" if test.returncode==0 else "FAIL", "queues": partitions,
              "queue_query_notes": errors, "default_walltime_seconds": 82800, "default_memory": "480G",
              "sbatch_test_only_exit": test.returncode, "stdout": test.stdout, "stderr": test.stderr,
              "job_submitted": False, "end_to_end_runtime_established": False}
    dump(ROOT/"reports/cluster_preflight.json", result)
    (ROOT/"reports/cluster_preflight.md").write_text("# Cluster preflight\n\nStatus: "+result["status"]+"\n\nThe actual longq MaxTime was checked for a 23-hour request. sbatch --test-only did not submit a job. This does not measure the calculation runtime.\n\n"+test.stdout+test.stderr)
    print(json.dumps(result, indent=2))
    if test.returncode: raise SystemExit(test.returncode)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        configure_workspace()
        dump(ROOT/"reports/cluster_preflight.json", {"status":"FAIL", "error":str(exc), "job_submitted":False})
        (ROOT/"reports/cluster_preflight.md").write_text("# Cluster preflight failed\n\n"+str(exc)+"\n\nNo job was submitted.\n")
        raise
