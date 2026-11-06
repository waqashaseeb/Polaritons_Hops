"""Read real Slurm queue limits; never infer private-cluster limits from names."""
import re
import subprocess
from run_support import time_seconds

CPU_QUEUES = ("shortq", "mediumq", "longq")


def parse_partition(text):
    fields = dict(re.findall(r"(\w+)=([^\s]+)", text))
    if "PartitionName" not in fields or "MaxTime" not in fields:
        raise ValueError("Could not read the partition name and MaxTime")
    fields["maximum_seconds"] = time_seconds(fields["MaxTime"])
    return fields


def query_partition(queue):
    result = subprocess.run(["scontrol", "-o", "show", "partition", queue], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Cannot inspect queue "+queue+": "+result.stderr.strip())
    return parse_partition(result.stdout)


def choose_queue(partitions, requested_seconds, uncertain=True):
    names = ["longq"] if uncertain else list(CPU_QUEUES)
    for name in names:
        item = partitions.get(name)
        if not item or item.get("State", "UP")!="UP":
            continue
        maximum = item["maximum_seconds"]
        if maximum is None or requested_seconds<=maximum:
            return name
    raise ValueError("No eligible inspected CPU queue fits the requested walltime; reduce the request or check actual queue limits")


def slurm_time(seconds):
    days, remain = divmod(int(seconds), 86400)
    hours, remain = divmod(remain, 3600)
    minutes, seconds = divmod(remain, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return (str(days)+"-" if days else "")+clock
