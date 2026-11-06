"""Standard-library helpers shared by Slurm workers and local synthetic tests."""
from pathlib import Path
import json
import math
import os
import re


def memory_mib(value):
    text = str(value).strip().upper()
    match = re.fullmatch(r"([0-9.]+)([KMGT]?)(?:I?B)?", text)
    if not match:
        raise ValueError("Unsupported Slurm memory value: " + text)
    scale = {"": 1, "K": 1/1024, "M": 1, "G": 1024, "T": 1024**2}[match[2]]
    return int(float(match[1])*scale)


def time_seconds(value):
    text = str(value).strip()
    if text.upper() in ("UNLIMITED", "INFINITE"):
        return None
    if text.isdigit():
        return int(text)*60
    days = 0
    if "-" in text:
        first, text = text.split("-", 1)
        days = int(first)
    parts = [int(p) for p in text.split(":")]
    if len(parts)==3: hours, minutes, seconds = parts
    elif len(parts)==2: hours, minutes, seconds = 0, parts[0], parts[1]
    else: raise ValueError("Unsupported Slurm time: " + str(value))
    return days*86400+hours*3600+minutes*60+seconds


def worker_capacity(cpu_slots, memory_budget_mib, reserve_mib, per_worker_mib, remaining_tasks, configured_maximum):
    if per_worker_mib<=0 or cpu_slots<1:
        raise ValueError("Invalid CPU or memory budget")
    available = memory_budget_mib-reserve_mib
    by_memory = math.floor(available/per_worker_mib)
    if by_memory<1 and remaining_tasks:
        raise ValueError("The allocation cannot fit one worker's configured memory budget")
    return max(0, min(cpu_slots, by_memory, remaining_tasks, configured_maximum))


def claim_once(directory, task_id, rank):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/("task_%06d.claim" % task_id)
    try:
        descriptor = os.open(target, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"task_id": task_id, "rank": rank}, stream)
    return True


def task_inventory(folder, tasks):
    folder = Path(folder)
    rows = []
    for task in tasks:
        path = folder/"trajectories"/("task_%06d.npz" % task["id"])
        state = "COMPLETE" if path.is_file() else "LOCKED" if path.with_suffix(".lock").is_file() else "FAILED" if path.with_suffix(".failure.json").is_file() else "PENDING"
        rows.append({"id": task["id"], "stage": task["stage"], "case": task["case"], "seed": task["seed"], "state": state})
    return rows


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes this project: " + str(path))
    return path
