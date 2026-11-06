"""Synthetic scheduler/filesystem tests. They do not generate scientific results."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_support import memory_mib, time_seconds, worker_capacity, claim_once, inside
from queue_policy import parse_partition, choose_queue
import allocation_runner as allocation


class Helpers(unittest.TestCase):
    def test_resource_bounds(self):
        self.assertEqual(memory_mib("480G"), 491520)
        self.assertEqual(worker_capacity(40, 480*1024, 24*1024, 24*1024, 32, 40), 19)
        self.assertEqual(worker_capacity(40, 480*1024, 24*1024, 128*1024, 8, 40), 3)
        with self.assertRaises(ValueError): worker_capacity(40, 64*1024, 24*1024, 128*1024, 8, 40)

    def test_queue_limits(self):
        self.assertEqual(time_seconds("1-01:02:03"), 90123)
        with self.assertRaises(ValueError): time_seconds("NOT_SET")
        queues = {q: parse_partition(f"PartitionName={q} State=UP MaxTime={t}") for q,t in [("shortq","02:00:00"),("mediumq","12:00:00"),("longq","1-00:00:00")]}
        self.assertEqual(choose_queue(queues, 3600, False), "shortq")
        self.assertEqual(choose_queue(queues, 23*3600, True), "longq")
        with self.assertRaises(ValueError): choose_queue(queues, 48*3600, True)

    def test_atomic_claim_and_isolation(self):
        base = ROOT/"outputs/synthetic_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            with ThreadPoolExecutor(max_workers=12) as pool:
                claims = list(pool.map(lambda rank: claim_once(tmp, 7, rank), range(24)))
            self.assertEqual(sum(claims), 1)
            with self.assertRaises(ValueError): inside(tmp, Path(tmp).parent/"another_project")


class AllocationContract(unittest.TestCase):
    def test_failure_resume_and_reports(self):
        base = ROOT/"outputs/synthetic_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root = Path(tmp); folder = root/"results/campaigns/synthetic"
            (folder/"trajectories").mkdir(parents=True)
            (folder/"analysis").mkdir()
            config = root/"config.json"
            cfg = {"default_sequence":["pilot"], "stages":{"pilot":{"resources":{"memory":"24G"},"maximum_parallel":40}}}
            config.write_text(json.dumps(cfg))
            tasks = [{"id":i,"stage":"pilot","case":"SYNTHETIC_NOT_PHYSICAL","seed":100+i} for i in range(3)]
            meta = {"tasks":tasks,"config":cfg}
            fail = [True]
            attempted = []
            def fake_srun(command, log_base):
                if "prepare" in command:
                    (folder/"campaign.json").write_text(json.dumps(meta))
                elif any(str(value).endswith("worker_rank.py") for value in command):
                    for task in tasks:
                        path = folder/"trajectories"/("task_%06d.npz" % task["id"])
                        if path.is_file(): continue
                        attempted.append(task["id"])
                        if fail[0] and task["id"]==1: continue
                        path.write_bytes(b"SYNTHETIC ORCHESTRATION PLACEHOLDER; NOT SCIENTIFIC DATA")
                    return 1 if fail[0] else 0
                return 0
            environment = {"SLURM_JOB_ID":"91001","SLURM_JOB_NAME":"synthetic_test","SLURM_NTASKS":"40", "SLURM_MEM_PER_NODE":"491520","HOPS_DEADLINE_EPOCH":"4102444800"}
            with patch.object(allocation,"ROOT",root), patch.object(allocation,"configure_workspace",lambda:None), \
                 patch.object(allocation,"campaign_identity",lambda c:(folder,cfg,{},"synthetic-signature")), \
                 patch.object(allocation,"run_step",fake_srun), patch.dict(os.environ,environment):
                for job_id, expected in [("91001",1),("91002",0)]:
                    job = root/"outputs/jobs"/job_id
                    os.environ["SLURM_JOB_ID"] = job_id
                    argv = ["allocation_runner.py","--config",str(config),"--job-directory",str(job)]
                    with patch.object(sys,"argv",argv): self.assertEqual(allocation.main(),expected)
                    summary = json.loads((job/"summary.json").read_text())
                    self.assertEqual(summary["exit_code"],expected)
                    self.assertTrue((root/"reports/latest_report.md").is_file())
                    fail[0] = False
            self.assertEqual(attempted,[0,1,2,1])
            summary = json.loads((root/"reports/latest_summary.json").read_text())
            self.assertEqual(summary["completed_experiments"],3)
            self.assertFalse(summary["paper_reproduction_verified"])


if __name__=="__main__": unittest.main()
