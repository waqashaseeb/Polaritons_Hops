# Exact commands for this university cluster

All project paths below are inside `/home/tariq.qayyum/hpc/polariton_hops/`.
The final `hpc/<PROJECT_NAME>` requirement was chosen over earlier generic and
waqas directory examples. Existing projects are untouched.

## Submit and monitor

```bash
ssh tariq.qayyum@10.90.24.15
cd /home/tariq.qayyum/hpc/polariton_hops
module load slurm
module load anaconda3
bash scripts/preflight.sh
sbatch /home/tariq.qayyum/hpc/polariton_hops/Batch
```

Replace JOB_ID with the numeric ID actually printed by sbatch:

```bash
squeue -u tariq.qayyum
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%30
tail -f /home/tariq.qayyum/hpc/polariton_hops/logs/polHOPS_N4_d2d3_JOB_ID.out
tail -f /home/tariq.qayyum/hpc/polariton_hops/logs/polHOPS_N4_d2d3_JOB_ID.err
```

For the worker steps, rather than only the low-CPU batch coordinator:

```bash
sstat --allsteps -j JOB_ID --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize
sstat -j JOB_ID.batch --format=JobID,AveCPU,MaxRSS,MaxVMSize
```

Accounting availability and sampling precision depend on cluster configuration.
Do not poll the Slurm controller in a tight loop.

## Completed workers/experiments and recent progress

```bash
/home/tariq.qayyum/hpc/polariton_hops/.venv/bin/python \
  /home/tariq.qayyum/hpc/polariton_hops/monitor.py --job JOB_ID

grep -RnhE 'START|COMPLETE|FAILED|STAGE' \
  /home/tariq.qayyum/hpc/polariton_hops/logs/JOB_ID/ | tail -n 40

grep -RniE 'error|fatal|failed|traceback|oom|killed|timeout' \
  /home/tariq.qayyum/hpc/polariton_hops/logs/
```

`monitor.py` reports completed experiment counts, completed rank/shard counts,
current tasks and failed worker log paths. Each task has its own `.out` and
`.err` file under `logs/JOB_ID/STAGE/`. Rank-level srun logs are separate.
The backend's detailed trajectory log is inside its campaign `task_logs/`.

## Results and reports

| Path relative to the project | Contents |
|---|---|
| results/campaigns/CAMPAIGN/trajectories/ | Atomic completed raw trajectories and failure/lock records |
| results/campaigns/CAMPAIGN/analysis/ | Numerical means, SEs, paired controls and plots |
| figures/CAMPAIGN/ | Figure copies from ensemble analysis |
| outputs/jobs/JOB_ID/ | Rank states, claims, per-job summary and report |
| reports/latest_summary.json | Latest machine-readable completion report |
| reports/latest_report.md | Latest human-readable report |
| reports/cluster_preflight.json | Actual queue and scheduler test-only evidence |
| results/archives/job_JOB_ID.zip | Job reports/logs and available scientific archive |
| checkpoints/ | Job-level checkpoint namespace; completed trajectory NPZs remain the numerical restart unit |

No mid-trajectory wavefunction restart is implemented. Interrupted trajectories
restart with their original seeds; successfully saved trajectories are skipped.

## Recover a failed or timed-out Batch

First inspect squeue/sacct. Then, for the explicitly identified ended project job:

```bash
cd /home/tariq.qayyum/hpc/polariton_hops
.venv/bin/python recover_project.py --job JOB_ID
sbatch /home/tariq.qayyum/hpc/polariton_hops/Batch
```

Use the original `--export=ALL,HOPS_CONFIG=...` argument if resuming a nondefault
profile. The helper refuses active/unverifiable/foreign jobs. It removes only
matching stale locks, never completed data, and never cancels any job. If source
or scientific settings changed, the new campaign intentionally has a different
identity rather than silently mixing results.

## Download results separately to the Mac

Inspect what would transfer first:

```bash
rsync -azn --progress \
  tariq.qayyum@10.90.24.15:/home/tariq.qayyum/hpc/polariton_hops/results/archives/ \
  "$HOME/Downloads/polariton_hops_results/"
```

After reviewing the dry run, use `-az` in place of `-azn`. This explicit result
transfer is separate from source upload. Neither command uses `--delete`.
