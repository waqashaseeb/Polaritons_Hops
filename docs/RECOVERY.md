# Recovery

Use the project-specific commands in [HPC_RUNBOOK.md](HPC_RUNBOOK.md).
No script cancels jobs. The recovery helper requires an explicit job ID,
a matching project record and a terminal sacct state. It removes only that
ended job's stale locks. Saved trajectories are preserved.

Resubmit Batch with the same configuration and original seeds. Completed
trajectories are skipped. No mid-trajectory restart is implemented.
