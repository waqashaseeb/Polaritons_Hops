# Polariton HOPS: resumable university-HPC workflow

This project compares the saved provisional polariton model with Pauli dynamics
using the user's pinned CPU MesoHOPS implementation. It includes the actual
saved Hamiltonian and bath inputs, source/license checksums, Slurm entrypoints,
parallel trajectory scheduling, progress monitoring and partial-result recovery.

The computational unit is an independent stochastic HOPS trajectory. Different
CPUs run different trajectories. One hierarchy is not distributed across 40 CPUs.
No CUDA backend, GPU training, multi-excitation annihilation model or automatic
paper-reproduction claim is included.

## Project location and allocation

The user's final path requirement is used consistently:

    /home/tariq.qayyum/hpc/polariton_hops/

This is a separate project; older `waqas` calculations are not accessed or moved.
Runtime paths, source, environments, caches, temporary files, logs, figures,
reports and checkpoints stay inside this directory. Historical input-provenance
strings describe earlier files; they are not output destinations.

`Batch` requests **longq, 23 hours, one node, 40 CPU slots, one CPU per task,
480G RAM, and no GPU**, with job name **polHOPS_N4_d2d3**. RAM fits within the
roughly 512 GB CPU-node hardware previously reported, but actual partition and
account allocation rules must be checked on the private cluster.

With the present conservative RAM budgets, a 480G allocation runs at most
19 simultaneous 24G pilot workers, or three 128G depth-3 workers. Other slots
remain idle in memory-bound stages. The program prints the actual worker count.
An optional Slurm-array route spreads trajectories across more nodes; see below.

The total runtime of the new workflow has not been measured. A 23-hour request
is a limit, not a completion estimate. Finished trajectories are saved; an
unfinished allocation exits nonzero and can resume with the same configuration.

## Upload from the Mac

Extract the source bundle locally, for example to `~/Downloads/polariton_hops`.
Create the project and the log directory before sbatch opens its output files:

```bash
ssh tariq.qayyum@10.90.24.15 \
  'mkdir -p /home/tariq.qayyum/hpc/polariton_hops/logs'

rsync -az --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.tools/' \
  --exclude='.conda-git/' \
  --exclude='.config/' \
  --exclude='.locks/' \
  --exclude='runtime/' \
  --exclude='logs/' \
  --exclude='results/' \
  --exclude='outputs/' \
  --exclude='figures/' \
  --exclude='reports/' \
  --exclude='checkpoints/' \
  --exclude='__pycache__/' \
  "$HOME/Downloads/polariton_hops/" \
  tariq.qayyum@10.90.24.15:/home/tariq.qayyum/hpc/polariton_hops/
```

This command does not transfer existing results/checkpoints and does not use
`--delete`. Transfer results separately and explicitly when needed. Avoid
changing source files while a campaign is running: input/source hashes identify
the scientific calculation and changed code requires a separate campaign.

## Preflight and submit

```bash
ssh tariq.qayyum@10.90.24.15
cd /home/tariq.qayyum/hpc/polariton_hops
module load slurm
module load anaconda3
bash scripts/preflight.sh
sbatch /home/tariq.qayyum/hpc/polariton_hops/Batch
```

The preflight loads `anaconda3`, creates a project-local Python 3.11 environment,
checks shell/Python syntax and pinned checksums, reads the real queue limits and
uses `sbatch --test-only`. It does not submit a job. It refuses the default request
if longq's MaxTime or configured memory limit is too small. Do not infer those
limits from the queue names alone.

The shared `/cm/shared/apps/anaconda3` installation is read-only input. No pip
or conda install/update targets it. If its Python is not 3.11, the bootstrap can
create a private interpreter under `.tools/python311`, with local package caches
and external environment registration disabled, and then creates `.venv`.
Older Conda versions that cannot disable registration stop safely: provide a
Python 3.11 interpreter through `HOPS_BOOTSTRAP_PYTHON` instead. HPC simulation
does not require Git because the pinned core is already included.

For automatic queue selection after a runtime estimate is available, use:

```bash
.venv/bin/python submit_project.py --expected-hours 2 --name polHOPS_N4_short
```

It chooses the shortest inspected CPU queue that fits that declared request.
Without `--expected-hours`, it uses longq and the 23-hour request. It prints the
real job ID returned by sbatch and exact monitoring commands. No real job ID is
included in this bundle because no private-cluster submission was made here.

## Calculation choices

| Profile | Configuration | Planned trajectories | Purpose |
|---|---|---:|---|
| Default validation | configs/validation.json | 66 | Two resource probes, 32 pilot trajectories, four controls with eight paired seeds each |
| Larger sampling | configs/large_sampling.json | 514 | 256 pilot trajectories and four controls with 64 paired seeds each |
| Parameter/size controls | configs/parameter_scan.json | 396 | Six supplied-model cases, including bare N1 and available N2..4 cavity cases |
| Conditional references | configs/paper_reference_grid.json | 0 HOPS | Pauli coupling/size scans, energies, weights and overlaps |

Validation/pilot controls span **20 fs**, with 0.5 fs saved spacing. Larger
sampling improves stochastic precision; it does not validate a longer time range.
Parameter scans are short-time controls, not long-time transfer-rate results.

```bash
sbatch --job-name=polHOPS_N4_MC256 \
  --export=ALL,HOPS_CONFIG=configs/large_sampling.json \
  /home/tariq.qayyum/hpc/polariton_hops/Batch

sbatch --job-name=polHOPS_size_gscan \
  --export=ALL,HOPS_CONFIG=configs/parameter_scan.json \
  /home/tariq.qayyum/hpc/polariton_hops/Batch
```

Run validation first. Different configurations use different campaign identities.
Resubmitting the same configuration preserves completed trajectories and their
original seeds. Reruns with the same seeds are not independent data.

For the CPU-only conditional references, with a declared three-hour budget:

```bash
.venv/bin/python submit_project.py --mode references --expected-hours 3 --name polHOPS_Pauli_grid
```

## More nodes: optional trajectory arrays

After the local environment is ready, this alternative submits preparation,
per-trajectory CPU arrays and an analysis job. It can spread up to 40 concurrent
trajectories across CPU nodes when account/queue/memory limits permit:

```bash
.venv/bin/python submit_hpc.py \
  --config configs/large_sampling.json --sequence validation --submit
```

Use either the single-allocation Batch route or this array route for a particular
campaign at a time. Do not submit both simultaneously for the same campaign.
The single-allocation route always propagates required failures through Batch's
exit status. In the array route, inspect both worker jobs and the analysis report;
submission success alone is not execution success.

## Monitor, reports and recovery

See `docs/HPC_RUNBOOK.md` for copyable commands. The main report paths are:

    reports/latest_summary.json
    reports/latest_report.md
    outputs/jobs/JOB_ID/summary.json
    outputs/jobs/JOB_ID/report.md

Partial numerical trajectories and analysis are under `results/campaigns/`.
Figures are also collected in `figures/CAMPAIGN/`. Result archives are under
`results/archives/`, including `job_JOB_ID.zip` and the scientific archive.

No task failure is hidden by replacing its seed. No other project's files are
deleted. No script cancels a Slurm job. Recovery inspects only an explicitly
identified ended job with a matching project record before clearing its locks.

## GitHub source

Repository: https://github.com/waqashaseeb/Polaritons_Hops

The repository contains source, documentation, frozen inputs and local-check
records. Generated environments, caches, checkpoints and runtime results are
excluded. Git attributes preserve byte-for-byte checksums on checkout.

To clone to the local source-directory name used by the upload commands:

```bash
git clone https://github.com/waqashaseeb/Polaritons_Hops.git "$HOME/Downloads/polariton_hops"
```

Git refuses that command if the destination is already nonempty; review the
existing folder rather than overwriting it. The HPC does not need Git for the
rsync-and-sbatch route. The optional `publish_github.py` helper is for creating
a separate new private repository, not for deploying this existing repository;
it refuses an existing repository or origin and never force-pushes.

## What has actually been checked

`docs/LOCAL_VALIDATION.json` records real compilation, shell parsing, directory/
path checks and checksum verification. `tests/test_workflow.py` uses synthetic
scheduler/filesystem placeholders to test failure, resume, resource and queue
logic; it does not create scientific populations. `docs/SAVED_INPUT_AUDIT.json`
records real, cheap checks of the saved Hamiltonian/Pauli generator and analytic
bath line shapes. No new HOPS trajectory was propagated here.

The saved generator is reconstructed within about 2.7e-13 relative error.
Both frozen fits pass the declared line-shape tolerance at 20 fs, while both
fail it at 12 ps. Longer templates therefore remain disabled and bath-gated.
Full quantitative reproduction still needs author inputs, missing N=5..7 models
and verified preparation/fit/spectral conventions. See `docs/PAPER_TARGETS.md`.
