# Sources and preserved inputs

The simulation core, saved backend, bath tools, source license and provisional
physical input were copied from the package inside the user's attached
`hops_resource_probe_depth2_20261102T102011_121026Z.zip`. The b16 coefficients
were copied from its completed campaign. See `inputs/source_archives.json` and
the package SHA256 manifest. No source is downloaded when the jobs run.

New orchestration code is a separate layer. The pinned vendor and saved adapter
remain byte-for-byte copies. New orchestration success must be established on
the user's cluster before scientific use.

Primary technical references:

- Theory preprint: https://arxiv.org/html/2504.02066v2
- Published article DOI: https://doi.org/10.1063/5.0273374
- MesoHOPS authors' repository: https://github.com/MesoscienceLab/mesohops
- Slurm job arrays: https://slurm.schedmd.com/job_array.html
- Slurm sbatch options and dependencies: https://slurm.schedmd.com/sbatch.html

The MesoHOPS authors recommend independent Slurm trajectory arrays for large
calculations and emphasize the need for convergence scans. This package retains
the user's pinned version; current repository installation instructions are not
used to upgrade it.
