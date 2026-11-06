# Connect the calculations to the theory paper

Target: Rosenkampff and Pullerits, J. Chem. Phys. 163, 044305 (2025),
DOI 10.1063/5.0273374. Public preprint: https://arxiv.org/html/2504.02066v2 .
Check against the final article and supplied author data before quantitative
claims; preprint and final values need not coincide.

| Target | What this package can compute | Remaining requirement |
|---|---|---|
| Fig. 1: bare relaxation/calibration | Frozen-j0 bare Pauli verification; add a bare HOPS case with `cavity=false` | Verified author model and preparation; matched bath/lifetime conventions |
| Fig. 2: N4 coupling dependence and pathways | Conditional Pauli coupling scan, HOPS cases, eigenenergies/weights/overlaps | Validated HOPS controls, sufficient time/seeds, exact rate-fit windows and assignments |
| Fig. 3: strong-coupling dynamics and spectra | Conditional Pauli dynamics and static state information | Converged strong-coupling HOPS, author spectral processing; eigenenergies are not absorption maxima |
| Fig. 4: size dependence | Available N<=4 subsets; larger inputs through documented schema | Actual N=5..7 matrices/realizations and long-time convergence |
| Fig. 5: overlap analysis | Overlap arrays with explicitly declared classification thresholds | Author state assignments and corresponding missing size models |

These static and Pauli references do not become HOPS results merely because
they are generated in the same package. HOPS runs are explicitly identified
by completed stochastic trajectory files and their ensemble analysis.

The goal is to establish where the economical population theory is reliable,
where a reproducible discrepancy survives numerical controls, and whether
that discrepancy changes a relevant observable. A short-time stochastic
trajectory difference alone answers none of those questions.

For new HOPS grid cases, copy `configs/validation.json`, edit its `cases` list,
and first retain the same short-horizon controls. Example case objects:

```json
[
  {"name":"bare_N1", "rings":1, "g0_cm_inverse":0.0, "cavity":false},
  {"name":"N4_g0p2", "rings":4, "g0_cm_inverse":0.2, "cavity":true},
  {"name":"N4_g0p8", "rings":4, "g0_cm_inverse":0.8, "cavity":true},
  {"name":"N4_g2", "rings":4, "g0_cm_inverse":2.0, "cavity":true}
]
```

A larger list multiplies the number of trajectories and required resources.
Different system dimensions or bath filters do not have identical stochastic
noise paths merely because they share an integer seed.

