# Numerical conventions and interpretation

The model contains one B800 site, one 18-site donor B850 ring, additional
18-site B850 rings and, for cavity cases, one photon. The supplied input has
four rings. Subsets N=1..4 use leading complete rings; no ring beyond the
available input is created. The independent bath couplings are molecular site
projectors; the photon has no direct phonon bath. The full system state basis
is retained. Only hierarchy adaptivity is enabled.

The reference Hamiltonian is the user's saved provisional N4,g0=2 input. Other
g0 values rescale its photon-to-molecule couplings by g0/2 while retaining its
frozen molecular block. This is conditional on its original dipole convention
and disorder realization. It is not a substitute for the authors' matrices.

The initial preparation is the eigenstate with the largest B800 weight, subject
to a minimum weight of 0.5. For the exact saved anchor case, the saved initial
vector is retained. Record and check state selection at each changed coupling;
an eigenstate-index label alone can change meaning at avoided crossings.

The frozen calibration is j0=13933.556774473034 (cm^-1)^2. It is not refitted
for each HOPS case. The declared bath is

    D(E) = j0 E^2/E0^3 exp(-E/E0), E0=100 cm^-1, T=300 K.

These temperature and calibration inputs remain provisional. The public energy
and time units are cm^-1 and ps, with hbar=5.3088374588761456 cm^-1 ps and
kB=0.6950348004861274 cm^-1 K^-1. Correlations passed to the backend are in
ps^-2, already including hbar^-2. The saved backend applies the same historical
MesoHOPS unit-rescaling convention used in the completed probes. It is not
silently upgraded to a newer MesoHOPS API.

The Hamiltonian convention is vertical, with no counterterm and factorized
thermal bath preparation. Molecular reorganization energies are recorded in
each bath audit. A HOPS-versus-Pauli difference can reflect preparation,
coherence, energy renormalization, memory, perturbation error or numerical error.
The present calculation does not isolate those mechanisms automatically.

## Pauli references

The generator uses the same site-projector overlap sums and continuum thermal
spectrum as the supplied model. Its column sums vanish. A matrix exponential
over the saved spacing is iterated; direct endpoint exponentiation, probability
normalization and positivity are checked. Negative-eigenvalue underflow or
near-zero-mode fitting is not used to reconstruct long-time curves.

Fitted-bath references use S(w)=2 Re sum_k c_k/(nu_k-iw). Each case compares its
actual weighted generator and Pauli curve with the continuum. Diagonal entries
of the transition matrix are zero before constructing the escape-rate diagonal.
Thus agreement of population-transition generators does not validate the
zero-frequency dephasing sector.

## Finite-time bath audit

For alpha(t)=sum c_k exp(-nu_k t), define the dimensionless line shape

    G(t) = integral_0^t (t-s) alpha(s) ds
         = sum c_k [exp(-nu_k t)-1+nu_k t]/nu_k^2.

For a unit difference in two commuting projector eigenvalues, the corresponding
pure-dephasing coherence magnitude is exp[-Re G(t)]. This diagnostic is not a
bound on the error of the full multilevel HOPS density matrix.

The new audit compares G with the continuum over each proposed time horizon.
The continuum integral is evaluated using the following analytic thermal sum.
Let q=t/hbar, a=1/E0, theta=kB*T, A=1+theta/E0. Then

    G_cont(t) = (j0/E0^3) [ q^2/{a^2(a+iq)}
                    + 2 theta Re{digamma(A+i theta q)-digamma(A)} ].

One obtains the first term by twice integrating 2/(a+it/hbar)^3. The paired
thermal terms reduce to the convergent sum of
2/(n+A)-1/(n+A+i theta q)-1/(n+A-i theta q), which gives the digamma difference.
The expression includes the complete thermal tail and avoids extrapolating a
finite-series correlation error to a long-time line shape.

This analytic audit has been evaluated against the saved inputs, without running a HOPS trajectory. Its reported
threshold (default max|Gfit-Gcont| <= 0.01) is a declared numerical diagnostic,
not an established universal HOPS error guarantee. Compare the actual b8/b16
HOPS ensemble results as well.

The continuum's S(0) is zero, whereas the frozen b8/b16 fits have nonzero
residuals. These values are always reported. A short finite-horizon gate may
pass without reproducing the exact zero-frequency limit; such a result is not
permission to extrapolate to arbitrary long times. A failed finite-horizon gate
blocks that task. Do not set S(0) to zero in a rate table while retaining the
same fitted correlation and call the bath repaired.

## HOPS and resource controls

The pinned backend runs normalized nonlinear HOPS. Each trajectory starts from
the same declared pure system preparation and a thermal stochastic environment.
One-trajectory observables are stochastic samples, not physical ensemble means.
Norm preservation is partly enforced by explicit stepwise renormalization.

The hierarchy resource check and basis update occur after every 0.5 fs step
(0.25 fs for the timestep control). These checks occur after propagation;
temporary allocations during a step can be larger. Slurm memory enforcement is
therefore also necessary. A resource stop is not a physical instability result.

With M independent exponential modes and total depth K, the full hierarchy has
at most binomial(M+K,K) auxiliaries. The upper bound, vector storage and active
cap are logged. For K>2 the adaptive calculation is allowed to use a smaller
subset than this potentially enormous full bound. No full-hierarchy allocation
or total-memory guarantee is inferred from that estimate.

Every accepted trajectory is saved atomically. Finished trajectories are the
restart unit. No mid-trajectory restart or automatic replacement seed is used.

## Statistical and convergence diagnostics

Ensemble SE is the sample standard deviation divided by sqrt(N). Plots show
pointwise +/-2 SE, not simultaneous confidence bands. Paired controls use
per-seed differences. The diagnostic max_t,O(|mean difference|+2 SE) is compared
with a declared tolerance, provided all planned control pairs are complete.
This is not a formal confidence statement for the maximum over all observations.

The defaults require eight paired controls, population tolerance 0.001 and
baseline sampling SE <=0.002. They are research starting criteria, not a paper
standard. Hierarchy, timestep, adaptivity, bath approximation, energy convention,
Hamiltonian and preparation uncertainties must all be considered. Short-window
agreement cannot certify long-time rates, overshoots or steady populations.

