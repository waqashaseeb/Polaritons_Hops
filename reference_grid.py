"""Conditional Pauli/static reference scans; this module does not run HOPS."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from research_hpc import ROOT, read, dump, digest, source_hashes, numerical_modules, make_model, np_save, stamp


def run(grid_config):
    grid = read(grid_config)
    model_config_path = ROOT/grid["model_config"]
    cfg = read(model_config_path)
    _, _, old = numerical_modules()
    signature = digest({"grid": grid, "model_configuration": cfg, "sources": source_hashes(cfg)})
    out = ROOT/"results/reference_runs"/("paper_reference_"+signature[:12])
    out.mkdir(parents=True, exist_ok=True)
    specs = [{"name": "bare_N1", "rings": 1, "g0_cm_inverse": 0., "cavity": False, "horizon_ps": grid["bare_horizon_ps"], "kind": "bare"}]
    for squared in grid["weak_g_squared_cm_inverse_squared"]:
        specs.append({"name": "N4_gsq_"+str(squared).replace(".", "p"), "rings": 4, "g0_cm_inverse": float(np.sqrt(squared)),
                      "cavity": True, "horizon_ps": max(25., grid["weak_horizon_scale_ps_cm_inverse_squared"]/squared), "kind": "weak"})
    for squared in grid["strong_g_squared_cm_inverse_squared"]:
        specs.append({"name": "N4_strong_gsq_"+str(squared).replace(".", "p"), "rings": 4, "g0_cm_inverse": float(np.sqrt(squared)),
                      "cavity": True, "horizon_ps": grid["strong_horizon_ps"], "kind": "strong"})
    for rings in range(2, grid["requested_maximum_rings"]+1):
        for g0 in grid["size_scan_g0_cm_inverse"]:
            specs.append({"name": "N"+str(rings)+"_size_g"+str(g0).replace(".", "p"), "rings": rings, "g0_cm_inverse": g0,
                          "cavity": True, "horizon_ps": grid["size_horizon_ps"], "kind": "size"})
    rows, eigenrows, failures, curve_index = [], [], [], []
    assignment = grid["molecular_assignment_threshold"]
    for spec in specs:
        try:
            model, checks = make_model(spec, cfg, np)
            times = np.linspace(0, spec["horizon_ps"], grid["saved_points"])
            pops, checks_prop = old.pauli_curve(model["R_ps_inverse"], model["initial_eigenpopulations"], model["observable_weights"], times)
            np_save(out/(spec["name"]+".npz"), time_ps=times, populations=pops, **model)
            E, weights, overlap = model["eigenenergies_cm_inverse"], model["observable_weights"], model["overlap"]
            donor = np.flatnonzero(weights[:,1]>assignment)
            donor = donor[np.argsort(E[donor])][:2]
            polariton = np.flatnonzero(weights[:,3]>grid["polariton_photon_weight_minimum"])
            external = np.flatnonzero(weights[:,2]>assignment)
            lower_external = external[E[external]<E[donor].max()] if len(donor) else np.array([], int)
            valid_assignment = len(donor)==2 and len(polariton)>0
            row = {"case": spec["name"], "kind": spec["kind"], "rings": spec["rings"], "g0_cm_inverse": spec["g0_cm_inverse"],
                   "g_squared_cm_inverse_squared": spec["g0_cm_inverse"]**2, "horizon_ps": spec["horizon_ps"],
                   "external_population_at_horizon": float(pops[-1,2]), "external_population_maximum": float(pops[:,2].max()),
                   "time_of_external_maximum_ps": float(times[np.argmax(pops[:,2])]),
                   "late_external_population_change": float(abs(pops[-1,2]-pops[round(.9*(len(pops)-1)),2])),
                   "donor_polariton_overlap_sum": float(overlap[np.ix_(donor, polariton)].sum()) if valid_assignment else None,
                   "donor_lower_external_overlap_sum": float(overlap[np.ix_(donor, lower_external)].sum()) if valid_assignment else None,
                   "polariton_lower_external_overlap_sum": float(overlap[np.ix_(polariton, lower_external)].sum()) if valid_assignment else None,
                   "classification_valid": valid_assignment, "classification_is_verified_author_protocol": False,
                   "rate_fit_performed": False, "model_status": checks["source_input_status"], "paper_reproduction_verified": False}
            # A diagnostic horizon-defined rise time; it is not the paper's fit.
            signal = pops[:,2]
            monotone = bool(np.min(np.diff(signal))>=-1e-8 and signal[-1]-signal[0]>1e-5)
            if monotone:
                rise = (signal-signal[0])/(signal[-1]-signal[0])
                t10, t90 = np.interp([.1,.9], rise, times)
                row["horizon_defined_10_to_90_effective_rate_ps_inverse"] = float(np.log(9)/(t90-t10)) if t90>t10 else None
            else:
                row["horizon_defined_10_to_90_effective_rate_ps_inverse"] = None
            window = grid.get("explicit_initial_rise_fit_windows_ps", {}).get(spec["name"])
            if window is not None:
                use = (times>=window[0]) & (times<=window[1])
                if use.sum()<8:
                    raise ValueError("Configured rate window has fewer than eight saved points; increase grid resolution")
                tx, yy = times[use], pops[use,2]
                def rise_function(t, offset, amplitude, rate):
                    return offset+amplitude*(-np.expm1(-rate*(t-tx[0])))
                pars, covariance = curve_fit(rise_function, tx, yy, p0=(yy[0], max(float(np.ptp(yy)), 1e-8), 1/max(tx[-1]-tx[0], 1e-9)),
                                              bounds=([-1.,0.,1e-12],[1.,2.,1e8]), maxfev=30000)
                row.update(rate_fit_performed=True, declared_initial_rise_fit_rate_ps_inverse=float(pars[2]),
                           initial_rise_fit_RMSE=float(np.sqrt(np.mean((rise_function(tx,*pars)-yy)**2))),
                           fit_window_start_ps=window[0], fit_window_stop_ps=window[1], fit_window_author_verified=False)
            rows.append(row)
            for i in range(len(E)):
                eigenrows.append({"case": spec["name"], "state": i, "energy_cm_inverse": E[i], "B800_weight": weights[i,0],
                                  "donor_B850_weight": weights[i,1], "external_B850_weight": weights[i,2], "photon_weight": weights[i,3]})
            curve_index.append({"case": spec["name"], "npz": spec["name"]+".npz", "propagation_checks": checks_prop, "input_checks": checks})
            if spec["name"]=="bare_N1":
                # Verify the frozen calibration on the declared bare model; never retune j0.
                dense_times = np.linspace(0, grid["bare_fit_window_ps"][1], 1001)
                bare, _ = old.pauli_curve(model["R_ps_inverse"], model["initial_eigenpopulations"], model["observable_weights"], dense_times)
                use = dense_times>=grid["bare_fit_window_ps"][0]
                def decay(t, offset, amplitude, rate): return offset+amplitude*np.exp(-rate*t)
                pars, _ = curve_fit(decay, dense_times[use], bare[use,0], p0=(0., bare[0,0], 1/.9), bounds=([0.,0.,1e-8],[1.,1.,1e4]), maxfev=30000)
                dump(out/"bare_calibration_check.json", {"frozen_j0_cm_inverse_squared": cfg["protocol"]["j0_cm_inverse_squared"],
                    "fitted_tau_ps": float(1/pars[2]), "fit_window_ps": grid["bare_fit_window_ps"], "j0_refitted": False,
                    "calibration_is_independent_validation": False, "fit_window_author_verified": False})
                np_save(out/"bare_dense.npz", time_ps=dense_times, populations=bare,
                        populations_with_declared_200ps_survival=bare*np.exp(-dense_times[:,None]/200.))
            print("Reference completed:", spec["name"], flush=True)
        except Exception as exc:
            failures.append({"case": spec["name"], "rings": spec["rings"], "error": str(exc)})
            print("Reference unavailable:", spec["name"], str(exc), flush=True)
    old.table(out/"conditional_reference_metrics.csv", rows)
    old.table(out/"eigenenergies_and_Hopfield_weights.csv", eigenrows)
    dump(out/"curve_index.json", curve_index)
    dump(out/"unavailable_cases.json", failures)
    dump(out/"provenance.json", {"grid": grid, "model_configuration": cfg, "source_hashes": source_hashes(cfg),
         "HOPS_trajectories_run": 0, "published_rate_fit_protocol_verified": False, "LP_MP_UP_absorption_maxima_reproduced": False,
         "paper_reproduction_verified": False, "model_status": "CONDITIONAL_ON_SUPPLIED_INPUTS"})
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11,4))
        weak = [row for row in rows if row["kind"]=="weak" and row.get("horizon_defined_10_to_90_effective_rate_ps_inverse") is not None]
        if weak:
            axes[0].loglog([row["g_squared_cm_inverse_squared"] for row in weak], [row["horizon_defined_10_to_90_effective_rate_ps_inverse"] for row in weak], "o-")
        axes[0].set(xlabel="g0 squared (cm^-2)", ylabel="Horizon-defined 10-90 effective rate (ps^-1)", title="Conditional Pauli diagnostic; not paper fit")
        for g0 in grid["size_scan_g0_cm_inverse"]:
            selected = sorted([row for row in rows if row["kind"]=="size" and row["g0_cm_inverse"]==g0], key=lambda row: row["rings"])
            axes[1].plot([row["rings"] for row in selected], [row["external_population_at_horizon"] for row in selected], "o-", label=f"g0={g0}")
        axes[1].set(xlabel="Number of rings supplied", ylabel="External population at declared horizon", title="No missing rings synthesized")
        axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"conditional_reference_overview.png", dpi=180); plt.close(fig)
    print("REFERENCE OUTPUT:", out, flush=True)
    (ROOT/"results/archives").mkdir(parents=True, exist_ok=True)
    archive = ROOT/"results/archives"/("conditional_pauli_references_"+stamp()+".zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file(): z.write(file, Path("references")/file.relative_to(out))
        for file in ROOT.glob("*.py"):
            z.write(file, Path("package")/file.name)
        for file in (ROOT/"configs").glob("*.json"):
            z.write(file, Path("package/configs")/file.name)
    print("SHARE THIS ZIP:", archive, flush=True)
