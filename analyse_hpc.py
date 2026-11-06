"""Analyse saved HOPS trajectories only; no dynamics are run by this module."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from research_hpc import load_campaign, task_path, task_fingerprint, dump, np_save, numerical_modules

NAMES = ["B800", "donor_B850", "external_B850", "photon"]


def moments(values):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.full_like(mean, np.nan)
    return mean, se


def analyse(folder):
    folder = Path(folder).resolve()
    out = folder/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    if not (folder/"campaign.json").is_file():
        dump(out/"status.json", {"status": "PREPARATION_NOT_COMPLETED", "paper_reproduction_verified": False})
        print("No prepared campaign; inspect preparation logs.", flush=True)
        return
    folder, meta = load_campaign(folder)
    _, _, old = numerical_modules()
    inventory, metric_rows, groups = [], [], {}
    for task in meta["tasks"]:
        file = task_path(folder, task)
        state = "COMPLETE" if file.is_file() else "LOCKED_OR_INTERRUPTED" if file.with_suffix(".lock").is_file() else "FAILED" if file.with_suffix(".failure.json").is_file() else "PENDING"
        inventory.append({"id": task["id"], "stage": task["stage"], "case": task["case"], "seed": task["seed"], "status": state})
        if not file.is_file():
            continue
        with np.load(file, allow_pickle=False) as d:
            info = json.loads(str(d["metadata"]))
            if info["task_fingerprint"] != task_fingerprint(meta, task):
                raise ValueError("Trajectory metadata mismatch: " + str(file))
            sample = {key: d[key] for key in ("time_ps", "populations", "diagonal_eigen_observables", "coherence_contribution")}
        sample.update(seed=task["seed"], index=task["index"])
        groups.setdefault((task["case"], task["stage"]), []).append(sample)
        metric_rows.append({"task_id": task["id"], "case": task["case"], "stage": task["stage"], **info["metrics"]})
    old.table(out/"task_inventory.csv", inventory)
    old.table(out/"trajectory_metrics.csv", metric_rows)
    reports, curves = [], []
    for (case, stage), samples in sorted(groups.items()):
        samples.sort(key=lambda v: v["index"])
        t = samples[0]["time_ps"]
        if any(not np.array_equal(t, sample["time_ps"]) for sample in samples):
            raise ValueError("Saved time grids differ within an ensemble")
        full, se = moments([sample["populations"] for sample in samples])
        diagonal, diagonal_se = moments([sample["diagonal_eigen_observables"] for sample in samples])
        coherence, coherence_se = moments([sample["coherence_contribution"] for sample in samples])
        spec = meta["config"]["stages"][stage]
        with np.load(folder/"references"/(case+"_continuum.npz"), allow_pickle=False) as d:
            indices = np.rint(t/meta["config"]["save_every_ps"]).astype(int)
            if not np.allclose(t, d["time_ps"][indices], atol=1e-12, rtol=0):
                raise ValueError("Reference and trajectory times differ")
            continuum = d["populations"][indices]
        with np.load(folder/"references"/(case+"_"+spec["bath"]+".npz"), allow_pickle=False) as d:
            fitted = d["populations"][indices]
        complete = len(samples) == int(spec["trajectories"])
        report = {"case": case, "stage": stage, "completed": len(samples), "planned": spec["trajectories"], "complete": complete,
                  "maximum_sampling_SE": float(np.max(se)) if len(samples)>1 else None,
                  "maximum_full_population_difference_from_continuum_Pauli": float(np.max(abs(full-continuum))),
                  "maximum_full_population_difference_from_fitted_Pauli": float(np.max(abs(full-fitted))),
                  "maximum_dressed_diagonal_difference_from_fitted_Pauli": float(np.max(abs(diagonal-fitted))),
                  "maximum_absolute_mean_coherence_contribution": float(np.max(abs(coherence))),
                  "last_time_ps": float(t[-1]), "time_points": len(t),
                  "sampling_bands_are_pointwise_not_simultaneous": True,
                  "partial_ensemble_may_be_failure_biased": not complete,
                  "transfer_rate_extracted": False, "convergence_certified": False}
        reports.append(report)
        stem = case+"__"+stage
        np_save(out/(stem+"_ensemble.npz"), time_ps=t, populations=full, population_SE=se,
                diagonal_eigen_observables=diagonal, diagonal_SE=diagonal_se, coherence_contribution=coherence,
                coherence_SE=coherence_se, continuum_pauli=continuum, fitted_bath_pauli=fitted,
                seeds=np.array([sample["seed"] for sample in samples]))
        for i, time_value in enumerate(t):
            for j, name in enumerate(NAMES):
                curves.append({"case": case, "stage": stage, "time_ps": time_value, "observable": name,
                               "HOPS_mean": full[i,j], "sampling_SE": se[i,j], "dressed_diagonal_mean": diagonal[i,j],
                               "coherence_mean": coherence[i,j], "coherence_SE": coherence_se[i,j],
                               "continuum_Pauli": continuum[i,j], "fitted_bath_Pauli": fitted[i,j], "ensemble_complete": complete})
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for j, (axis, name) in enumerate(zip(axes.flat, NAMES)):
            axis.plot(t, continuum[:,j], "k--", label="Continuum Pauli")
            axis.plot(t, fitted[:,j], color="gray", ls=":", label="Fitted-bath Pauli")
            axis.plot(t, full[:,j], label="HOPS mean")
            axis.plot(t, diagonal[:,j], ls="-.", label="HOPS dressed diagonal")
            if len(samples)>1:
                axis.fill_between(t, full[:,j]-2*se[:,j], full[:,j]+2*se[:,j], alpha=.2, label="Pointwise +/-2 SE")
            axis.set(xlabel="Time (ps)", ylabel="Population", title=name)
            axis.legend(fontsize=6)
        fig.suptitle(f"{case} | {stage} | Ntraj={len(samples)} | {'complete' if complete else 'INCOMPLETE'} | provisional comparison", fontsize=10)
        fig.tight_layout(); fig.savefig(out/(stem+"_comparison.png"), dpi=180); plt.close(fig)
    old.table(out/"ensemble_curves.csv", curves)
    old.table(out/"stage_reports.csv", reports)
    dump(out/"stage_reports.json", reports)
    paired = []
    thresholds = meta["config"]["convergence_thresholds"]
    for case in meta["cases"]:
        for control, base_stage in meta["config"].get("paired_controls", {}).items():
            base = {sample["seed"]: sample for sample in groups.get((case, base_stage), [])}
            controls = groups.get((case, control), [])
            matched = [(sample, base[sample["seed"]]) for sample in controls if sample["seed"] in base]
            expected = int(meta["config"]["stages"][control]["trajectories"])
            record = {"case": case, "control": control, "baseline": base_stage, "matched_pairs": len(matched), "expected_pairs": expected,
                      "paired_diagnostic_tolerance_met": False, "pointwise_not_simultaneous": True}
            if matched:
                if any(not np.array_equal(a["time_ps"], b["time_ps"]) for a,b in matched):
                    raise ValueError("Paired control horizons/grids differ")
                delta = np.stack([a["populations"]-b["populations"] for a,b in matched])
                mean, se = moments(delta)
                location = np.unravel_index(np.argmax(abs(mean)), mean.shape)
                bound = float(np.max(abs(mean)+thresholds["SE_multiplier"]*se)) if len(matched)>1 else None
                passed = len(matched)==expected and len(matched)>=thresholds["minimum_control_pairs"] and bound is not None and bound<=thresholds["population_absolute_tolerance"]
                record.update(maximum_absolute_mean_difference=float(np.max(abs(mean))),
                              SE_at_largest_difference=float(se[location]) if len(matched)>1 else None,
                              maximum_abs_difference_plus_pointwise_SE=bound,
                              observable_at_largest_difference=NAMES[location[1]],
                              time_at_largest_difference_ps=float(matched[0][0]["time_ps"][location[0]]),
                              paired_diagnostic_tolerance_met=bool(passed))
                np_save(out/(case+"__"+control+"_paired.npz"), time_ps=matched[0][0]["time_ps"], mean_difference=mean, paired_SE=se)
            record["bath_control_note"] = "Same random seed with a different filter is not an identical physical noise realization" if "bath" in control else "Same seed and fixed noise grid"
            paired.append(record)
    old.table(out/"paired_controls.csv", paired)
    dump(out/"paired_controls.json", paired)
    complete = all(row["status"]=="COMPLETE" for row in inventory)
    minimum_precision = thresholds["sampling_SE_target"]
    baseline_stages = set(meta["config"].get("paired_controls", {}).values())
    baselines = [r for r in reports if r["stage"] in baseline_stages]
    precision_met = bool(baselines) and all(r["complete"] and r["maximum_sampling_SE"] is not None and r["maximum_sampling_SE"]<=minimum_precision for r in baselines)
    control_count = len(meta["cases"])*len(meta["config"].get("paired_controls", {}))
    control_checks_met = control_count>0 and len(paired)==control_count and all(p["paired_diagnostic_tolerance_met"] for p in paired)
    result = {"all_planned_tasks_completed": complete, "paired_pointwise_diagnostics_met": control_checks_met,
              "baseline_sampling_precision_met": precision_met, "tested_horizon_ps": max(s["tmax_ps"] for s in meta["config"]["stages"].values()),
              "longer_horizon_accuracy_inferred": False, "finite_hierarchy_and_bath_errors_remain_conditional": True,
              "paper_reproduction_verified": False, "author_inputs_independently_verified": False,
              "formal_convergence_certificate": False, "transfer_rate_extracted": False,
              "next_action": "Review paired controls, bath line-shape errors and resource scaling before configuring a longer horizon or more cases."}
    dump(out/"convergence_report.json", result)
    dump(out/"status.json", {"completed": sum(r["status"]=="COMPLETE" for r in inventory), "planned": len(inventory), **result})
    (out/"READ_FIRST.md").write_text("These are conditional HOPS-versus-Pauli calculations. Completion is not convergence.\n"
        "The +/-2 SE bands address pointwise stochastic sampling only. They exclude model, bath, time-step and hierarchy errors.\n"
        "Incomplete ensembles may be biased by failed trajectories. Same-seed reruns are not independent data.\n"
        "No claim about the theory paper, a transfer rate, or an experiment is certified automatically.\n"
        "A short-horizon depth comparison does not validate a longer time window. Inspect bath audits, especially the low-frequency residual.\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
