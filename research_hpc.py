"""HPC campaign preparation, HOPS workers and result collection.

The assistant checked syntax, synthetic orchestration and saved-input analytic
diagnostics. No new HOPS trajectory or HPC job was run by the assistant.
The supplied MesoHOPS core and saved backend are copied byte-for-byte from
the user's completed depth-2 probe. No GPU backend is implemented.
"""
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REQUIRED_WORKSPACE = Path("/home/tariq.qayyum/hpc/polariton_hops").resolve()
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def configure_workspace():
    """Keep this workflow's generated files inside the requested HPC workspace."""
    if ROOT != REQUIRED_WORKSPACE:
        raise ValueError("Install this project at /home/tariq.qayyum/hpc/polariton_hops/. Current package: " + str(ROOT))
    paths = {
        "TMPDIR": ROOT/"runtime/tmp",
        "PIP_CACHE_DIR": ROOT/"runtime/pip-cache",
        "XDG_CACHE_HOME": ROOT/"runtime/xdg-cache",
        "XDG_CONFIG_HOME": ROOT/"runtime/xdg-config",
        "XDG_DATA_HOME": ROOT/"runtime/xdg-data",
    }
    for key, default in paths.items():
        # Preserve per-job subdirectories set by slurm/environment.sh only
        # when they are also inside the requested workspace.
        candidate = Path(os.environ.get(key, str(default))).expanduser().resolve()
        if not candidate.is_relative_to(ROOT):
            candidate = default
        candidate.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(candidate)
    for key, name in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
        candidate = Path(os.environ.get(key, str(ROOT/"runtime"/name))).expanduser().resolve()
        if not candidate.is_relative_to(ROOT):
            candidate = ROOT/"runtime"/name
        candidate.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(candidate)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Input paths must stay inside the transferred package: " + str(value))
    return path


def source_hashes(cfg):
    paths = list(ROOT.glob("*.py"))
    paths += [ROOT/"Batch", ROOT/"requirements.txt"]
    paths += list((ROOT/"scripts").glob("*.sh"))
    paths += list((ROOT/"slurm").glob("*.sh"))+list((ROOT/"slurm").glob("*.sbatch"))
    paths += list((ROOT / "engine").rglob("*.py"))
    paths += [p for p in (ROOT / "engine/inputs").glob("*") if p.is_file()]
    paths += [ROOT / "inputs/frozen_b16_from_probe.npz", ROOT / "inputs/source_archives.json"]
    paths += [ROOT / "engine/third_party/MESOHOPS_LICENSE", ROOT / "engine/third_party/VENDOR_SHA256.json",
              ROOT / "engine/third_party/SOURCE.json"]
    if cfg.get("molecular_input"):
        paths += [package_path(cfg["molecular_input"]), package_path(cfg["molecular_provenance"])]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def campaign_identity(config):
    configure_workspace()
    cfg = read(config)
    if not cfg.get("enabled", False):
        raise ValueError("This configuration is a disabled template. Read its enable_instructions before adapting it.")
    hashes = source_hashes(cfg)
    signature = digest({"configuration": cfg, "source_hashes": hashes})
    name = Path(config).stem + "_" + signature[:12]
    return ROOT / "results/campaigns" / name, cfg, hashes, signature


def numerical_modules():
    sys.path.insert(0, str(ROOT / "engine"))
    import numpy as np
    import scipy
    import physical_campaign as old
    from setup_backend import setup, activate
    setup(ROOT / "engine")  # All pinned files must already be supplied and match.
    activate()
    old.backend.imports()
    return np, scipy, old


def np_save(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def input_model(cfg, np):
    if cfg.get("molecular_input"):
        path = package_path(cfg["molecular_input"])
        provenance = read(package_path(cfg["molecular_provenance"]))
        if sha(path) != provenance.get("npz_sha256"):
            raise ValueError("External molecular input hash does not match its declared provenance")
        with np.load(path, allow_pickle=False) as data:
            Hmol = np.asarray(data["H_molecular_cm_inverse"], complex)
            gunit = np.asarray(data["g_unit"], complex)
        status = "EXTERNAL_INPUT_USER_DECLARED_NOT_INDEPENDENTLY_VERIFIED"
    else:
        path = ROOT / "engine/inputs/physical_N4_g2.npz"
        provenance = read(ROOT / "engine/inputs/provenance.json")
        if sha(path) != provenance["model_file_sha256"]:
            raise ValueError("Saved provisional input hash changed")
        with np.load(path, allow_pickle=False) as data:
            Hmol = np.asarray(data["H_cm_inverse"][:-1, :-1], complex)
            gunit = np.asarray(data["H_cm_inverse"][:-1, -1], complex) / 2.0
        status = "PROVISIONAL_NOT_AUTHOR_SUPPLIED"
    if Hmol.ndim != 2 or Hmol.shape[0] != Hmol.shape[1] or gunit.shape != (len(Hmol),):
        raise ValueError("Molecular Hamiltonian or coupling-vector shape is invalid")
    if not np.all(np.isfinite(Hmol)) or not np.all(np.isfinite(gunit)):
        raise ValueError("Nonfinite molecular input")
    if np.max(abs(Hmol - Hmol.conj().T)) > 1e-9 or (len(Hmol) - 1) % 18:
        raise ValueError("Input must be Hermitian and ordered B800, then complete 18-site rings")
    if provenance.get("ordering", "B800_then_complete_B850_rings") != "B800_then_complete_B850_rings":
        raise ValueError("Unsupported site ordering; explicitly reorder your author inputs first")
    return Hmol, gunit, provenance, status


def continuum_generator(E, U, nmol, protocol, np):
    gap = E[None, :] - E[:, None]  # row target, column source
    energy = abs(gap)
    E0, temperature, j0 = protocol["E0_cm_inverse"], protocol["temperature_K"], protocol["j0_cm_inverse_squared"]
    density = j0 * energy**2 / E0**3 * np.exp(-energy / E0)
    occupation = np.zeros_like(energy)
    positive = energy > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        occupation[positive] = 1 / np.expm1(energy[positive] / (0.6950348004861274 * temperature))
    spectrum = 2 * np.pi / 5.3088374588761456 * density * (occupation + (gap > 0))
    weights = abs(U[:nmol])**2
    overlap = weights.T @ weights
    rates = spectrum * overlap
    np.fill_diagonal(rates, 0.0)
    rates[np.diag_indices_from(rates)] = -rates.sum(axis=0)
    return rates, overlap


def make_model(spec, cfg, np):
    Hmol, gunit, origin, status = input_model(cfg, np)
    rings = int(spec["rings"])
    available = (len(Hmol) - 1) // 18
    if rings < 1 or rings > available:
        raise ValueError(f"Requested N={rings}, but supplied molecular input contains N<={available}. No synthetic rings are created.")
    nmol = 1 + 18 * rings
    cavity = bool(spec.get("cavity", True))
    n = nmol + int(cavity)
    H = np.zeros((n, n), complex)
    H[:nmol, :nmol] = Hmol[:nmol, :nmol]
    if cavity:
        H[-1, -1] = cfg["protocol"]["cavity_energy_cm_inverse"]
        H[:nmol, -1] = float(spec["g0_cm_inverse"]) * gunit[:nmol]
        H[-1, :nmol] = H[:nmol, -1].conj()
    E, U = np.linalg.eigh(H)
    index = int(np.argmax(abs(U[0])**2))
    if abs(U[0, index])**2 < cfg["preparation"]["minimum_B800_weight"]:
        raise ValueError("No initial eigenstate meets the declared B800 preparation threshold")
    initial = U[:, index].copy()
    # Preserve the exact saved preparation for the user's already-run anchor case.
    if not cfg.get("molecular_input") and rings == 4 and cavity and float(spec["g0_cm_inverse"]) == 2.0:
        with np.load(ROOT / "engine/inputs/physical_N4_g2.npz", allow_pickle=False) as original:
            if np.max(abs(H - original["H_cm_inverse"])) < 1e-10:
                initial = original["initial_vector"].copy()
    p0 = abs(U.conj().T @ initial)**2
    groups = [np.array([0]), np.arange(1, 19), np.arange(19, nmol), np.array([nmol], int) if cavity else np.array([], int)]
    weights = np.stack([(abs(U[group])**2).sum(axis=0) for group in groups], axis=1)
    R, overlap = continuum_generator(E, U, nmol, cfg["protocol"], np)
    checks = {"Hermiticity_max_cm_inverse": float(np.max(abs(H - H.conj().T))),
              "eigenpair_residual_max_cm_inverse": float(np.max(abs(H @ U - U * E))),
              "orthonormality_max_error": float(np.max(abs(U.conj().T @ U - np.eye(n)))),
              "initial_norm_error": float(abs(np.vdot(initial, initial) - 1)),
              "initial_B800_weight": float(abs(initial[0])**2), "initial_eigenstate_index": int(np.argmax(p0)),
              "initial_energy_cm_inverse": float(E[np.argmax(p0)]), "number_molecular_baths": nmol,
              "source_input_status": status, "case": spec, "energy_convention": "vertical_no_counterterm",
              "calibration_is_independent_validation": False, "author_model_verified": False}
    if not cfg.get("molecular_input") and rings == 4 and cavity and float(spec["g0_cm_inverse"]) == 2.0:
        with np.load(ROOT / "engine/inputs/physical_N4_g2.npz", allow_pickle=False) as original:
            anchor_protocol = cfg["protocol"]
            if (np.max(abs(H-original["H_cm_inverse"]))<1e-10
                    and anchor_protocol["j0_cm_inverse_squared"] == 13933.556774473034
                    and anchor_protocol["temperature_K"] == 300 and anchor_protocol["E0_cm_inverse"] == 100):
                anchor_error = float(np.max(abs(R-original["R_ps_inverse"]))/max(float(np.max(abs(original["R_ps_inverse"]))), 1e-30))
                checks["relative_generator_difference_from_user_saved_anchor"] = anchor_error
                if anchor_error>1e-6:
                    raise ValueError("Reconstructed continuum generator differs from the user's saved N4_g2 anchor")
    if checks["orthonormality_max_error"] > 1e-8 or checks["eigenpair_residual_max_cm_inverse"] > 1e-7:
        raise ValueError("Eigenproblem validation failed")
    arrays = dict(H_cm_inverse=H, eigenenergies_cm_inverse=E, eigenvectors=U, initial_vector=initial,
                  initial_eigenpopulations=p0, observable_weights=weights, overlap=overlap, R_ps_inverse=R,
                  number_molecular_baths=np.array(nmol))
    return arrays, checks


def exact_lineshape(t, protocol, np):
    """Double integral of the declared continuum alpha, including thermal tail.

    Formula is derived in docs/METHODS.md; evaluated on the user's machine.
    It avoids the finite thermal-series tail in a long-horizon integral.
    """
    from scipy.special import digamma
    q = np.asarray(t, float) / 5.3088374588761456
    E0, temperature, j0 = protocol["E0_cm_inverse"], protocol["temperature_K"], protocol["j0_cm_inverse_squared"]
    a = 1 / E0
    theta = 0.6950348004861274 * temperature
    A = 1 + theta / E0
    vacuum = q**2 / (a*a*(a + 1j*q))
    thermal = 2 * theta * np.real(digamma(A + 1j*theta*q) - digamma(A))
    return j0 / E0**3 * (vacuum + thermal)


def bath_horizon_audit(c, nu, horizon, cfg, np):
    grid = np.unique(np.r_[np.linspace(0, min(horizon, .2), 1001),
                           np.linspace(0, horizon, cfg["bath_gate"]["audit_points"])])
    x = np.outer(grid, nu)
    function = np.expm1(-x) + x
    small = abs(x) < 1e-4
    function[small] = x[small]**2/2 - x[small]**3/6 + x[small]**4/24 - x[small]**5/120
    fitted = (function / nu[None, :]**2) @ c
    continuum = exact_lineshape(grid, cfg["protocol"], np)
    error = float(np.max(abs(fitted-continuum)))
    zero = float(2*np.real(np.sum(c/nu)))
    return {"horizon_ps": horizon, "maximum_complex_lineshape_error": error,
            "maximum_real_lineshape_error": float(np.max(abs((fitted-continuum).real))),
            "maximum_imaginary_lineshape_error": float(np.max(abs((fitted-continuum).imag))),
            "maximum_unit_projector_coherence_magnitude_difference": float(np.max(abs(np.exp(-fitted.real)-np.exp(-continuum.real)))),
            "fitted_zero_frequency_spectrum_ps_inverse": zero,
            "continuum_zero_frequency_spectrum_ps_inverse": 0.0,
            "zero_frequency_limit_reproduced": abs(zero) < cfg["bath_gate"]["zero_frequency_tolerance_ps_inverse"],
            "finite_horizon_passed": error <= cfg["bath_gate"]["maximum_complex_lineshape_error"],
            "diagnostic_is_not_full_system_bath_convergence": True}, (grid, continuum, fitted)


def load_campaign(folder, verify=True):
    folder = Path(folder).resolve()
    meta = read(folder / "campaign.json")
    if verify:
        if source_hashes(meta["config"]) != meta["source_hashes"]:
            raise ValueError("Scientific source/input hashes changed. Prepare a separate campaign; do not mix results.")
        if digest({"configuration": meta["config"], "source_hashes": meta["source_hashes"]}) != meta["signature"]:
            raise ValueError("Prepared configuration signature changed")
        for name, expected in meta["prepared_file_hashes"].items():
            if sha(folder / name) != expected:
                raise ValueError("Prepared scientific file changed: " + name)
    return folder, meta


def prepare(config, destination=None):
    folder, cfg, hashes, signature = campaign_identity(config)
    if destination and Path(destination).resolve() != folder:
        raise ValueError("Submission destination and calculated scientific identity differ")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "campaign.json").is_file():
        load_campaign(folder)
        print("Existing campaign:", folder, flush=True)
        return folder
    np, scipy, old = numerical_modules()
    lock = folder / "prepare.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "started_utc": stamp()}, stream)
    try:
        for name in ("models", "baths", "references", "trajectories", "task_logs", "analysis", "slurm_logs"):
            (folder / name).mkdir(exist_ok=True)
        protocol = cfg["protocol"]
        if protocol["temperature_K"] <= 0 or protocol["E0_cm_inverse"] <= 0 or protocol["j0_cm_inverse_squared"] <= 0:
            raise ValueError("Bath temperature, cutoff and calibration must be positive")
        save_step = float(cfg["save_every_ps"])
        maximum_time = max(float(stage["tmax_ps"]) for stage in cfg["stages"].values())
        if save_step <= 0 or maximum_time < save_step:
            raise ValueError("Invalid save spacing or horizon")
        if abs(maximum_time/save_step-round(maximum_time/save_step)) > 1e-8:
            raise ValueError("Maximum horizon must be divisible by save spacing")
        times = np.arange(round(maximum_time/save_step)+1) * save_step
        bath_names = list(dict.fromkeys(stage["bath"] for stage in cfg["stages"].values()))
        baths, bath_audits = {}, {}
        for name in bath_names:
            if name not in ("b8", "b16"):
                raise ValueError("Supported bath names are b8 and b16; add a separately audited input to extend this")
            settings = dict(cfg["bath_fit_settings"])
            settings.update(temperature_K=protocol["temperature_K"], E0_cm=protocol["E0_cm_inverse"])
            settings["rank"] = 8 if name == "b8" else 16
            if protocol["temperature_K"] == 300 and protocol["E0_cm_inverse"] == 100:
                path = ROOT/"engine/inputs/frozen_b8.npz" if name == "b8" else ROOT/"inputs/frozen_b16_from_probe.npz"
                with np.load(path, allow_pickle=False) as data:
                    cunit, nu = data["c_unit_ps2"], data["nu_ps"]
                fit_info = {"source": "USER_RUN_FROZEN_PROBE_BATH", "input_sha256": sha(path), "terms": len(cunit)}
            else:
                cunit, nu, fit_info = old.bath.fit_exponentials(settings)
            audit, correlation_data, rate_data = old.bath.audit(cunit, nu, settings)
            if not audit["passed"]:
                dump(folder/"baths"/(name+"_failed_audit.json"), audit)
                raise ValueError(name + " correlation audit failed")
            c = cunit * protocol["j0_cm_inverse_squared"]
            if np.any(nu.real <= 0):
                raise ValueError("Unstable bath pole")
            np_save(folder/"baths"/(name+".npz"), c_unit_ps2=cunit, c_ps2=c, nu_ps=nu)
            horizons = {}
            for horizon in sorted({float(stage["tmax_ps"]) for stage in cfg["stages"].values() if stage["bath"] == name}):
                report, (grid, exact, fitted) = bath_horizon_audit(c, nu, horizon, cfg, np)
                horizons[str(horizon)] = report
                np_save(folder/"baths"/(name+"_lineshape_"+str(horizon)+"ps.npz"), time_ps=grid, continuum=exact, fitted=fitted)
            bath_audits[name] = {"correlation_audit": audit, "fit_info": fit_info, "horizon_audits": horizons,
                                 "reorganization_energy_cm_inverse": float(-old.HBAR*np.imag(np.sum(c/nu))),
                                 "counterterm_added": False, "energy_convention": "vertical_no_counterterm"}
            dump(folder/"baths"/(name+"_audit.json"), bath_audits[name])
            baths[name] = (c, nu)
        cases, physical_audits = {}, {}
        for spec in cfg["cases"]:
            case = spec["name"]
            if not case.replace("_", "").replace("-", "").isalnum() or case in cases:
                raise ValueError("Case identifiers must be unique simple names")
            model, checks = make_model(spec, cfg, np)
            np_save(folder/"models"/(case+".npz"), **model)
            dump(folder/"models"/(case+"_checks.json"), checks)
            continuum, ref_checks = old.pauli_curve(model["R_ps_inverse"], model["initial_eigenpopulations"], model["observable_weights"], times)
            np_save(folder/"references"/(case+"_continuum.npz"), time_ps=times, populations=continuum,
                    eigenenergies_cm_inverse=model["eigenenergies_cm_inverse"], observable_weights=model["observable_weights"],
                    overlap=model["overlap"], R_ps_inverse=model["R_ps_inverse"])
            physical_audits[case] = {}
            for name, (c, nu) in baths.items():
                Rfit, spectrum = old.fitted_generator(model, c, nu)
                fitted, fit_checks = old.pauli_curve(Rfit, model["initial_eigenpopulations"], model["observable_weights"], times)
                matrix_error = float(np.max(abs(Rfit-model["R_ps_inverse"]))/max(float(np.max(abs(model["R_ps_inverse"]))), 1e-30))
                curve_error = float(np.max(abs(fitted-continuum)))
                report = {"maximum_gap_cm_inverse": float(np.ptp(model["eigenenergies_cm_inverse"])),
                          "relative_maximum_generator_error": matrix_error, "maximum_Pauli_population_error": curve_error,
                          "passed": matrix_error <= cfg["physical_bath_generator_tolerance"] and curve_error <= cfg["physical_bath_population_tolerance"],
                          "continuum_propagation_checks": ref_checks, "fitted_propagation_checks": fit_checks}
                physical_audits[case][name] = report
                np_save(folder/"references"/(case+"_"+name+".npz"), time_ps=times, populations=fitted, R_ps_inverse=Rfit)
            cases[case] = {"spec": spec, "checks": checks}
        tasks = []
        for stage, spec in cfg["stages"].items():
            settings = dict(cfg["backend"])
            settings.update(spec.get("backend_overrides", {}))
            dt = float(settings["dt_ps"])
            if dt<=0 or settings["noise_tau_ps"]<=0:
                raise ValueError("Integration and noise spacing must be positive")
            settings["chunk_ps"], settings["update_steps"] = dt, 1
            horizon = float(spec["tmax_ps"])
            for value in (dt/(2*settings["noise_tau_ps"]), save_step/dt):
                if abs(value-round(value)) > 1e-8:
                    raise ValueError("Integration/noise/save grids are incompatible")
            if abs(horizon/save_step-round(horizon/save_step)) > 1e-8:
                raise ValueError("Stage duration must be divisible by save spacing")
            horizon_pass = bath_audits[spec["bath"]]["horizon_audits"][str(horizon)]["finite_horizon_passed"]
            for case in cases:
                for index in range(int(spec["trajectories"])):
                    sample_index = index + int(spec.get("seed_offset", 0))
                    tasks.append({"id": len(tasks), "stage": stage, "case": case, "index": sample_index,
                                  "seed": int(cfg["seed_base"]+1009*sample_index), "settings": settings,
                                  "bath": spec["bath"], "tmax_ps": horizon,
                                  "bath_gate_passed": bool(horizon_pass and physical_audits[case][spec["bath"]]["passed"])})
        prepared = [p for base in ("models", "references", "baths") for p in (folder/base).rglob("*.npz")]
        meta = {"signature": signature, "config": cfg, "source_hashes": hashes, "cases": cases, "tasks": tasks,
                "prepared_file_hashes": {str(p.relative_to(folder)): sha(p) for p in prepared},
                "bath_audits": bath_audits, "physical_audits": physical_audits, "created_utc": stamp(),
                "execution_backend": "CPU_NUMPY_SCIPY_PINNED_MESOHOPS", "new_HOPS_trajectories_run_by_assistant": 0,
                "paper_reproduction_verified": False, "author_inputs_independently_verified": False,
                "energy_convention": "vertical_no_counterterm", "initial_bath_state": "factorized thermal",
                "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}}
        dump(folder/"config_snapshot.json", cfg)
        dump(folder/"campaign.json", meta)
        dump(folder/"prepare_status.json", {"status": "PREPARED", "tasks": len(tasks),
                                          "bath_blocked_tasks": sum(not t["bath_gate_passed"] for t in tasks)})
        print("Prepared:", folder, "Tasks:", len(tasks), flush=True)
        return folder
    except BaseException as exc:
        dump(folder/"prepare_failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def task_path(folder, task):
    return folder / "trajectories" / ("task_%06d.npz" % task["id"])


def task_fingerprint(meta, task):
    return digest({"signature": meta["signature"], "task": task})


def validate_prerequisites(folder, meta, task):
    for stage in meta["config"].get("prerequisites", {}).get(task["stage"], []):
        selected = [t for t in meta["tasks"] if t["stage"] == stage and t["case"] == task["case"]]
        if not selected or any(not task_path(folder, t).is_file() for t in selected):
            raise ValueError("Unfinished prerequisite stage: " + stage)


def worker(folder, stage, position):
    folder, meta = load_campaign(folder)
    selected = [t for t in meta["tasks"] if t["stage"] == stage]
    if position < 0 or position >= len(selected):
        raise ValueError("Array position outside prepared stage")
    task = selected[position]
    output = task_path(folder, task)
    identity = task_fingerprint(meta, task)
    if output.is_file():
        import numpy as np
        with np.load(output, allow_pickle=False) as d:
            if json.loads(str(d["metadata"]))["task_fingerprint"] != identity:
                raise ValueError("Existing trajectory fingerprint mismatch")
        print("Already completed; preserved task", task["id"], flush=True)
        return
    lock = output.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": stamp()}, stream)
    try:
        validate_prerequisites(folder, meta, task)
        if not task["bath_gate_passed"]:
            raise ValueError("Finite-horizon bath or physical Pauli bath audit failed. Inspect audits; do not bypass this for production.")
        np, scipy, old = numerical_modules()
        with np.load(folder/"models"/(task["case"]+".npz"), allow_pickle=False) as d:
            model = {key: d[key] for key in d.files}
        with np.load(folder/"baths"/(task["bath"]+".npz"), allow_pickle=False) as d:
            c, nu = d["c_ps2"], d["nu_ps"]
        settings = task["settings"]
        nmol, n = int(model["number_molecular_baths"]), len(model["H_cm_inverse"])
        modes = nmol*len(c)
        full_bound = math.comb(modes+int(settings["depth"]), int(settings["depth"]))
        preflight = {"maximum_mode_count": modes, "total_order_full_hierarchy_bound": full_bound,
                     "one_full_hierarchy_complex_vector_GiB": full_bound*n*16/1024**3,
                     "active_auxiliary_cap": settings["max_active_auxiliaries"],
                     "count_check_occurs_after_each_integration_step": True,
                     "temporary_allocations_can_exceed_active_vector_storage": True,
                     "slurm_memory_allocation_is_the_OS_resource_limit": True}
        if int(settings["depth"]) <= 2 and full_bound > settings["max_active_auxiliaries"]:
            raise MemoryError("Bounded depth<=2 full hierarchy exceeds its configured cap")
        print("TASK:", json.dumps(task), flush=True)
        print("PREFLIGHT:", json.dumps(preflight), flush=True)
        operators = [np.diag(np.eye(n)[i]) for i in range(nmol)]
        last = [time.monotonic()]
        console = sys.stdout
        def progress(fraction, active):
            now = time.monotonic()
            if now-last[0] >= 30:
                print(f"Task {task['id']} {100*fraction:.1f}% active={active}", file=console, flush=True)
                last[0] = now
        started = time.perf_counter()
        times, psi, metrics = old.backend.trajectory(model["H_cm_inverse"], operators, model["initial_vector"], c, nu,
                                                    task["tmax_ps"], settings, task["seed"],
                                                    folder/"task_logs"/("task_%06d.log" % task["id"]), progress)
        stride = round(meta["config"]["save_every_ps"]/settings["dt_ps"])
        times, psi = times[::stride], psi[::stride]
        groups = [np.array([0]), np.arange(1, 19), np.arange(19, nmol), np.arange(nmol, n)]
        populations = np.stack([(abs(psi[:,group])**2).sum(axis=1) for group in groups], axis=1)
        eigenpopulation = abs(psi @ model["eigenvectors"].conj())**2
        diagonal = eigenpopulation @ model["observable_weights"]
        if np.max(abs(populations.sum(axis=1)-1)) > settings["norm_tolerance"]:
            raise FloatingPointError("Saved site populations do not sum to one")
        import resource
        metrics["process_peak_RSS_MiB_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        metrics["whole_worker_elapsed_seconds"] = time.perf_counter()-started
        info = {"task_fingerprint": identity, "task": task, "metrics": metrics, "preflight": preflight,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "slurm": {k: v for k, v in os.environ.items() if k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")},
                "calculation_is_convergence_certificate": False, "paper_reproduction_verified": False}
        arrays = dict(time_ps=times, populations=populations, eigenpopulations=eigenpopulation,
                      diagonal_eigen_observables=diagonal, coherence_contribution=populations-diagonal,
                      metadata=np.array(json.dumps(clean(info))))
        if meta["config"].get("save_wavefunctions", True):
            arrays["psi"] = psi
        np_save(output, **arrays)
        output.with_suffix(".failure.json").unlink(missing_ok=True)
        print("Completed task", task["id"], "METRICS:", json.dumps(clean(metrics)), flush=True)
    except BaseException as exc:
        dump(output.with_suffix(".failure.json"), {"task": task, "error": str(exc), "traceback": traceback.format_exc(),
                                                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        raise
    finally:
        lock.unlink(missing_ok=True)


def collect(folder, include_trajectories=False):
    folder = Path(folder).resolve()
    (ROOT / "results/archives").mkdir(parents=True, exist_ok=True)
    destination = ROOT / "results/archives" / ("hops_hpc_results_" + folder.name + "_" + stamp() + ".zip")
    with ZipFile(destination, "w", ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and ".tmp." not in p.name and (include_trajectories or not (p.parent.name == "trajectories" and p.suffix == ".npz")):
                z.write(p, Path("campaign")/p.relative_to(folder))
        for directory in ("engine", "configs", "docs", "slurm", "inputs"):
            for p in sorted((ROOT/directory).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, Path("package")/p.relative_to(ROOT))
        for p in ROOT.glob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json", ".sh"):
                z.write(p, Path("package")/p.name)
    print("SHARE THIS ZIP:", destination, flush=True)
    return destination


def status(folder):
    folder = Path(folder).resolve()
    if not (folder/"campaign.json").is_file():
        print("Not prepared; inspect prepare Slurm log and prepare_failure.json:", folder)
        return
    meta = read(folder/"campaign.json")
    result = {}
    for task in meta["tasks"]:
        row = result.setdefault(task["stage"], {"complete": 0, "failed": 0, "pending": 0, "locked": 0})
        path = task_path(folder, task)
        key = "complete" if path.is_file() else "locked" if path.with_suffix(".lock").is_file() else "failed" if path.with_suffix(".failure.json").is_file() else "pending"
        row[key] += 1
    print(json.dumps(result, indent=2))


def unlock(folder, task_id):
    import subprocess
    folder = Path(folder).resolve()
    if task_id == "prepare":
        raise ValueError("For a stale prepare.lock, inspect the prepare job with squeue before removing that one file manually")
    path = folder/"trajectories"/("task_%06d.lock" % int(task_id))
    data = read(path)
    job = data.get("slurm_job_id")
    if not job:
        raise ValueError("No recorded Slurm job ID: verify the host/PID manually before removing this lock")
    result = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], text=True, capture_output=True)
    if result.returncode or result.stdout.strip():
        raise ValueError("Cannot establish that the recorded Slurm job has ended. Check squeue/sacct manually before removing " + str(path))
    path.unlink()
    print("Removed inactive-job lock only:", path)


def main():
    configure_workspace()
    def interrupted(signum, frame):
        raise InterruptedError("Trajectory interrupted by signal " + str(signum) + "; completed trajectories are preserved")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--config", type=Path, required=True); p.add_argument("--campaign", type=Path)
    p = sub.add_parser("worker"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--stage", required=True); p.add_argument("--position", type=int, required=True)
    p = sub.add_parser("analyse"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--collect", action="store_true")
    p = sub.add_parser("status"); p.add_argument("--campaign", type=Path, required=True)
    p = sub.add_parser("collect"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--include-trajectories", action="store_true")
    p = sub.add_parser("unlock"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--task-id", required=True)
    p = sub.add_parser("paper-reference-grid"); p.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.config, args.campaign)
    elif args.command == "worker": worker(args.campaign, args.stage, args.position)
    elif args.command == "status": status(args.campaign)
    elif args.command == "collect": collect(args.campaign, args.include_trajectories)
    elif args.command == "unlock": unlock(args.campaign, args.task_id)
    elif args.command == "paper-reference-grid":
        from reference_grid import run
        run(args.config)
    elif args.command == "analyse":
        try:
            from analyse_hpc import analyse
            analyse(args.campaign)
        finally:
            if args.collect: collect(args.campaign)


if __name__ == "__main__":
    main()
