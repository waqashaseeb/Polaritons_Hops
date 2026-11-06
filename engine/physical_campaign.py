"""Matched physical HOPS pilot. New campaign code: not run by the assistant."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bath_tools_saved as bath
import adaptive_backend_saved as backend
from setup_backend import setup, activate

ROOT = Path(__file__).resolve().parent
NAMES = ['B800', 'donor_B850', 'external_B850', 'photon']
HBAR = 5.3088374588761456


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def numba_version():
    try:
        return version('numba')
    except PackageNotFoundError:
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(clean(rows))


def sources():
    paths = list(ROOT.glob('*.py'))+[ROOT/'inputs/physical_N4_g2.npz', ROOT/'inputs/provenance.json',
                                   ROOT/'inputs/frozen_b8.npz', ROOT/'inputs/frozen_b8_audit.json',
                                   ROOT/'inputs/expected_backend_files.json']
    expected = read(ROOT/'inputs/expected_backend_files.json')
    paths += [ROOT/name for name in expected]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def load_model():
    provenance = read(ROOT/'inputs/provenance.json')
    if sha(ROOT/'inputs/physical_N4_g2.npz') != provenance['model_file_sha256']:
        raise ValueError('Physical model hash changed')
    with np.load(ROOT/'inputs/physical_N4_g2.npz', allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}
    H = model['H_cm_inverse']
    U = model['eigenvectors']
    initial = model['initial_vector']
    n = len(H)
    if n != 74 or np.max(abs(H-H.conj().T)) > 1e-10 or abs(np.vdot(initial, initial)-1) > 1e-10:
        raise ValueError('Wrong dimension, Hamiltonian or initial state')
    if np.max(abs(U.conj().T@U-np.eye(n))) > 1e-9:
        raise ValueError('Saved eigenvectors fail orthonormality')
    p0 = abs(U.conj().T@initial)**2
    if np.max(abs(p0-model['initial_eigenpopulations'])) > 1e-9:
        raise ValueError('Initial vector differs from the saved Pauli preparation')
    groups = [np.array([0]), np.arange(1, 19), np.arange(19, n-1), np.array([n-1])]
    return model, provenance, groups


def pauli_curve(R, initial, weights, times, tolerance=1e-6):
    if len(times) < 2 or not np.allclose(np.diff(times), times[1]-times[0], atol=1e-12, rtol=0):
        raise ValueError('Reference time grid must be uniform')
    p = np.asarray(initial, float).copy()
    E = expm(R*(times[1]-times[0]))
    pop = []
    for _ in times:
        pop.append(p.copy())
        p = E@p
    pop = np.asarray(pop)
    error = float(np.max(abs(pop[-1]-expm(R*times[-1])@initial)))
    norm = float(np.max(abs(pop.sum(axis=1)-1)))
    if not np.all(np.isfinite(pop)) or norm > tolerance or pop.min() < -tolerance or error > tolerance:
        raise FloatingPointError('Pauli reference propagation failed')
    return pop@weights, {'normalization_error': norm, 'minimum_population': float(pop.min()), 'endpoint_error': error}


def fitted_generator(model, c, nu):
    e = model['eigenenergies_cm_inverse']
    omega = (e[None, :]-e[:, None])/HBAR
    spectrum = bath.spectrum(omega.reshape(-1), c, nu).reshape(omega.shape)
    W = spectrum*model['overlap']
    np.fill_diagonal(W, 0)
    if W.min() < -1e-10*max(float(W.max()), 1e-30):
        raise FloatingPointError('Fitted bath produces negative physical-model transition rates')
    R = W.copy()
    R[np.diag_indices_from(R)] = -W.sum(axis=0)
    return R, spectrum


def prepare(config_path):
    cfg = read(config_path)
    backend_info = setup(cfg['existing_workflow_dir'])
    activate()
    # Import-check the actual pinned source and unit convention on the USER's machine.
    backend.imports()
    model, provenance, groups = load_model()
    protocol = provenance['protocol']
    if protocol['temperature_K'] != 300 or protocol['E0_cm_inverse'] != 100:
        raise ValueError('Frozen bath is tied to 300 K and E0=100 cm^-1')
    if cfg['bath_fit_settings']['temperature_K'] != protocol['temperature_K'] or cfg['bath_fit_settings']['E0_cm'] != protocol['E0_cm_inverse']:
        raise ValueError('Bath fit settings differ from the frozen physical protocol')
    source_hashes = sources()
    signature = digest({'config': cfg, 'source_hashes': source_hashes})
    folder = ROOT/'campaigns'/('physical_N4_g2_'+signature[:12])
    if (folder/'campaign.json').is_file():
        read_campaign(folder)
        print('Existing campaign:', folder, flush=True)
        print('CAMPAIGN:', folder, flush=True)
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    print('CAMPAIGN:', folder, flush=True)
    for name in ['baths', 'references', 'trajectories', 'logs', 'analysis']:
        (folder/name).mkdir(exist_ok=True)
    dump(folder/'config_snapshot.json', cfg)
    dump(folder/'backend_setup.json', backend_info)
    dump(folder/'provenance.json', provenance)
    dump(folder/'preparation_environment.json', {'python': sys.version, 'platform': platform.platform(),
                                                'numpy': np.__version__, 'scipy': scipy.__version__, 'numba': numba_version()})
    j0 = float(provenance['calibration']['j0_cm_inverse_squared'])
    max_time = max(float(s['tmax_ps']) for s in cfg['stages'].values())
    save_step = cfg['save_every_ps']
    times = np.arange(round(max_time/save_step)+1)*save_step
    original, original_checks = pauli_curve(model['R_ps_inverse'], model['initial_eigenpopulations'], model['observable_weights'], times)
    save(folder/'references/continuum_pauli.npz', time_ps=times, populations=original)
    dump(folder/'references/continuum_pauli_checks.json', original_checks)
    bath_hashes, bath_reports = {}, {}
    for name in ['b8', 'b16']:
        settings = copy.deepcopy(cfg['bath_fit_settings'])
        if name == 'b8':
            if sha(ROOT/'inputs/frozen_b8.npz') != provenance['frozen_b8_sha256']:
                raise ValueError('Frozen b8 correlation changed')
            with np.load(ROOT/'inputs/frozen_b8.npz', allow_pickle=False) as data:
                c_unit, nu = data['c_unit_ps2'], data['nu_ps']
            prior = read(ROOT/'inputs/frozen_b8_audit.json')
            if not prior['passed']:
                raise ValueError('Previous b8 audit was not passed')
            fit_info = {'source': 'exact frozen b8 used in completed benchmarks', 'rank': 8, 'terms': len(c_unit)}
        else:
            settings['rank'] = 16
            c_unit, nu, fit_info = bath.fit_exponentials(settings)
        # Re-audit the correlation, and then the actual physical transition matrix.
        audited, correlation_data, rate_data = bath.audit(c_unit, nu, settings)
        if not audited['passed']:
            dump(folder/'baths'/(name+'_failure_audit.json'), audited)
            raise ValueError(name+' bath correlation audit failed')
        c = c_unit*j0
        Rfit, S = fitted_generator(model, c, nu)
        fitted, fit_checks = pauli_curve(Rfit, model['initial_eigenpopulations'], model['observable_weights'], times)
        rate_scale = max(float(np.max(abs(model['R_ps_inverse']))), 1e-30)
        generator_error = float(np.max(abs(Rfit-model['R_ps_inverse']))/rate_scale)
        population_error = float(np.max(abs(fitted-original)))
        reorg = float(-HBAR*np.imag(np.sum(c/nu)))
        actual = {'maximum_actual_energy_gap_cm_inverse': float(np.ptp(model['eigenenergies_cm_inverse'])),
                  'weighted_generator_max_error_over_max_entry': generator_error,
                  'maximum_Pauli_population_difference_over_reference_horizon': population_error,
                  'reference_horizon_ps': max_time, 'reorganization_energy_cm_inverse_from_fit': reorg,
                  'continuum_reorganization_energy_cm_inverse': j0/protocol['E0_cm_inverse'],
                  'zero_frequency_spectrum_ps_inverse': float(bath.spectrum(np.array([0.]), c, nu)[0]),
                  'counterterm_added': False, 'energy_convention': 'vertical_no_counterterm',
                  'passed': generator_error <= cfg['physical_bath_generator_tolerance'] and population_error <= cfg['physical_bath_population_tolerance']}
        report = {'correlation_audit': audited, 'fit_info': fit_info, 'physical_model_audit': actual,
                  'Pauli_reference_checks': fit_checks, 'j0_was_refitted': False}
        dump(folder/'baths'/(name+'_audit.json'), report)
        if not actual['passed']:
            raise ValueError(name+' fitted bath is not accurate enough for this physical Pauli model; inspect the saved audit')
        path = folder/'baths'/(name+'.npz')
        save(path, c_unit_ps2=c_unit, c_ps2=c, nu_ps=nu)
        bath_hashes[name] = sha(path)
        bath_reports[name] = report
        save(folder/'references'/(name+'_pauli.npz'), time_ps=times, populations=fitted, R_ps_inverse=Rfit)
        tc, exact_c, approximate_c = correlation_data
        save(folder/'baths'/(name+'_correlation_audit.npz'), time_ps=tc, exact_unit_correlation=exact_c, fitted_unit_correlation=approximate_c)
        print(name, 'bath prepared; weighted generator error:', generator_error, 'Pauli curve error:', population_error, flush=True)
    tasks = []
    for stage, spec in cfg['stages'].items():
        settings = copy.deepcopy(cfg['backend'])
        settings.update(spec.get('backend_overrides', {}))
        for index in range(int(spec['trajectories'])):
            task = {'id': len(tasks), 'stage': stage, 'index': index, 'seed': int(cfg['seed_base']+1009*index),
                    'bath': spec['bath'], 'tmax_ps': float(spec['tmax_ps']), 'settings': settings}
            dt, noise = settings['dt_ps'], settings['noise_tau_ps']
            if abs(dt/(2*noise)-round(dt/(2*noise))) > 1e-8 or abs(save_step/dt-round(save_step/dt)) > 1e-8:
                raise ValueError('Integration/save spacing incompatible with the fixed noise grid')
            if abs(task['tmax_ps']/dt-round(task['tmax_ps']/dt)) > 1e-8:
                raise ValueError('Stage duration is not an integer number of integration steps')
            tasks.append(task)
    meta = {'signature': signature, 'config': cfg, 'source_hashes': source_hashes, 'bath_hashes': bath_hashes,
            'reference_hashes': {p.name: sha(p) for p in (folder/'references').glob('*.npz')},
            'bath_reports': bath_reports, 'tasks': tasks, 'created_utc': stamp(),
            'model_status': 'PROVISIONAL_NOT_AUTHOR_SUPPLIED', 'case': 'N4_g2',
            'energy_convention': 'vertical_no_counterterm', 'initial_bath_state': 'factorized thermal bath',
            'completion_is_convergence_certification': False, 'new_campaign_code_tested_by_assistant': False}
    dump(folder/'campaign.json', meta)
    dump(folder/'runtime_limits.json', {k: cfg['backend'][k] for k in
                                      ['max_noise_memory_MB', 'max_active_auxiliaries', 'max_trajectory_seconds']})
    make_slurm(folder, meta)
    print('Prepared campaign:', folder, flush=True)
    return folder


def read_campaign(path):
    folder = Path(path).resolve()
    meta = read(folder/'campaign.json')
    if sources() != meta['source_hashes']:
        raise ValueError('Campaign source/input/backend changed; do not mix results under an old campaign')
    if digest({'config': meta['config'], 'source_hashes': meta['source_hashes']}) != meta['signature']:
        raise ValueError('Prepared scientific configuration changed')
    for name, expected in meta['bath_hashes'].items():
        if sha(folder/'baths'/(name+'.npz')) != expected:
            raise ValueError('Prepared bath coefficients changed: '+name)
    for name, expected in meta['reference_hashes'].items():
        if sha(folder/'references'/name) != expected:
            raise ValueError('Prepared Pauli reference changed: '+name)
    return folder, meta


def task_path(folder, task):
    return folder/'trajectories'/f"task_{task['id']:06d}.npz"


def fingerprint(task, meta):
    return digest({'task': task, 'campaign_signature': meta['signature'], 'bath_sha256': meta['bath_hashes'][task['bath']]})


def execute(folder, meta, task):
    output = task_path(folder, task)
    identity = fingerprint(task, meta)
    if output.is_file():
        with np.load(output, allow_pickle=False) as d:
            info = json.loads(str(d['metadata']))
        if info['task_fingerprint'] != identity:
            raise ValueError('Existing trajectory fingerprint mismatch')
        print('Already complete:', task['id'], flush=True)
        return
    lock = output.with_suffix('.lock')
    descriptor = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as f:
        json.dump({'pid': os.getpid(), 'host': platform.node(), 'started_utc': stamp()}, f)
    try:
        activate()
        model, provenance, groups = load_model()
        with np.load(folder/'baths'/(task['bath']+'.npz'), allow_pickle=False) as d:
            c, nu = d['c_ps2'], d['nu_ps']
        settings = copy.deepcopy(task['settings'])
        limits = read(folder/'runtime_limits.json')
        allowed = {'max_noise_memory_MB', 'max_active_auxiliaries', 'max_trajectory_seconds'}
        if set(limits)-allowed:
            raise ValueError('Runtime overrides may change resource ceilings only')
        settings.update(limits)
        n = len(model['H_cm_inverse'])
        operators = [np.diag(np.eye(n)[i]) for i in range(n-1)]
        console = sys.stdout
        started = time.monotonic()
        last = [0.]
        def progress(fraction, active):
            now = time.monotonic()
            if now-last[0] >= 30:
                print(f"Task {task['id']}: {fraction*100:.1f}%, {active} auxiliaries, {now-started:.0f}s", file=console, flush=True)
                last[0] = now
        print(f"Starting {task['stage']} task {task['id']}, seed={task['seed']}, depth={settings['depth']}", flush=True)
        times, psi, metrics = backend.trajectory(model['H_cm_inverse'], operators, model['initial_vector'], c, nu,
                                                task['tmax_ps'], settings, task['seed'],
                                                folder/'logs'/f"task_{task['id']:06d}.log", progress)
        stride = round(meta['config']['save_every_ps']/settings['dt_ps'])
        times, psi = times[::stride], psi[::stride]
        populations = np.stack([(abs(psi[:, group])**2).sum(axis=1) for group in groups], axis=1)
        eigenamps = psi@model['eigenvectors'].conj()
        eigenpop = abs(eigenamps)**2
        diagonal = eigenpop@model['observable_weights']
        coherence_part = populations-diagonal
        if np.max(abs(populations.sum(axis=1)-1)) > settings['norm_tolerance']:
            raise FloatingPointError('Site observables fail the probability sum check')
        info = {'task_fingerprint': identity, 'task': task, 'effective_settings': settings, 'metrics': metrics,
                'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'numba': numba_version(),
                'model_status': meta['model_status'], 'counterterm_added': False,
                'single_trajectory_is_ensemble_validation': False}
        arrays = {'time_ps': times, 'populations': populations, 'diagonal_eigen_observables': diagonal,
                  'coherence_contribution': coherence_part, 'eigenpopulations': eigenpop,
                  'metadata': np.array(json.dumps(clean(info))), 'observable_names': np.array(NAMES)}
        if meta['config']['save_wavefunctions']:
            arrays['psi'] = psi
        save(output, **arrays)
        output.with_suffix('.failure.json').unlink(missing_ok=True)
        print('Completed task', task['id'], flush=True)
    except Exception as exc:
        dump(output.with_suffix('.failure.json'), {'task': task, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def run_local(folder, meta, stage, workers, limit):
    selected = [t for t in meta['tasks'] if t['stage'] == stage]
    pending_ids = [t['id'] for t in selected if not task_path(folder, t).is_file()]
    if limit is not None:
        pending_ids = pending_ids[:limit]
    if workers < 1:
        raise ValueError('workers must be positive')
    # A real one-trajectory smoke must succeed before a physical ensemble.
    if stage != 'smoke':
        smoke = next(t for t in meta['tasks'] if t['stage'] == 'smoke')
        if not task_path(folder, smoke).is_file():
            raise ValueError('Run the smoke stage once before an ensemble')
    if stage in ['half_dt', 'depth5', 'bath16', 'tighter_delta']:
        base = [t for t in meta['tasks'] if t['stage'] == 'pilot'][:len(selected)]
        if any(not task_path(folder, t).is_file() for t in base):
            raise ValueError('Complete the matching pilot seeds before paired controls')
    print(len(pending_ids), 'unfinished trajectories;', workers, 'independent Python process(es)', flush=True)
    def launch(task_id):
        command = [sys.executable, '-u', str(ROOT/'physical_campaign.py'), 'worker', '--campaign', str(folder), '--task-id', str(task_id)]
        return task_id, subprocess.run(command, cwd=ROOT).returncode
    failure = None
    pending = iter(pending_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        for _ in range(workers):
            task_id = next(pending, None)
            if task_id is not None:
                active[pool.submit(launch, task_id)] = task_id
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                task_id, code = future.result()
                del active[future]
                if code:
                    failure = (task_id, code)
            if failure is None:
                for _ in range(workers-len(active)):
                    task_id = next(pending, None)
                    if task_id is not None:
                        active[pool.submit(launch, task_id)] = task_id
    if failure:
        raise RuntimeError('Worker failed at task '+str(failure[0])+'. Keep its seed; share the failure JSON. Completed trajectories are preserved.')


def mean_se(samples):
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1)/np.sqrt(len(samples)) if len(samples) >= 2 else np.full_like(mean, np.nan)
    return mean, se


def analyse(folder, meta):
    out = folder/'analysis'
    completed, inventory, metric_rows = {}, [], []
    for task in meta['tasks']:
        file = task_path(folder, task)
        if not file.is_file():
            inventory.append({'task_id': task['id'], 'stage': task['stage'], 'status': 'FAILED' if file.with_suffix('.failure.json').is_file() else 'PENDING'})
            continue
        with np.load(file, allow_pickle=False) as d:
            info = json.loads(str(d['metadata']))
            if info['task_fingerprint'] != fingerprint(task, meta):
                raise ValueError('Saved trajectory fingerprint mismatch')
            item = {k: d[k] for k in ['time_ps', 'populations', 'diagonal_eigen_observables', 'coherence_contribution']}
        item['seed'] = task['seed']
        item['index'] = task['index']
        completed.setdefault(task['stage'], []).append(item)
        inventory.append({'task_id': task['id'], 'stage': task['stage'], 'status': 'COMPLETE'})
        metric_rows.append({'task_id': task['id'], 'stage': task['stage'], **info['metrics']})
    table(out/'task_inventory.csv', inventory)
    table(out/'trajectory_metrics.csv', metric_rows)
    reports = {}
    for stage, samples in completed.items():
        samples.sort(key=lambda r: r['index'])
        t = samples[0]['time_ps']
        if any(not np.array_equal(t, r['time_ps']) for r in samples):
            raise ValueError('Stage time grids differ')
        selected = [task for task in meta['tasks'] if task['stage'] == stage]
        spec = meta['config']['stages'][stage]
        with np.load(folder/'references/continuum_pauli.npz', allow_pickle=False) as d:
            exact = d['populations'][:len(t)]
        with np.load(folder/'references'/(spec['bath']+'_pauli.npz'), allow_pickle=False) as d:
            fit = d['populations'][:len(t)]
        full, full_se = mean_se(np.stack([r['populations'] for r in samples]))
        diagonal, diagonal_se = mean_se(np.stack([r['diagonal_eigen_observables'] for r in samples]))
        coherence, coherence_se = mean_se(np.stack([r['coherence_contribution'] for r in samples]))
        save(out/(stage+'_ensemble.npz'), time_ps=t, populations=full, population_SE=full_se,
             diagonal_eigen_observables=diagonal, diagonal_SE=diagonal_se,
             coherence_contribution=coherence, coherence_SE=coherence_se,
             continuum_pauli=exact, fitted_bath_pauli=fit, seeds=np.array([r['seed'] for r in samples]))
        rows = []
        for i, time_value in enumerate(t):
            for j, name in enumerate(NAMES):
                rows.append({'time_ps': time_value, 'observable': name, 'HOPS_mean': full[i, j], 'sampling_SE': full_se[i, j],
                             'continuum_Pauli': exact[i, j], 'fitted_bath_Pauli': fit[i, j],
                             'HOPS_minus_fitted_Pauli': full[i, j]-fit[i, j],
                             'dressed_coherence_contribution': coherence[i, j], 'coherence_sampling_SE': coherence_se[i, j]})
        table(out/(stage+'_curves.csv'), rows)
        report = {'stage': stage, 'completed_trajectories': len(samples), 'planned_trajectories': len(selected),
                  'ensemble_complete': len(samples) == len(selected),
                  'maximum_population_difference_from_continuum_Pauli': float(np.max(abs(full-exact))),
                  'maximum_population_difference_from_fitted_bath_Pauli': float(np.max(abs(full-fit))),
                  'maximum_sampling_SE': float(np.nanmax(full_se)) if len(samples) >= 2 else None,
                  'maximum_absolute_dressed_coherence_contribution': float(np.max(abs(coherence))),
                  'sampling_intervals_are_pointwise_not_simultaneous': True,
                  'numerical_convergence_certified': False, 'transfer_rate_extracted': False,
                  'interpretation': 'Smoke is a functional check; pilot/control stages are not full-transfer validation.'}
        reports[stage] = report
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for j, (axis, name) in enumerate(zip(axes.flat, NAMES)):
            axis.plot(t, exact[:, j], color='black', ls='--', label='Continuum Pauli')
            axis.plot(t, fit[:, j], color='gray', ls=':', label='Fitted-bath Pauli')
            axis.plot(t, full[:, j], label=f'HOPS N={len(samples)}')
            if len(samples) >= 2:
                axis.fill_between(t, full[:, j]-2*full_se[:, j], full[:, j]+2*full_se[:, j], alpha=.2, label='Pointwise +/-2 SE')
            axis.set(xlabel='Time (ps)', ylabel='Population', title=name)
            axis.legend(fontsize=6)
        fig.suptitle(stage+' | provisional model | '+('complete' if report['ensemble_complete'] else 'INCOMPLETE ENSEMBLE'), fontsize=10)
        fig.tight_layout()
        fig.savefig(out/(stage+'_comparison.png'), dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j, (axis, name) in enumerate(zip(axes.flat, NAMES)):
            axis.plot(t, coherence[:, j])
            if len(samples) >= 2:
                axis.fill_between(t, coherence[:, j]-2*coherence_se[:, j], coherence[:, j]+2*coherence_se[:, j], alpha=.2)
            axis.axhline(0, color='gray', lw=.7)
            axis.set(xlabel='Time (ps)', ylabel='Site population minus dressed-diagonal part', title=name)
        fig.suptitle(stage+' | dressed-coherence contribution; pointwise sampling bands only', fontsize=10)
        fig.tight_layout()
        fig.savefig(out/(stage+'_coherence_contribution.png'), dpi=180)
        plt.close(fig)
    pair_reports = []
    if 'pilot' in completed:
        base = {r['index']: r for r in completed['pilot']}
        for stage in ['half_dt', 'depth5', 'bath16', 'tighter_delta']:
            if stage not in completed:
                continue
            pairs = [(r, base[r['index']]) for r in completed[stage] if r['index'] in base]
            if not pairs:
                continue
            if any(r['seed'] != b['seed'] or not np.array_equal(r['time_ps'], b['time_ps']) for r, b in pairs):
                raise ValueError('Paired control seed/time grids differ')
            difference = np.stack([r['populations']-b['populations'] for r, b in pairs])
            mean, se = mean_se(difference)
            location = np.unravel_index(np.argmax(abs(mean)), mean.shape)
            report = {'control': stage, 'paired_seeds': len(pairs),
                      'maximum_absolute_mean_difference': float(np.max(abs(mean))),
                      'SE_at_maximum_difference': float(se[location]) if len(pairs) >= 2 else None,
                      'time_at_maximum_ps': float(pairs[0][0]['time_ps'][location[0]]),
                      'observable_at_maximum': NAMES[location[1]], 'convergence_certified': False,
                      'noise_pairing_note': 'Same seed/noise grid. Bath16 has a changed filter, so its physical noise is not identical.'}
            pair_reports.append(report)
            save(out/(stage+'_paired_difference.npz'), time_ps=pairs[0][0]['time_ps'], mean_difference=mean, paired_SE=se)
    table(out/'paired_controls.csv', pair_reports)
    dump(out/'stage_reports.json', reports)
    dump(out/'status.json', {'completed_total': sum(len(v) for v in completed.values()), 'planned_total_including_optional_stages': len(meta['tasks']),
                            'stages_with_data': list(completed), 'HOPS_trajectories_completed': sum(len(v) for v in completed.values()),
                            'default_smoke_and_pilot_sequence_complete': all(reports.get(s, {}).get('ensemble_complete', False) for s in ['smoke', 'pilot']),
                            'paper_reproduction_verified': False, 'convergence_certified': False,
                            'incomplete_ensembles_must_not_be_interpreted_as_unbiased_results': True})
    (out/'READ_FIRST.md').write_text('''# Physical HOPS pilot

Read stage_reports.json, trajectory_metrics.csv, paired_controls.csv and the plots.
The saved model, initial state, independent site-projector baths and calibrated j0
are fixed. The two Pauli curves separate exponential-bath approximation from the
HOPS-versus-Pauli difference. No counterterm or excitation loss has been added.

Smoke checks execution only. The short pilot does not span the full transfer rise.
No transfer rate or Redfield-validity conclusion is extracted automatically.
Sampling error bars do not include timestep, hierarchy, adaptivity, bath-fit or
Hamiltonian uncertainty. Incomplete ensembles may be biased by failed trajectories.

Dressed-coherence contribution is the full site-observable value minus its
eigenpopulation-only reconstruction. It can diagnose a limitation of population-only
comparisons; it does not by itself separate memory, nonperturbative bath effects,
energy renormalization and preparation effects.

Read the bath reorganization energy and energy convention in baths/*_audit.json.
The paper's exact convention remains unverified, so any quantitative discrepancy
must be interpreted for this declared provisional model before attributing it to
failure of the paper's assumptions. No author input has been substituted.
''', encoding='utf-8')


def make_slurm(folder, meta):
    for stage in meta['config']['stages']:
        ids = [t['id'] for t in meta['tasks'] if t['stage'] == stage]
        text = f'''#!/usr/bin/env bash
# Submit only after prepare and a successful smoke on the same transferred package.
#SBATCH --job-name=hops-{stage}
#SBATCH --array=0-{len(ids)-1}%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=hops-{stage}-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
# Activate your existing NumPy/SciPy/Numba environment using your cluster's rules.
campaign_dir="${{SLURM_SUBMIT_DIR:?Submit from the prepared campaign directory}}"
test -f "$campaign_dir/campaign.json" || {{ echo "Submit from the prepared campaign directory" >&2; exit 2; }}
package_dir="$(cd -- "$campaign_dir/../.." && pwd)"
python -u "$package_dir/physical_campaign.py" worker --campaign "$campaign_dir" --stage {stage} --index "$SLURM_ARRAY_TASK_ID"
'''
        (folder/('slurm_'+stage+'.sh')).write_text(text, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'analyse'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pilot.json')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'half_dt', 'depth5', 'bath16', 'tighter_delta', 'preview', 'transfer'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.command == 'prepare':
        folder = prepare(args.config)
        print('CAMPAIGN:', folder, flush=True)
        return
    folder = args.campaign or prepare(args.config)
    folder, meta = read_campaign(folder)
    if args.command == 'worker':
        if args.task_id is not None:
            task = next(t for t in meta['tasks'] if t['id'] == args.task_id)
        else:
            chosen = [t for t in meta['tasks'] if t['stage'] == args.stage]
            if args.index is None or not 0 <= args.index < len(chosen):
                raise IndexError('Worker index outside stage')
            task = chosen[args.index]
        execute(folder, meta, task)
    elif args.command == 'run':
        if args.stage is None:
            parser.error('run requires --stage')
        try:
            run_local(folder, meta, args.stage, args.workers, args.limit)
        finally:
            analyse(folder, meta)
    else:
        analyse(folder, meta)
    print('CAMPAIGN:', folder, flush=True)


if __name__ == '__main__':
    main()
