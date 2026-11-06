"""Unit-checked interface to unmodified, vendored MesoHOPS v1.5.0.

Only hierarchy adaptivity is enabled. The full system basis is retained so weak
cavity couplings are not discarded by a state-selection threshold.
"""
from pathlib import Path
import contextlib
import re
import sys
import time
import numpy as np

HBAR=5.3088374588761456


def imports():
    third=Path(__file__).resolve().parent/'third_party'
    if str(third) not in sys.path:sys.path.insert(0,str(third))
    try:
        from mesohops.dynamics.hops_trajectory import HopsTrajectory
        from mesohops.dynamics.bath_corr_functions import bcf_exp
        from mesohops.util.physical_constants import hbar
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError('Install requirements.txt with this same Python interpreter. '
                          'The pinned MesoHOPS source is already included in third_party/.') from exc
    if abs((hbar/1000)/HBAR-1)>1e-3:
        raise RuntimeError('Unexpected MesoHOPS hbar convention: expected cm^-1 fs')
    return HopsTrajectory,bcf_exp,coo_matrix,float(hbar/1000)


def trajectory(H,operators,initial,c,nu,tmax,settings,seed,logpath,progress=None):
    HOPS,bcf_exp,coo,backend_hbar_ps=imports()
    dt=settings['dt_ps'];nt=round(tmax/dt)
    if abs(nt*dt-tmax)>1e-10:raise ValueError('tmax must be a multiple of dt_ps')
    dtfs=1000*dt;noise_tau=1000*settings.get('noise_tau_ps',dt/2)
    if abs(dtfs/(2*noise_tau)-round(dtfs/(2*noise_tau)))>1e-9:
        raise ValueError('dt_ps must be an integer multiple of twice noise_tau_ps')
    # Long padding resolves the correlation tail in the circulant embedding.
    tlen=max(1000*(tmax+settings['noise_padding_ps']),1000*settings['noise_min_window_ps'])
    estimate_mb=8*len(operators)*(int(tlen/noise_tau)+1)*8/1024**2
    if estimate_mb>settings['max_noise_memory_MB']:
        raise MemoryError(f'Conservative noise working-memory estimate {estimate_mb:.1f} MB exceeds budget')
    ls=[];gw=[]
    for op in operators:
        sparse=coo(np.asarray(op,complex))
        for cm,nm in zip(c,nu):
            ls.append(sparse);gw.append([complex(cm*backend_hbar_ps**2),complex(nm*backend_hbar_ps)])
    hamiltonian=np.asarray(H,complex)-np.trace(H).real/len(H)*np.eye(len(H))
    # The pinned backend uses a rounded historical constant. Rescale its energy
    # inputs so the public cm^-1 / ps convention remains exactly HBAR above.
    hamiltonian*=backend_hbar_ps/HBAR
    started=time.perf_counter();logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream):
        hops=HOPS({'HAMILTONIAN':hamiltonian,'GW_SYSBATH':gw,'L_HIER':ls,'L_NOISE1':ls,
                   'ALPHA_NOISE1':bcf_exp,'PARAM_NOISE1':gw},
                  eom_param={'EQUATION_OF_MOTION':'NORMALIZED NONLINEAR'},
                  noise_param={'MODEL':'FFT_FILTER','SEED':int(seed),'TLEN':float(tlen),'TAU':float(noise_tau)},
                  hierarchy_param={'MAXHIER':int(settings['depth'])},
                  storage_param={'phi_traj':False,'aux_list':False,'state_list':True,
                                 'list_nhier':True,'list_nstate':True})
        hops.make_adaptive(float(settings['delta_h']),0.,update_step=int(settings['update_steps']))
        hops.initialize(np.asarray(initial,complex))
        stream.flush()
        text=logpath.read_text(encoding='utf-8')
        fractions=[float(x) for x in re.findall(r'fractional negative:\s*([0-9.eE+-]+)',text)]
        clipping=max(fractions,default=0.)
        if clipping>settings['max_fft_negative_fraction']:
            raise RuntimeError(f'Noise FFT clipping fraction {clipping:g} exceeds tolerance. Increase noise_min_window_ps.')
        chunk_steps=max(1,round(settings['chunk_ps']/dt));done=0
        while done<nt:
            count=min(chunk_steps,nt-done);hops.propagate(count*dtfs,dtfs);done+=count
            active=len(hops.auxiliary_list)
            if active>settings['max_active_auxiliaries']:
                raise MemoryError(f'Active hierarchy {active} exceeds configured budget; partial run is not accepted')
            if time.perf_counter()-started>settings['max_trajectory_seconds']:
                raise TimeoutError('Trajectory exceeded configured time budget; increase it for a deliberate long run')
            if progress:progress(done/nt,active)
        psi=np.asarray(hops.storage['psi_traj']);t=np.asarray(hops.storage['t_axis'])/1000
        active_history=np.asarray(hops.storage['list_nhier'])
    norm_error=float(np.max(abs(np.sum(abs(psi)**2,axis=1)-1)))
    if len(t)!=nt+1 or not np.all(np.isfinite(psi)) or norm_error>settings['norm_tolerance']:
        raise FloatingPointError('Trajectory shape, finiteness or normalized nonlinear HOPS norm check failed')
    return t,psi,{'seed':int(seed),'maximum_active_auxiliaries':int(active_history.max()),
                  'norm_error':norm_error,'noise_fft_negative_fraction':clipping,
                  'elapsed_seconds':time.perf_counter()-started,'noise_memory_estimate_MB':estimate_mb,
                  'backend_hbar_cm_ps':backend_hbar_ps,'backend_energy_rescale':backend_hbar_ps/HBAR}


def ensemble(H,operators,initial,c,nu,tmax,settings,seeds,out,tag,groups=None,keep_density=True):
    rho_sum=None;rho_sq=None;pop_sum=None;pop_sq=None;metadata=[];per_seed=[]
    for k,seed in enumerate(seeds):
        print(f'{tag}: trajectory {k+1}/{len(seeds)}, seed={seed}',flush=True)
        t,psi,meta=trajectory(H,operators,initial,c,nu,tmax,settings,seed,
                            out/'trajectory_logs'/f'{tag}_{seed}.log')
        metadata.append(meta)
        sample=np.einsum('ti,tj->tij',psi,psi.conj()) if keep_density else None
        if keep_density:
            if rho_sum is None:rho_sum=np.zeros_like(sample);rho_sq=np.zeros(sample.shape)
            rho_sum+=sample;rho_sq+=abs(sample)**2
        if groups:
            pops=np.stack([np.sum(abs(psi[:,idx])**2,axis=1) for idx in groups.values()],axis=1)
            per_seed.append(pops)
            if pop_sum is None:pop_sum=np.zeros_like(pops);pop_sq=np.zeros_like(pops)
            pop_sum+=pops;pop_sq+=pops*pops
        # Save each completed trajectory immediately. Automatic resume is not implemented.
        np.savez_compressed(out/f'{tag}_trajectory_{seed}.npz',t_ps=t,psi=psi)
    n=len(seeds)
    if n<2:raise ValueError('At least two trajectories are required for sample standard errors')
    result={'t_ps':t,'metadata':metadata}
    if keep_density:
        mean=rho_sum/n;se=np.sqrt(np.maximum((rho_sq-n*abs(mean)**2)/(n-1)/n,0))
        result.update(rho=mean,rho_SE=se)
    if groups:
        mean=pop_sum/n;se=np.sqrt(np.maximum((pop_sq-n*mean*mean)/(n-1)/n,0))
        result.update(populations=mean,population_SE=se,individual_populations=np.stack(per_seed))
    return result
