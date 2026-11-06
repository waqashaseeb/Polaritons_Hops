"""Energy-domain super-Ohmic bath and a stable exponential approximation.

All correlation functions returned here are in ps^-2. The approximation is
audited against the continuum; a reported positive-spectrum correction is part
of the model and is never hidden from the user.
"""
import numpy as np
from scipy.linalg import hankel,svd
from scipy.integrate import trapezoid

HBAR=5.3088374588761456
KB=0.6950348004861274


def continuum(t,E0=100.,T=300.,j0=1.,terms=5000):
    t=np.asarray(t,float);a=1/E0+np.arange(1,terms+1)/(KB*T)
    value=np.empty(t.shape,complex)
    for start in range(0,len(t),128):
        q=t[start:start+128]/HBAR
        value[start:start+128]=2*j0/(HBAR**2*E0**3)*(1/(1/E0+1j*q)**3+
            (1/(a[None,:]+1j*q[:,None])**3+1/(a[None,:]-1j*q[:,None])**3).sum(axis=1))
    return value


def correlation(t,c,nu):
    return np.exp(-np.outer(np.asarray(t),nu))@c


def spectrum(w,c,nu):
    return 2*np.real(np.sum(c[None,:]/(nu[None,:]-1j*np.asarray(w)[:,None]),axis=1))


def spectral_rates(E,E0,T,j0=1.):
    E=np.asarray(E,float);n=1/np.expm1(E/(KB*T))
    D=j0*E**2/E0**3*np.exp(-E/E0)
    return 2*np.pi/HBAR*D*(n+1),2*np.pi/HBAR*D*n


def fit_exponentials(settings):
    dt=settings['fit_dt_ps'];t=np.arange(round(settings['fit_window_ps']/dt)+1)*dt
    target=continuum(t,settings['E0_cm'],settings['temperature_K'])
    rows=min(settings['pencil_rows'],len(t)//2)
    u,s,_=svd(hankel(target[:rows],target[rows-1:-1]),full_matrices=False)
    rank=settings['rank'];basis=u[:,:rank]
    eig=np.linalg.eigvals(np.linalg.lstsq(basis[:-1],basis[1:],rcond=None)[0])
    nu=-np.log(eig)/dt
    if np.any(nu.real<=0):raise ValueError('Unstable fitted pole: increase fit rank/window or inspect bath_fit.json')
    c=np.linalg.lstsq(np.exp(-np.outer(t,nu)),target,rcond=None)[0]
    # C(0) must be real. Remove only the fitted imaginary intercept residual.
    imaginary_intercept=float(c.sum().imag)
    c[np.argmax(abs(c))]-=1j*imaginary_intercept
    # A small broad positive Lorentzian enforces numerical nonnegative spectrum
    # over a dense logarithmic audit grid and makes the leading 1/w^2 tail positive.
    w=np.r_[0.,-np.logspace(-5,8,30000),np.logspace(-5,8,30000)]
    S=spectrum(w,c,nu);G=settings['positivity_pole_ps']
    correction=max(0.,float(np.max(-S*(G*G+w*w)/(2*G))),float(-np.real(np.sum(c*nu))/G))
    correction*=1.05
    if correction:
        c=np.r_[c,correction];nu=np.r_[nu,G+0j]
    order=np.argsort(nu.real);c=c[order];nu=nu[order]
    return c,nu,{'rank':rank,'terms':len(c),'removed_imaginary_C0':imaginary_intercept,
                 'positive_spectrum_correction_ps2_unit_j0':correction,
                 'leading_singular_values':s[:24].tolist()}


def audit(c,nu,settings):
    short=np.linspace(0,2,2001)
    long=np.geomspace(2.001,settings['audit_horizon_ps'],1200)
    t=np.r_[short,long];exact=continuum(t,settings['E0_cm'],settings['temperature_K'])
    fit=correlation(t,c,nu);C0=float(exact[0].real)
    E=np.linspace(10,settings['rate_gap_max_cm'],501)
    down,up=spectral_rates(E,settings['E0_cm'],settings['temperature_K'])
    fd=spectrum(E/HBAR,c,nu);fu=spectrum(-E/HBAR,c,nu)
    w=np.r_[0.,-np.logspace(-5,8,30000),np.logspace(-5,8,30000)]
    S=spectrum(w,c,nu)
    rate_error=float(max(np.max(abs(fd-down)),np.max(abs(fu-up)))/down.max())
    ratio_error=float(np.max(abs(fu/fd-np.exp(-E/(KB*settings['temperature_K'])))))
    integral_error=float(trapezoid(abs(fit-exact),t)/trapezoid(abs(exact),t))
    metrics={'correlation_max_error_over_C0':float(np.max(abs(fit-exact))/C0),
             'correlation_integrated_relative_error':integral_error,
             'spectrum_min_over_peak':float(S.min()/S.max()),
             'spectral_rate_max_error_over_peak':rate_error,'Boltzmann_ratio_max_absolute_error':ratio_error,
             'audit_horizon_ps':settings['audit_horizon_ps'],'continuum_C0_unit_j0_ps2':C0,
             'fitted_C0_unit_j0_ps2':float(c.sum().real)}
    metrics['passed']=bool(metrics['correlation_max_error_over_C0']<settings['correlation_tolerance']
        and integral_error<settings['integral_tolerance'] and rate_error<settings['rate_tolerance']
        and ratio_error<settings['balance_tolerance'] and metrics['spectrum_min_over_peak']>-1e-10)
    return metrics,(t,exact,fit),(E,down,up,fd,fu)
