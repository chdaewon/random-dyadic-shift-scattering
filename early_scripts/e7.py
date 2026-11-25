import numpy as np
from shift import Shift
N=2**13; J=6; lmax=J+3; lmin=2
n=np.arange(N)
for w,m in [('haar',0),('db4',0),('db4',4),('sym4',3)]:
    S=Shift(w,N,lmax,lmin,m)
    # adjoint check
    rng=np.random.default_rng(0); a=rng.standard_normal(N); b=rng.standard_normal(N)
    adj=abs(np.dot(S.apply(a),b)-np.dot(a,S.adj(b)))
    iso=np.linalg.norm(S.apply(a))
    ks=np.unique(np.round(2**np.linspace(np.log2(N/2**lmax*4),np.log2(N/2**(lmin+2)),24)).astype(int))
    m_=np.array([np.vdot(np.exp(2j*np.pi*k*n/N),S.apply(np.cos(2*np.pi*k*n/N))+1j*S.apply(np.sin(2*np.pi*k*n/N)))/N for k in ks])
    print(f"{w} m={m}: adj err {adj:.1e}; Im range [{m_.imag.min():.3f},{m_.imag.max():.3f}] Re range [{m_.real.min():.3f},{m_.real.max():.3f}]")
