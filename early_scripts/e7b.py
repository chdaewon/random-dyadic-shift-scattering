import numpy as np
from shift import Shift
N=2**13; J=6; lmax=J+3; lmin=2
n=np.arange(N)
for w,m in [('haar',0),('db4',0),('db4',1),('db4',-1),('sym4',0),('sym4',1),('sym4',-1)]:
    S=Shift(w,N,lmax,lmin,m)
    ks=np.unique(np.round(2**np.linspace(np.log2(N/2**lmax*2),np.log2(N/2**(lmin+3)),24)).astype(int))
    m_=np.array([np.vdot(np.exp(2j*np.pi*k*n/N),S.apply(np.cos(2*np.pi*k*n/N))+1j*S.apply(np.sin(2*np.pi*k*n/N)))/N for k in ks])
    print(f"discrete {w} m={m}: Im[{m_.imag.min():.3f},{m_.imag.max():.3f}] Re[{m_.real.min():.3f},{m_.real.max():.3f}]")
