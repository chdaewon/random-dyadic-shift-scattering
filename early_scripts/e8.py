import numpy as np
from mult import psi_samples, ft
def m0(name,m,t):
    x,p,dx=psi_samples(name,10); ph=ft(x,p,dx,t); p2=ft(x,p,dx,t/2)
    return np.conj(ph)*0.5*p2*np.exp(-1j*np.pi*m*t)*(1-np.exp(-1j*np.pi*t))
u=np.linspace(-10,6,1600); t=2.0**u
for name in ['db4','sym4']:
    for m in [2,3,4,5]:
        k=np.trapezoid(m0(name,m,t),u)
        s=np.linspace(0,1,17)[:-1]; js=np.arange(-12,9)
        MT=np.array([np.sum(m0(name,m,2.0**(ss+js))) for ss in s])
        print(f"{name} m={m}: TD kappa={k:.3f}  T-only Im[{MT.imag.min():.3f},{MT.imag.max():.3f}] Re[{MT.real.min():.3f},{MT.real.max():.3f}]")
