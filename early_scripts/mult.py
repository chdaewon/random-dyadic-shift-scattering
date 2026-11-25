import numpy as np, pywt
def psi_samples(name, level=12):
    if name=='haar':
        x=np.linspace(0,1,2**level,endpoint=False); p=np.where(x<0.5,1.0,-1.0); dx=x[1]-x[0]; return x,p,dx
    w=pywt.Wavelet(name); phi,psi,x=w.wavefun(level=level); dx=x[1]-x[0]
    psi=psi/np.sqrt(np.sum(psi**2)*dx); return x,psi,dx
def ft(x,p,dx,xi):  # \hat g(xi)=\int g e^{-2pi i x xi}
    return np.array([np.sum(p*np.exp(-2j*np.pi*x*z))*dx for z in np.atleast_1d(xi)])
def m0(name,xi,cache={}):
    x,p,dx=psi_samples(name)
    ph=ft(x,p,dx,xi); ph2=ft(x,p,dx,xi/2)
    Th=0.5*ph2*(1-np.exp(-1j*np.pi*xi))
    return np.conj(ph)*Th
for name in ['haar','db2','db4','sym4','coif3']:
    # dilation+translation average: (1/ln2) \int_0^inf m0(t) dt/t, on log grid
    u=np.linspace(-14,8,4000); t=2.0**u
    mt=m0(name,t)
    MTD=np.trapezoid(mt,u)   # dt/t = ln2 du, divided by ln2
    # translation-only: sum_j m0(2^{-j} xi) for xi = 2^{s}, s in [0,1)
    s=np.linspace(0,1,41)[:-1]
    MT=[]
    for ss in s:
        js=np.arange(-14,9); MT.append(np.sum(m0(name,2.0**(ss+js))))
    MT=np.array(MT)
    print(f"{name:6s} TD-avg = {MTD.real:+.4f}{MTD.imag:+.4f}i | T-only: Im range [{MT.imag.min():+.4f},{MT.imag.max():+.4f}] Re range [{MT.real.min():+.4f},{MT.real.max():+.4f}] ripple(Im)={(MT.imag.max()-MT.imag.min())/abs(MT.imag.mean()):.3f}")
    np.save(f'MT_{name}.npy',np.vstack([s,MT.real,MT.imag]))
