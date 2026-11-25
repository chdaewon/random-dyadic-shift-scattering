"""Independent recomputation of Table 1 (aligned constants) from the PyWavelets filters.
psi_hat(xi)=m1(xi/2) prod_{k>=2} m0(xi/2^k), m0/m1 the low/high-pass reconstruction symbols with m0(0)=1
(40 factors); kappa_{psi,m}=(1/ln2) int_0^inf conj(psi_hat(t)) Theta_hat(t) dt/t, Theta_hat(t)=1/2 psi_hat(t/2) e^{-pi i m t}(1-e^{-pi i t})."""
import numpy as np, pywt, json
def sym(f,xi): k=np.arange(len(f)); return (f[None,:]*np.exp(-2j*np.pi*np.outer(xi,k))).sum(1)/np.sqrt(2)
def psihat(name,xi):
    if name=='haar':
        x=np.where(xi==0,1e-300,xi); return np.exp(-1j*np.pi*x)*2j*np.sin(np.pi*x/2)**2/(np.pi*x)
    w=pywt.Wavelet(name); h=np.array(w.rec_lo); g=np.array(w.rec_hi)
    out=sym(g,xi/2)
    for k in range(2,42): out=out*sym(h,xi/2**k)
    return out
def kappa(name,m,T=600.0,dt=0.004):
    t=np.arange(dt/2,T,dt)                           # midpoint rule on (0,T]
    vals=[]
    ph=psihat(name,t); ph2=psihat(name,t/2)
    Th=0.5*ph2*np.exp(-1j*np.pi*m*t)*(1-np.exp(-1j*np.pi*t))
    return (np.conj(ph)*Th/t).sum()*dt/np.log(2)
res={}
for name,L in [('haar',1),('db2',3),('db4',7),('sym4',7),('sym6',11),('coif3',17)]:
    ks={m:kappa(name,m) for m in range(L)}
    ms=max(ks,key=lambda m:abs(ks[m].imag)); k=ks[ms]
    res[name]=dict(L=L,mstar=ms,kappa=[k.real,k.imag],c0=1/abs(k.imag),re_over_im=abs(k.real)/abs(k.imag),c0_m0=1/abs(ks[0].imag))
    print(f"{name:6s} L={L:2d} m*={ms:2d} kappa={k.real:+.4f}{k.imag:+.4f}i |c0|={1/abs(k.imag):.3f} Re/Im={100*abs(k.real)/abs(k.imag):.2f}%  |c0|(m=0)={1/abs(ks[0].imag):.1f}")
json.dump(res,open('recompute_tab1.json','w'),indent=1)
