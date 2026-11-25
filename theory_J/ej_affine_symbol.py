"""Exact lattice-averaged symbol for an affine warp tau(x)=c+a x  (Note T-J4, Theorem B):
   M_J(xi) = avg_x | sum_{j=0}^J mu(2^-j xi, 2^j x) |^2,  mu(s,t)=sum_k Theta(t-k)[F_{t-k}(s) - lam F_{t-k}(lam s)],
   F_y(s)=psihat(-s) e^{-2 pi i s y},  lam=1/(1-a).   sup_xi M_J = sup_f E_beta ||G~_beta f||^2/||f||^2."""
import numpy as np, pywt, json, sys
from pathlib import Path
def psihat_factory(name,K=40):
    if name=='haar':
        def ph(s):
            s=np.asarray(s,float); out=np.zeros(s.shape,complex); nz=np.abs(s)>1e-12
            out[nz]=(1-np.exp(-1j*np.pi*s[nz]))**2/(2j*np.pi*s[nz]); return out
        return ph,1,0
    w=pywt.Wavelet(name); lo=np.array(w.rec_lo); hi=np.array(w.rec_hi); n=np.arange(len(lo))
    H=lambda x:(lo[None,:]*np.exp(-2j*np.pi*np.outer(x,n))).sum(1)/np.sqrt(2)
    G=lambda x:(hi[None,:]*np.exp(-2j*np.pi*np.outer(x,n))).sum(1)/np.sqrt(2)
    def ph(s):
        s=np.asarray(s,float); sh=s.shape; s=s.ravel(); out=G(s/2)
        for k in range(2,K+2): out=out*H(s/2**k)
        return out.reshape(sh)
    L=len(lo)-1; m={'sym4':3,'db4':4,'sym6':5,'db2':1}[name]
    return ph,L,m
def theta_factory(name,L,m):
    if name=='haar':
        h=lambda t: np.where((t>=0)&(t<.5),1.,np.where((t>=.5)&(t<1),-1.,0.))
    else:
        _,ps,xx=pywt.Wavelet(name).wavefun(level=12); h=lambda t: np.interp(t,xx,ps,left=0.,right=0.)
    return lambda t: h(2*t-m)-h(2*t-m-1)
def M(name,a,J,xis,spc=32):
    ph,L,m=psihat_factory(name); Th=theta_factory(name,L,m); lam=1/(1-a)
    nx=spc*2**J; t=(np.arange(nx)+0.5)/nx              # x in [0,1), spc samples per finest cell
    pre=[]
    for j in range(J+1):
        tt=2**j*t; fl=np.floor(tt).astype(int)
        pre.append([(tt-(fl-d),Th(tt-(fl-d))) for d in range(L+1)])
    out=[]
    for xi in xis:
        acc=np.zeros(nx,complex)
        for j in range(J+1):
            s=xi/2**j; p0=ph(np.array([-s]))[0]; p1=ph(np.array([-lam*s]))[0]
            for y,th in pre[j]:
                acc+=th*(p0*np.exp(-2j*np.pi*s*y)-lam*p1*np.exp(-2j*np.pi*lam*s*y))
        out.append(float(np.mean(np.abs(acc)**2)))
    return np.array(out)
if __name__=='__main__':
    name=sys.argv[1]; res=[]
    xis=2.0**np.linspace(-2,16,145)
    for a in [0.05,0.1,0.3]:
        for J in [2,5,8,11]:
            Mv=M(name,a,J,xis[xis<2**(J+6)])
            res.append(dict(name=name,a=a,J=J,sup=float(np.sqrt(Mv.max())),sup_over_a=float(np.sqrt(Mv.max())/a)))
            print(name,a,J,'sup sqrt(M)/a =',round(np.sqrt(Mv.max())/a,4),flush=True)
    json.dump(res,open(Path(__file__).with_name(f'ej_affine_symbol_{name}.json'),'w'),indent=0)
