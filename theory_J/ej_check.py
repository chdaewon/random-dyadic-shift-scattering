"""Parallel check for the (J+1) question.
Covariance error  E(f) = ||Psi1'(L_tau f) - L_tau Psi1' f|| / ||f||  with an exhaustive offset average.
Scales: target levels lt..lt+J (J+1 shift channels), finest target level lt fixed (scale invariance)."""
import numpy as np, pywt, json, sys, time
from pathlib import Path
SQ=np.sqrt(2)
N=2**14; LT=3                       # finest target level: cells of 8 samples
x=np.arange(N,dtype=float)

def shift_batch(G,w,lmax,lmin,md=0):
    """rows of G -> S(rows); source levels lmin+1..lmax, target level l-1."""
    c=pywt.wavedec(G,w,mode='periodization',level=lmax,axis=1)
    o=[np.zeros_like(a) for a in c]
    idx=lambda l: lmax-l+1
    for l in range(lmin+1,lmax+1):
        d=c[idx(l)]; t=o[idx(l-1)]
        inter=np.empty((d.shape[0],2*d.shape[1])); inter[:,0::2]=d; inter[:,1::2]=-d
        t+=np.roll(inter/SQ,md,axis=1)
    return pywt.waverec(o,w,mode='periodization',axis=1)

def Psi1(g,w,lmax,lmin,chunk=256):
    P=2**lmax; acc=np.zeros(N); ar=np.arange(N)
    for b0 in range(0,P,chunk):
        bs=np.arange(b0,min(P,b0+chunk))
        G=g[(ar[None,:]+bs[:,None])%N]            # roll(g,-b)
        Y=shift_batch(G,w,lmax,lmin)
        Y=np.take_along_axis(Y,(ar[None,:]-bs[:,None])%N,axis=1)  # roll back by b
        acc+=np.abs(Y).sum(0)
    return acc/P

_UP=16
def trig_eval(F,y):
    """evaluate the band-limited periodic signal with rfft coefficients F at points y:
    FFT zero-padding by 16x, then periodic cubic spline on the fine grid (rel. error ~1e-6)."""
    from scipy.interpolate import CubicSpline
    M=N*_UP; G=np.zeros(M//2+1,complex); G[:len(F)]=F
    if len(F)==N//2+1: G[N//2]*=0.5          # split Nyquist
    fine=np.fft.irfft(G,M)*_UP
    xf=np.arange(M+1)/_UP
    cs=CubicSpline(xf,np.append(fine,fine[0]),bc_type='periodic')
    return cs(np.mod(y,N))

def bandlimit(f,kmax):
    F=np.fft.rfft(f); F[kmax+1:]=0; return F

def interp(u,y,kind):
    y=np.mod(y,N)
    if kind=='linear':
        i=np.floor(y).astype(int); t=y-i; return (1-t)*u[i]+t*u[(i+1)%N]
    from scipy.interpolate import CubicSpline
    cs=CubicSpline(np.append(x,N),np.append(u,u[0]),bc_type='periodic'); return cs(y)

def make_inputs(w,J,seed):
    rng=np.random.default_rng(seed); kmax=N//(2**LT)   # band covering target scales
    out={}
    F=np.zeros(N//2+1,complex); k=np.arange(1,kmax+1)
    F[1:kmax+1]=(rng.standard_normal(kmax)+1j*rng.standard_normal(kmax)); out['white']=F.copy()
    F[1:kmax+1]*=k**-0.5; out['pink']=F
    return out

def balanced(w,lmax,lmin,center=N//2+3):
    """sum over source levels of one wavelet each, all touching the same point, equal L2 energy."""
    c=pywt.wavedec(np.zeros(N),w,mode='periodization',level=lmax)
    for l in range(lmin+1,lmax+1):
        d=c[lmax-l+1]; d[(center>>l)%len(d)]=1.0
    f=pywt.waverec(c,w,mode='periodization')
    return np.fft.rfft(f)            # full band, exact representation

def err(F,w,lmax,lmin,tau,kind):
    f=trig_eval(F,x); ft=trig_eval(F,x-tau)
    a=Psi1(ft,w,lmax,lmin); b=interp(Psi1(f,w,lmax,lmin),x-tau,kind)
    return np.linalg.norm(a-b)/np.linalg.norm(f)

if __name__=='__main__':
    Jlist=list(range(0,8)); res=[]; t0=time.time()
    eps=0.02
    for w in ['haar','sym4']:
        for P in [N//8,N//64]:
            tau=eps*P/(2*np.pi)*np.sin(2*np.pi*x/P)
            for J in Jlist:
                lmin=LT; lmax=LT+J+1          # source levels LT+1..LT+J+1 -> J+1 channels
                ins={}
                for s in range(3):
                    for nm,F in make_inputs(w,J,s).items(): ins[f'{nm}{s}']=F
                ins['balanced']=balanced(w,lmax,lmin)
                for nm,F in ins.items():
                    e=err(F,w,lmax,lmin,tau,'linear')
                    res.append(dict(w=w,P=P,J=J,input=nm,err=e))
                # floor: constant translation by ||tau||_inf (eps=0 -> theory gives 0)
                a=np.full(N,eps*P/(2*np.pi))
                res.append(dict(w=w,P=P,J=J,input='floor_const_shift',err=err(ins['pink0'],w,lmax,lmin,a,'linear')))
                print(w,P,J,f'{time.time()-t0:.0f}s',{r['input']:round(r['err'],5) for r in res if r['w']==w and r['P']==P and r['J']==J},flush=True)
                json.dump(res,open(Path(__file__).with_name('ej_check.json'),'w'),indent=0)
