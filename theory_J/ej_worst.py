"""Worst-case input for the linear defect  (Q_b f)(x) = S_b L_tau f (x) - S_b T_{tau(x)} f (x),  b = all lattice offsets.
sigma_max^2 = max_f  E_b ||Q_b f||^2 / ||f||^2  over band-limited f, by power iteration on A^T A.
T_theta handled exactly by FFT on K+3 theta-nodes and 4-point Lagrange interpolation in theta."""
import numpy as np, pywt, json, sys, time
from pathlib import Path
SQ=np.sqrt(2)
N=int(sys.argv[3]) if len(sys.argv)>3 else 2**13
LT=3; K=4
x=np.arange(N); kmax=N//2**(LT+1)          # band: periods >= 16 samples

def shift_rows(G,w,lmax,lmin,adj=False):
    c=pywt.wavedec(G,w,mode='periodization',level=lmax,axis=1)
    o=[np.zeros_like(a) for a in c]; idx=lambda l: lmax-l+1
    for l in range(lmin+1,lmax+1):
        if not adj:
            d=c[idx(l)]; t=o[idx(l-1)]; t[:,0::2]+=d/SQ; t[:,1::2]-=d/SQ
        else:
            t=c[idx(l-1)]; o[idx(l)]+= (t[:,0::2]-t[:,1::2])/SQ
    return pywt.waverec(o,w,mode='periodization',axis=1)

def S_all(g,w,lmax,lmin,adj=False,chunk=256):
    """rows b: S_b g  (S_b = T_b S T_{-b}) for b=0..P-1 -> array [P,N]"""
    P=2**lmax; out=np.empty((P,N)); ar=np.arange(N)
    for b0 in range(0,P,chunk):
        bs=np.arange(b0,min(P,b0+chunk))
        G=g[(ar[None,:]+bs[:,None])%N]
        Y=shift_rows(G,w,lmax,lmin,adj)
        out[bs]=np.take_along_axis(Y,(ar[None,:]-bs[:,None])%N,axis=1)
    return out
def S_all_adj_sum(Z,w,lmax,lmin,chunk=256):
    """sum_b S_b^* Z[b]"""
    P=Z.shape[0]; acc=np.zeros(N); ar=np.arange(N)
    for b0 in range(0,P,chunk):
        bs=np.arange(b0,min(P,b0+chunk))
        G=np.take_along_axis(Z[bs],(ar[None,:]+bs[:,None])%N,axis=1)   # T_{-b}
        Y=shift_rows(G,w,lmax,lmin,adj=True)
        acc+=np.take_along_axis(Y,(ar[None,:]-bs[:,None])%N,axis=1).sum(0)
    return acc

def band(f):
    F=np.fft.rfft(f); F[kmax+1:]=0; F[0]=0; return np.fft.irfft(F,N)
def Tshift(f,a):  # f(x-a), exact for band-limited f
    F=np.fft.rfft(f); k=np.arange(len(F)); return np.fft.irfft(F*np.exp(-2j*np.pi*k*a/N),N)

class Op:
    def __init__(s,w,J,tau):
        s.w=w; s.lmin=LT; s.lmax=LT+J+1; s.P=2**s.lmax
        n=np.floor(tau).astype(int); th=tau-n; s.n=n
        s.nodes=np.arange(-1,K+2)/K                     # theta nodes
        i=np.minimum((th*K).astype(int),K-1); t=th*K-i  # cell, local coord in [0,1)
        L=[(-t*(t-1)*(t-2)/6),((t+1)*(t-1)*(t-2)/2),(-(t+1)*t*(t-2)/2),((t+1)*t*(t-1)/6)]
        s.W=np.zeros((len(s.nodes),N))
        for q in range(4): s.W[i+q,x]+=L[q]            # node index i-1+q+1 = i+q
        s.xs=(x-n)%N
    def Ltau(s,U):     # sum_m W_m(x) U_m(x-n(x))
        return np.sum(s.W*U[:,s.xs],0)
    def Ltau_adj(s,g):
        Ub=np.zeros((len(s.nodes),N))
        for m in range(len(s.nodes)): np.add.at(Ub[m],s.xs,s.W[m]*g)
        return Ub
    def fwd(s,f):
        U=np.stack([Tshift(f,a) for a in s.nodes])
        t1=S_all(s.Ltau(U),s.w,s.lmax,s.lmin)
        P=s.P; bb=np.arange(P)[:,None]; rows=(bb-s.n[None,:])%P
        t2=np.zeros((P,N))
        for m in range(len(s.nodes)):
            if not s.W[m].any(): continue
            Y=S_all(U[m],s.w,s.lmax,s.lmin)
            t2+=s.W[m][None,:]*Y[rows,s.xs[None,:]]
        return t1-t2
    def adj(s,Z):
        P=s.P; bb=np.arange(P)[:,None]; rows=(bb-s.n[None,:])%P
        g=S_all_adj_sum(Z,s.w,s.lmax,s.lmin)
        Ub=s.Ltau_adj(g)
        for m in range(len(s.nodes)):
            if not s.W[m].any(): continue
            Yb=np.zeros((P,N)); np.add.at(Yb,(rows,np.broadcast_to(s.xs[None,:],(P,N))),s.W[m][None,:]*Z)
            Ub[m]-=S_all_adj_sum(Yb,s.w,s.lmax,s.lmin)
        return sum(Tshift(Ub[m],-a) for m,a in enumerate(s.nodes))
    def AtA(s,f): return band(s.adj(s.fwd(band(f))))/s.P

def balanced(w,lmax,lmin,center=None):
    c=pywt.wavedec(np.zeros(N),w,mode='periodization',level=lmax)
    center=N//2+3 if center is None else center
    for l in range(lmin+1,lmax+1): d=c[lmax-l+1]; d[(center>>l)%len(d)]=1.0
    return band(pywt.waverec(c,w,mode='periodization'))

if __name__=='__main__':
    w=sys.argv[1]; Js=[int(j) for j in sys.argv[2].split(',')]; eps=float(sys.argv[4]) if len(sys.argv)>4 else 0.1
    OUT=Path(__file__).with_name(f'ej_worst_{w}_N{N}_eps{eps}.json'); res=[]
    warp=sys.argv[5] if len(sys.argv)>5 else 'smooth'; iters=int(sys.argv[6]) if len(sys.argv)>6 else 14
    if warp=='smooth':
        Pw=N//4; tau=eps*Pw/(2*np.pi)*np.sin(2*np.pi*x/Pw)
    else:   # multi-scale: periods 2^4..2^12 samples, ||tau'||<=eps
        rng=np.random.default_rng(11); Ps=[2**p for p in range(4,13)]; ph=rng.uniform(0,2*np.pi,len(Ps))
        tau=sum(eps/len(Ps)*P/(2*np.pi)*np.sin(2*np.pi*x/P+f) for P,f in zip(Ps,ph))
    OUT=Path(__file__).with_name(f'ej_worst_{w}_N{N}_eps{eps}_{warp}.json')
    t0=time.time()
    for J in Js:
        A=Op(w,J,tau); rng=np.random.default_rng(J)
        f=balanced(w,A.lmax,A.lmin)+0.05*band(rng.standard_normal(N)); f/=np.linalg.norm(f)
        hist=[]
        for it in range(iters):
            g=A.AtA(f); rq=float(f@g); hist.append(np.sqrt(rq)); f=g/np.linalg.norm(g)
        # localisation diagnostics of the maximiser
        F=np.abs(np.fft.rfft(f))**2; k=np.arange(len(F)); kc=float((k*F).sum()/F.sum())
        e=f**2; xc=np.angle((e*np.exp(2j*np.pi*x/N)).sum())*N/(2*np.pi)%N
        spread=float(np.sqrt((e*(((x-xc+N/2)%N)-N/2)**2).sum()))
        res.append(dict(w=w,J=J,eps=eps,N=N,sigma_hist=hist,sigma=hist[-1],sigma_over_eps=hist[-1]/eps,k_centroid=kc,x_spread=spread))
        json.dump(res,open(OUT,'w'),indent=0); np.save(OUT.with_suffix(f'.J{J}.npy'),f)
        print(w,J,'sigma',np.round(hist,4),'kc',round(kc,1),'spread',round(spread,1),f'{time.time()-t0:.0f}s',flush=True)
    print('DONE')
