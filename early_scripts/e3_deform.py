import numpy as np
from scipy.interpolate import CubicSpline
from shift import Shift
N=2**13; lmax=9; lmin=3; Pcell=2**lmax
rng=np.random.default_rng(1)
K=N//16; ks=np.arange(1,K+1); amp=rng.standard_normal(K)/np.sqrt(ks); ph=rng.uniform(0,2*np.pi,K)
x=np.arange(N,dtype=float)
def fev(y):
    out=np.zeros_like(y)
    for c in range(0,K,128):
        kk=ks[c:c+128,None]; out+=np.sum(amp[c:c+128,None]*np.cos(2*np.pi*kk*y[None,:]/N+ph[c:c+128,None]),0)
    return out
f=fev(x); nf=np.linalg.norm(f)
def Psi(S,g,avg=True):
    if not avg: return np.abs(S.apply(g))
    acc=np.zeros(N)
    for b in range(Pcell): acc+=np.abs(S.at(g,b))
    return acc/Pcell
def warp(u,tau):   # L_tau u (x) = u(x - tau(x)), periodic cubic spline
    cs=CubicSpline(np.append(x,N),np.append(u,u[0]),bc_type='periodic')
    return cs(np.mod(x-tau,N))
def run(w,eps,P):
    S=Shift(w,N,lmax,lmin)
    tau=eps*P/(2*np.pi)*np.sin(2*np.pi*x/P)
    ftau=fev(x-tau)
    ra=np.linalg.norm(Psi(S,ftau)-warp(Psi(S,f),tau))/nf
    rs=np.linalg.norm(Psi(S,ftau,False)-warp(Psi(S,f,False),tau))/nf
    return ra,rs
# interpolation floor check: pure integer translation via tau const is not smooth-periodic; check eps=0
out={}
for w in ['haar','db4']:
    A=[];B=[]
    for eps in [0.0025,0.005,0.01,0.02,0.04,0.08]:
        ra,rs=run(w,eps,N); A.append(ra);B.append(rs)
    out[w+'_eps']=(A,B); print(w,'eps sweep avg',np.round(A,4),'single',np.round(B,4))
    A=[];B=[]
    for P in [256,512,1024,2048,4096,8192]:
        ra,rs=run(w,0.02,P); A.append(ra);B.append(rs)
    out[w+'_P']=(A,B); print(w,'P sweep  avg',np.round(A,4),'single',np.round(B,4))
np.save('e3.npy',out,allow_pickle=True)
