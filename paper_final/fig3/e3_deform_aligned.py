# Re-run of e3_deform.py with aligned generators: Haar (m_d=0), sym4 (m_d=0, m=3), db4 (m_d=1, m=4).
import numpy as np, json, sys
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
def warp(u,tau):
    cs=CubicSpline(np.append(x,N),np.append(u,u[0]),bc_type='periodic')
    return cs(np.mod(x-tau,N))
def run(S,eps,P):
    tau=eps*P/(2*np.pi)*np.sin(2*np.pi*x/P)
    ftau=fev(x-tau)
    ra=np.linalg.norm(Psi(S,ftau)-warp(Psi(S,f),tau))/nf
    rs=np.linalg.norm(Psi(S,ftau,False)-warp(Psi(S,f,False),tau))/nf
    return ra,rs
EPS=[0.0025,0.005,0.01,0.02,0.04,0.08]; PS=[256,512,1024,2048,4096,8192]
out={'eps':EPS,'P':PS,'N':N,'lmax':lmax,'lmin':lmin}
for w,md in [('haar',0),('sym4',0),('db4',1)]:
    S=Shift(w,N,lmax,lmin,m=md)
    r=[run(S,e,N) for e in EPS]; out[w+'_eps']=[[a for a,b in r],[b for a,b in r]]
    r=[run(S,0.02,P) for P in PS]; out[w+'_P']=[[a for a,b in r],[b for a,b in r]]
    # floor: constant half-sample shift is not smooth-periodic; use tau == 0 instead
    print(w,md,'eps avg',np.round(out[w+'_eps'][0],5),'single',np.round(out[w+'_eps'][1],4))
    print(w,md,'P   avg',np.round(out[w+'_P'][0],5),'single',np.round(out[w+'_P'][1],4),flush=True)
json.dump(out,open('e3_aligned.json','w'),indent=1)
