"""Lanczos (full reorthogonalisation) for sigma_max^2 = lambda_max(A^*A), operator from ej_worst.py.
Reports top Ritz value theta, residual r=||A^*A y - theta y|| (an eigenvalue lies in [theta-r, theta+r]),
the second Ritz value, and the Kato-Temple bound r^2/gap.  Optional: full matrix on the 1024-dim band space."""
import numpy as np, json, sys, time
from pathlib import Path
sys_argv=sys.argv[:]; sys.argv=['x','haar','0']
import ej_worst as W
sys.argv=sys_argv
N=W.N; x=W.x; kmax=W.kmax
def tau_of(warp,eps):
    if warp=='smooth':
        Pw=N//4; return eps*Pw/(2*np.pi)*np.sin(2*np.pi*x/Pw)
    rng=np.random.default_rng(11); Ps=[2**p for p in range(4,13)]; ph=rng.uniform(0,2*np.pi,len(Ps))
    return sum(eps/len(Ps)*P/(2*np.pi)*np.sin(2*np.pi*x/P+f) for P,f in zip(Ps,ph))
def lanczos(op,v0,m):
    V=[v0/np.linalg.norm(v0)]; al=[]; be=[]
    for k in range(m):
        w=op(V[-1]); a=float(V[-1]@w); al.append(a)
        Vm=np.array(V); w=w-Vm.T@(Vm@w); w=w-Vm.T@(Vm@w)      # full reorth, twice
        b=np.linalg.norm(w)
        T=np.diag(al)+np.diag(be,1)+np.diag(be,-1)
        th,Y=np.linalg.eigh(T); res=b*abs(Y[-1,-1])
        if k>=3 and res<1e-9*th[-1]: break
        be.append(b); V.append(w/b)
    Vm=np.array(V[:len(al)]); y=Vm.T@Y[:,-1]; r=np.linalg.norm(op(y)-th[-1]*y)
    return th, r, len(al), y
def full_matrix(op):
    k=np.arange(1,kmax+1); B=[]
    for kk in k: B.append(np.cos(2*np.pi*kk*x/N)); B.append(np.sin(2*np.pi*kk*x/N))
    B=np.array(B); B/=np.linalg.norm(B,axis=1)[:,None]
    M=np.array([B@op(b) for b in B]); asym=np.abs(M-M.T).max()/np.abs(M).max()
    ev=np.linalg.eigvalsh((M+M.T)/2); return ev, asym
if __name__=='__main__':
    w,eps,warp=sys.argv[1],float(sys.argv[2]),sys.argv[3]; Js=[int(j) for j in sys.argv[4].split(',')]
    m=int(sys.argv[5]); full=sys.argv[6].split(',') if len(sys.argv)>6 and sys.argv[6] else []
    OUT=Path(f'lz_{w}_{warp}_eps{eps}.json'); res=json.load(open(OUT)) if OUT.exists() else []
    tau=tau_of(warp,eps); t0=time.time()
    for J in Js:
        if any(r['J']==J for r in res): continue
        A=W.Op(w,J,tau); rng=np.random.default_rng(J)
        v0=W.balanced(w,A.lmax,A.lmin)+0.05*W.band(rng.standard_normal(N))
        th,r,steps,y=lanczos(A.AtA,v0,m)
        gap=th[-1]-th[-2]
        d=dict(w=w,warp=warp,eps=eps,J=J,steps=steps,theta=float(th[-1]),theta2=float(th[-2]),resid=float(r),
               sigma=float(np.sqrt(th[-1])),sigma_over_eps=float(np.sqrt(th[-1])/eps),
               sigma_hi=float(np.sqrt(th[-1]+r)),kato_temple=float(r*r/gap) if gap>0 else None)
        if str(J) in full:
            ev,asym=full_matrix(A.AtA); d.update(full_lmax=float(ev[-1]),full_l2=float(ev[-2]),full_asym=float(asym),
                     full_sigma_over_eps=float(np.sqrt(ev[-1])/eps))
        res.append(d); json.dump(res,open(OUT,'w'),indent=0)
        print(w,warp,eps,J,steps,'sig/eps %.4f'%d['sigma_over_eps'],'rel res %.1e'%(r/th[-1]),'gap %.3e'%gap,
              ('full %.6f asym %.1e'%(d['full_sigma_over_eps'],d['full_asym'])) if 'full_lmax' in d else '', f'{time.time()-t0:.0f}s',flush=True)
    print('DONE',flush=True)
