"""E-J2 quick check (parallel to theory). Uses ej_check primitives.
N=2^14, finest target level LT=4 (16-sample cells), cubic interpolation of Psi1'f, exact trig warp of f."""
import numpy as np, json, time, sys
from pathlib import Path
import ej_check as E
E.LT=4; LT=4; N=E.N; x=E.x
OUT=Path(__file__).with_name('ej_run.json')
def tau_smooth(eps,P=N//8): return eps*P/(2*np.pi)*np.sin(2*np.pi*x/P)
def tau_multi(eps,seed=7):
    rng=np.random.default_rng(seed); Ps=[2**k for k in range(4,12)]; ph=rng.uniform(0,2*np.pi,len(Ps))
    return sum(eps/len(Ps)*P/(2*np.pi)*np.sin(2*np.pi*x/P+p) for P,p in zip(Ps,ph))
def inputs(w,lmax,lmin):
    d={}
    for s in range(3):
        for nm,F in E.make_inputs(w,0,s).items(): d[f'{nm}{s}']=F
    d['balanced']=E.balanced(w,lmax,lmin); return d
def er(F,w,lmax,lmin,tau): return E.err(F,w,lmax,lmin,tau,'cubic')
res=[]; t0=time.time()
def save(): json.dump(res,open(OUT,'w'),indent=0)
for w in ['haar','sym4']:
    for tn,tf in [('smooth',tau_smooth),('multi',tau_multi)]:
        tau=tf(0.1)
        for J in range(0,7):
            lmin=LT; lmax=LT+J+1
            for nm,F in inputs(w,lmax,lmin).items():
                res.append(dict(kind='total',w=w,tau=tn,eps=0.1,J=J,input=nm,err=er(F,w,lmax,lmin,tau)))
            res.append(dict(kind='floor',w=w,tau=tn,eps=0.0,J=J,input='pink0_shift0.5',err=er(E.make_inputs(w,0,0)['pink'],w,lmax,lmin,np.full(N,0.5))))
            if tn=='smooth':
                F=E.balanced(w,lmax,lmin)
                res.append(dict(kind='total',w=w,tau=tn,eps=0.2,J=J,input='balanced',err=er(F,w,lmax,lmin,tf(0.2))))
            save(); print(w,tn,J,f'{time.time()-t0:.0f}s',flush=True)
        # single-scale diagnostic at J=6 window, for balanced(J=6) and pink0
        lmaxW=LT+7; Fb=E.balanced(w,lmaxW,LT); Fp=E.make_inputs(w,0,0)['pink']
        for j in range(7):
            for nm,F in [('balanced',Fb),('pink0',Fp)]:
                res.append(dict(kind='single',w=w,tau=tn,eps=0.1,J=6,level=j,input=nm,err=er(F,w,LT+j+1,LT+j,tau)))
        for nm,F in [('balanced',Fb),('pink0',Fp)]:
            res.append(dict(kind='total_window',w=w,tau=tn,eps=0.1,J=6,input=nm,err=er(F,w,lmaxW,LT,tau)))
        save(); print(w,tn,'diag done',f'{time.time()-t0:.0f}s',flush=True)
print('DONE')
