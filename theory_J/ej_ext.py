"""Numerical counterparts of thm:sep, cor:depth2, cor:dil.
Setup as Table 2 (ej_run.py): N=2^14, finest target level LT=4 (16-sample cells), smooth warp P=N/8, eps=0.1,
exhaustive lattice average, exact trigonometric warp of f, cubic interpolation for L_tau of feature maps."""
import numpy as np, json, sys, time
from pathlib import Path
import ej_check as E
E.LT=4; LT=4; N=E.N; x=E.x
def tau_smooth(eps,P=N//8): return eps*P/(2*np.pi)*np.sin(2*np.pi*x/P)
def cub(u,y): return E.interp(u,y,'cubic')
def Psi_lev(g,w,l):            # scale-separated channel: source level l only (target l-1)
    return E.Psi1(g,w,l,l-1)
def run(w,J,F,tau,second=True):
    lmin=LT; lmax=LT+J+1; levs=range(lmin+1,lmax+1)
    f=E.trig_eval(F,x); ft=E.trig_eval(F,x-tau); nf=np.linalg.norm(f)
    out={}
    # summed channel
    out['sum']=np.linalg.norm(E.Psi1(ft,w,lmax,lmin)-cub(E.Psi1(f,w,lmax,lmin),x-tau))/nf
    # scale-separated
    U={l:Psi_lev(f,w,l) for l in levs}; Ut={l:Psi_lev(ft,w,l) for l in levs}
    e=[np.linalg.norm(Ut[l]-cub(U[l],x-tau)) for l in levs]
    out['sep']=np.sqrt(np.sum(np.square(e)))/nf; out['sep_by_level']=[v/nf for v in e]
    out['E1']=sum(np.linalg.norm(U[l])**2 for l in levs)/nf**2
    import pywt
    c=pywt.wavedec(f,w,mode='periodization',level=lmax); out['Pact']=sum(np.linalg.norm(c[lmax-l+1])**2 for l in levs)/nf**2
    if second and J>=1:
        e2=[];E2=0.0;P2=0.0
        for l1 in levs:
            cu=pywt.wavedec(U[l1],w,mode='periodization',level=lmax)
            P2+=sum(np.linalg.norm(cu[lmax-l2+1])**2 for l2 in levs if l2>l1)
            for l2 in levs:
                if l2<=l1: continue          # second scale coarser (j2<j1)
                V=Psi_lev(U[l1],w,l2); Vt=Psi_lev(Ut[l1],w,l2)
                e2.append(np.linalg.norm(Vt-cub(V,x-tau))**2); E2+=np.linalg.norm(V)**2
        out['sep2']=np.sqrt(sum(e2))/nf; out['E2']=E2/nf**2; out['npairs']=len(e2); out['P2act']=P2/nf**2
    return out
# ---------- dilation averaging ----------
RN=[16384,14336,12288,10240]          # N' multiples of 2^11 -> r = N/N' in {1,8/7,4/3,8/5}
def Psi_dil(F,w,lmax,lmin,shift=None):
    """(1/Q) sum_q D_rq E_b|S_b D_rq^{-1} f|, f=trig(F) possibly pre-warped via shift (x->x-shift)."""
    acc=np.zeros(N); per=[]
    for Np in RN:
        r=N/Np; k=np.arange(Np)
        h=E.trig_eval(F,r*k) if shift is None else E.trig_eval(F,r*k-shift(r*k))   # D_r^{-1} f at integer k (norm. dropped)
        E.N=Np; E.x=np.arange(Np,dtype=float)
        u=E.Psi1(h,w,lmax,lmin)
        E.N=N; E.x=x
        uu=np.append(u,u[0]); from scipy.interpolate import CubicSpline
        cs=CubicSpline(np.arange(Np+1),uu,bc_type='periodic'); v=cs(np.mod(x/r,Np))   # D_r u(x)=u(x/r)
        per.append(v); acc+=v
    return acc/len(RN), per
def run_dil(w,J,F,tau):
    lmin=LT; lmax=LT+J+1
    f=E.trig_eval(F,x); nf=np.linalg.norm(f)
    taufun=lambda y: np.interp(np.mod(y,N),np.append(x,N),np.append(tau,tau[0]))   # tau is a sine: use exact form below
    P=N//8; eps=EPS; tf=lambda y: eps*P/(2*np.pi)*np.sin(2*np.pi*y/P)
    A,per=Psi_dil(F,w,lmax,lmin); At,pert=Psi_dil(F,w,lmax,lmin,shift=tf)
    out={'dil':np.linalg.norm(At-cub(A,x-tau))/nf,
         'dil_nodes':[np.linalg.norm(b-cub(a,x-tau))/nf for a,b in zip(per,pert)]}
    return out
EPS=0.1
if __name__=='__main__':
    what=sys.argv[1]; w=sys.argv[2]; Js=[int(j) for j in sys.argv[3].split(',')]
    OUT=Path(f'ej_ext_{what}_{w}.json'); res=json.load(open(OUT)) if OUT.exists() else []
    tau=tau_smooth(EPS); t0=time.time()
    for J in Js:
        for s in [0]:
            for nm,F in E.make_inputs(w,0,s).items():
                if any(r['J']==J and r['input']==f'{nm}{s}' for r in res): continue
                d=dict(w=w,J=J,input=f'{nm}{s}',eps=EPS)
                d.update(run(w,J,F,tau,second=(what=='layers')) if what=='layers' else run_dil(w,J,F,tau))
                res.append(d); json.dump(res,open(OUT,'w'),indent=0)
                print(what,w,J,nm,{k:(round(v,5) if isinstance(v,float) else v) for k,v in d.items() if k not in('w','J','input','eps','sep_by_level','dil_nodes')},f'{time.time()-t0:.0f}s',flush=True)
    print('DONE')
