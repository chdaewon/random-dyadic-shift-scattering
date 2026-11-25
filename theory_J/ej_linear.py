"""Linear defect behind the bound:  Q_b f(x) = S_b L_tau f(x) - S_b T_{tau(x)} f(x)
and its Jacobian-free part G_b f = Q_b f - S_b[tau' L_tau f].  Report (E_b ||.||^2)^{1/2}/||f||.
S_b T_{tau(x)} f(x) = S_{b-n}(T_theta f)(x-n), tau(x)=n+theta; theta interpolated linearly on K bins."""
import numpy as np, json, time
from pathlib import Path
import ej_check as E
E.LT=4; LT=4; N=E.N; x=E.x; K=8
OUT=Path(__file__).with_name('ej_linear.json')
ar=np.arange(N)
def S_rows(g_rows_shifted_by,b,w,lmax,lmin):
    pass
def S_offsets(g,bs,w,lmax,lmin):
    """returns array [len(bs),N]: S_b g for lattice offsets b (integer)."""
    G=g[(ar[None,:]+bs[:,None])%N]
    Y=E.shift_batch(G,w,lmax,lmin)
    return np.take_along_axis(Y,(ar[None,:]-bs[:,None])%N,axis=1)
def linear_defect(F,w,lmax,lmin,tau,taup):
    P=2**lmax; f=E.trig_eval(F,x); nf=np.linalg.norm(f)
    Lf=E.trig_eval(F,x-tau); jac=taup*Lf
    n=np.floor(tau).astype(int); th=tau-n; mi=np.minimum((th*K).astype(int),K-1); wt=th*K-mi
    Tth=[E.trig_eval(F,x-m/K) for m in range(K+1)]    # T_theta f, theta=m/K
    nmin,nmax=n.min(),n.max(); q2=0.0; g2=0.0; chunk=128
    for b0 in range(0,P,chunk):
        bs=np.arange(b0,min(P,b0+chunk))
        B=S_offsets(Lf,bs,w,lmax,lmin); Jc=S_offsets(jac,bs,w,lmax,lmin)
        ext=np.arange(b0-nmax,bs[-1]-nmin+1)                 # needed b' = b-n
        C=[S_offsets(Tth[m],ext%P,w,lmax,lmin) for m in range(K+1)]
        rows=(bs[:,None]-n[None,:])-ext[0]                   # index into ext
        cols=(ar[None,:]-n[None,:])%N
        lo=np.stack(C)                                       # [K+1, len(ext), N]
        v0=lo[mi[None,:].repeat(len(bs),0),rows,cols]; v1=lo[(mi+1)[None,:].repeat(len(bs),0),rows,cols]
        ST=(1-wt)[None,:]*v0+wt[None,:]*v1
        Q=B-ST; Gm=Q-Jc
        q2+=np.sum(Q**2); g2+=np.sum(Gm**2)
    return np.sqrt(q2/P)/nf, np.sqrt(g2/P)/nf
if __name__=='__main__':
    res=[]; t0=time.time(); eps=0.1; Pw=N//8
    tau=eps*Pw/(2*np.pi)*np.sin(2*np.pi*x/Pw); taup=eps*np.cos(2*np.pi*x/Pw)
    for w in ['haar','sym4']:
        for J in range(0,7):
            lmin=LT; lmax=LT+J+1
            for nm,F in [('pink0',E.make_inputs(w,0,0)['pink']),('balanced',E.balanced(w,lmax,lmin))]:
                q,g=linear_defect(F,w,lmax,lmin,tau,taup)
                res.append(dict(w=w,J=J,input=nm,Q=q,G=g)); json.dump(res,open(OUT,'w'),indent=0)
                print(w,J,nm,round(q,5),round(g,5),f'{time.time()-t0:.0f}s',flush=True)
    print('DONE')
