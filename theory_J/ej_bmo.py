"""Direct evaluation of h = sum_{j<=J} h_j,  h_j(u) = sum_I int Theta_I(x) psi_I(u + tau(x)) dx   (= -(G~^* 1) o rho^{-1}),
for a fixed lattice beta, continuous model on a periodic domain of D unit cells (scale j=0 cell = 1).
h_j(u) = sum_k sum_q w_q psi( 2^j(u-beta) - k + 2^j tau(x_kq) ),  x_kq = 2^-j (k+s_q) + beta,  w_q = Theta(s_q) ds
(discrete Theta weights corrected to have exactly the continuous vanishing moments).  BMO via L2 mean oscillation."""
import numpy as np, pywt, json, sys, time
from pathlib import Path
D=4; BETA=0.3137
def tables(w):
    if w=='haar':
        L=1; m=0; psi=lambda t: np.where((t>=0)&(t<0.5),1.0,np.where((t>=0.5)&(t<1),-1.0,0.0)); nmom=1
    else:
        W=pywt.Wavelet(w); phi,ps,xx=W.wavefun(level=13); L=W.dec_len-1; m=(L-1)//2 if w!='db4' else 4
        nmom=W.vanishing_moments_psi
        def psi(t,ps=ps,xx=xx): return np.interp(t,xx,ps,left=0.0,right=0.0)
    return psi,L,m,nmom
def theta_weights(psi,L,m,nmom,Ms):
    s=(np.arange(L*Ms)+0.5)/Ms; ds=1.0/Ms
    th=psi(2*s-m)-psi(2*s-m-1); wq=th*ds
    # enforce vanishing moments 0..nmom exactly (Theta has nmom+1)
    V=np.vstack([s**p for p in range(nmom+1)])
    wq=wq-V.T@np.linalg.lstsq(V@V.T,V@wq,rcond=None)[0]
    return s,wq
def make_tau(kind,eps,J):
    if kind=='smooth':
        a=eps*D/(2*np.pi); return (lambda x: a*np.sin(2*np.pi*x/D))
    Ps=[D*2.0**(-p) for p in range(0,J+3)]; K=len(Ps); rng=np.random.default_rng(5); ph=rng.uniform(0,2*np.pi,K)
    return (lambda x: sum(eps/K*P/(2*np.pi)*np.sin(2*np.pi*x/P+f) for P,f in zip(Ps,ph)))
def h_scale(j,tau,psi,s,wq,L,u,chunk=4096):
    out=np.zeros_like(u); sc=2.0**j
    for c0 in range(0,len(u),chunk):
        uu=u[c0:c0+chunk]
        k0=np.floor(sc*(uu+tau(uu+tau(uu))-BETA)).astype(int)
        acc=np.zeros(len(uu))
        for dk in range(-L-2,3):
            k=(k0+dk)[:,None]
            x=(k+s[None,:])/sc+BETA
            t=sc*(uu[:,None]-BETA)-k+sc*tau(x)
            acc+=psi(t)@wq if False else (psi(t)*wq[None,:]).sum(1)
        out[c0:c0+chunk]=acc
    return out
def bmo(g,cell):
    n=len(g); G=np.concatenate([g,g]); c1=np.concatenate([[0],np.cumsum(G)]); c2=np.concatenate([[0],np.cumsum(G*G)])
    best=0.0; ln=cell
    while ln<=n//2:
        i=np.arange(n); m1=(c1[i+ln]-c1[i])/ln; m2=(c2[i+ln]-c2[i])/ln
        best=max(best,float(np.sqrt(np.maximum(m2-m1*m1,0).max()))); ln*=2
    return best
if __name__=='__main__':
    w=sys.argv[1]; Jmax=int(sys.argv[2]); kind=sys.argv[3]; epss=[float(e) for e in sys.argv[4].split(',')]
    Mu=int(sys.argv[5]) if len(sys.argv)>5 else 16; Ms=int(sys.argv[6]) if len(sys.argv)>6 else (512 if w=='haar' else 64)
    psi,L,m,nmom=tables(w); s,wq=theta_weights(psi,L,m,nmom,Ms)
    u=(np.arange(D*2**Jmax*Mu)+0.5)/(2**Jmax*Mu)
    OUT=Path(__file__).with_name(f'ej_bmo_{w}_{kind}.json'); res=[]; t0=time.time()
    for eps in epss:
        tau=make_tau(kind,eps,Jmax); H=np.zeros_like(u); per=[]
        for j in range(Jmax+1):
            hj=h_scale(j,tau,psi,s,wq,L,u); H+=hj
            per.append(dict(j=j,sup=float(np.abs(hj).max()),rms=float(np.sqrt(np.mean(hj**2)))))
            res.append(dict(w=w,kind=kind,eps=eps,J=j,bmo=bmo(H,Mu),sup_hj=per[-1]['sup'],rms_hj=per[-1]['rms']))
            print(w,kind,eps,j,'bmo/eps',round(res[-1]['bmo']/eps,4),'bmo/eps2',round(res[-1]['bmo']/eps**2,4),'sup hj/eps2',round(per[-1]['sup']/eps**2,4),f'{time.time()-t0:.0f}s',flush=True)
            json.dump(res,open(OUT,'w'),indent=0)
    print('DONE')
