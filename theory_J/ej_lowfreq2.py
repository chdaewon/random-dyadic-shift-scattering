"""Scale-consistent low-frequency test: input period = 4 x coarsest cell, warp period = 2 x coarsest cell,
J grows by adding FINER relative scales.  Lacunary mechanism predicts growth ~ eps^{1+delta} sqrt(J+1)."""
import numpy as np, json, sys, time
import ej_worst as W
N=W.N; x=W.x; res=[]; t0=time.time(); w=sys.argv[1]; Jmax=int(sys.argv[2])
for eps in [0.1,0.3]:
    for J in range(Jmax+1):
        lmax=W.LT+J+1; Pw=2**(lmax+1); kk=N//2**(lmax+2)
        tau=eps*Pw/(2*np.pi)*np.sin(2*np.pi*x/Pw)
        A=W.Op(w,J,tau); f=np.cos(2*np.pi*kk*x/N)
        r=np.linalg.norm(A.fwd(f))/np.sqrt(A.P)/np.linalg.norm(f)
        res.append(dict(w=w,eps=eps,J=J,ratio=r)); json.dump(res,open(f'ej_lowfreq2_{w}.json','w'),indent=0)
        print(w,eps,J,'r/eps',round(r/eps,5),'r/eps^1.4',round(r/eps**1.4,5),'r/eps^2',round(r/eps**2,5),f'{time.time()-t0:.0f}s',flush=True)
