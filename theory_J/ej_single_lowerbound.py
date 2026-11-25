"""Single-lattice lower bound (A5).  For tau(x)=eps*x on one lattice,
   G~_beta^* 1 (v) = - sum_{j=0}^J Phi_eps(2^j (v-beta)),  Phi_eps 1-periodic, mean zero,
   Fourier coefficients c_n = lam * psihat(lam n) * Thetahat(-eps lam n),  lam = 1/(1-eps).
   ||S_J||_{L2(0,1)}^2 = sum_{o odd} sum_{K>=0} |sum_{j=0}^{min(J,K)} c_{2^{K-j} o}|^2  (exact),
   and ||S_J||^2/(J+1) -> sigma^2 = sum_{o odd} |B_o|^2,  B_o = sum_{i>=0} c_{2^i o}."""
import numpy as np, json, sys
from ej_affine_symbol import psihat_factory
name=sys.argv[1] if len(sys.argv)>1 else 'sym4'
ph,L,m=psihat_factory(name)
def Thetahat(eta):
    eta=np.asarray(eta,float)
    return 0.5*ph(eta/2)*np.exp(-1j*np.pi*m*eta)*(1-np.exp(-1j*np.pi*eta))
def coeffs(eps,ns):
    lam=1/(1-eps); return lam*ph(lam*ns)*Thetahat(-eps*lam*ns)
res=[]
for eps in [0.05,0.1,0.2]:
    Imax=22; omax=4097
    odds=np.arange(1,omax+1,2)
    # c[o_index, i] = c_{2^i o}, positive o only; negative o give complex conjugates
    C=np.zeros((len(odds),Imax),complex)
    for i in range(Imax):
        n=odds*2.0**i; mask=n<2**24
        C[mask,i]=coeffs(eps,n[mask])
    B=C.sum(1); sigma2=2*np.sum(np.abs(B)**2)
    a0=2*np.sum(np.abs(C)**2)          # ||Phi||_2^2
    rows=[]
    for J in range(0,11):
        tot=0.0
        for K in range(Imax+J):
            # sum_{j=0}^{min(J,K)} c_{2^{K-j} o}  = sum over i=K-j in [K-min(J,K), K]
            lo=K-min(J,K); hi=min(K,Imax-1)
            if lo>hi: continue
            tot+=2*np.sum(np.abs(C[:,lo:hi+1].sum(1))**2)
        rows.append(dict(J=J,rms_over_sqrtJ1=float(np.sqrt(tot/(J+1)))))
    r=dict(gen=name,eps=eps,B1=[float(B[0].real),float(B[0].imag)],absB1=float(abs(B[0])),
           sigma=float(np.sqrt(sigma2)),normPhi=float(np.sqrt(a0)),
           sigma_over_eps14=float(np.sqrt(sigma2)/eps**1.4),rows=rows)
    res.append(r)
    print(name,eps,'|B_1|=%.3e'%abs(B[0]),'sigma=%.3e'%np.sqrt(sigma2),'||Phi||=%.3e'%np.sqrt(a0),
          'sigma/eps^1.4=%.4f'%(np.sqrt(sigma2)/eps**1.4),
          'rms/sqrt(J+1)/eps^1.4:',[round(x['rms_over_sqrtJ1']/eps**1.4,4) for x in rows],flush=True)
json.dump(res,open(f'ej_single_lowerbound_{name}.json','w'),indent=1)
