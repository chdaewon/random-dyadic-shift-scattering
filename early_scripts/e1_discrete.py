# discrete translation-averaged multiplier m[k] = <S e_k, e_k>/N averaged over all circular shifts
import numpy as np
from shift import Shift
N=2**12; J=8; lmax=J+2; lmin=1
S=Shift('haar',N,lmax,lmin)
n=np.arange(N); ks=np.unique(np.round(2**np.linspace(np.log2(4),np.log2(N/8),60)).astype(int))
m=[]
for k in ks:
    e=np.exp(2j*np.pi*k*n/N)
    Se=S.apply(e.real)+1j*S.apply(e.imag)
    m.append(np.vdot(e,Se)/N)   # <Se,e>/N : translation average eigenvalue
m=np.array(m)
# frequency in cycles per coarsest cell (2^lmax samples): xi = k*2^lmax/N
xi=ks*2**lmax/N
np.save('e1.npy',np.vstack([xi,m.real,m.imag]))
mid=(xi>8)&(xi<2**(J-1))
print("Im m range in mid band:",m.imag[mid].min(),m.imag[mid].max(),"Re max",abs(m.real).max())
print("ratio max/min",m.imag[mid].max()/m.imag[mid].min())
