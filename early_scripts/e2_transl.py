import numpy as np
from shift import Shift
N=2**14; lmax=10; lmin=4     # source levels 5..10 (J=5), finest source cell 32 samples
rng=np.random.default_rng(0)
def opnorm(A,At,it=40):
    x=rng.standard_normal(N); x/=np.linalg.norm(x)
    for _ in range(it):
        y=At(A(x)); nrm=np.linalg.norm(y); x=y/nrm
    return np.sqrt(nrm)
res={}
for w in ['haar','db2','db4']:
    S=Shift(w,N,lmax,lmin); out=[]
    for a in [1,2,4,8,16]:
        A=lambda f: S.apply(f)-S.at(f,a)
        At=lambda g: S.adj(g)-S.at_adj(g,a)
        out.append(opnorm(A,At))
    res[w]=out; print(w,np.round(out,4))
a=np.array([1,2,4,8,16])/32
for w in res:
    sl=np.polyfit(np.log(a[:3]),np.log(res[w][:3]),1)[0]; print(w,'slope(small a)=',round(sl,3))
np.save('e2.npy',np.array([a]+[res[w] for w in ['haar','db2','db4']]))
