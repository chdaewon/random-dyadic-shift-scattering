import numpy as np, pywt
from shift import Shift
from e3_deform import fev, Psi, warp  # reuses f
import e3_deform as E
N=E.N; x=E.x; f=E.f; nf=E.nf
# (a) J dependence of averaged covariance error, Haar, eps=0.02, P=N
print("J-dependence (haar, eps=.02):")
Jres=[]
for lmax in [5,6,7,8,9]:
    S=Shift('haar',N,lmax,3); E.Pcell=2**lmax
    tau=0.02*N/(2*np.pi)*np.sin(2*np.pi*x/N); ftau=fev(x-tau)
    r=np.linalg.norm(Psi(S,ftau)-warp(Psi(S,f),tau))/nf; Jres.append((lmax-3-1,r)); print(lmax-4,round(r,5))
np.save('e4a.npy',np.array(Jres))
