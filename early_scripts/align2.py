import numpy as np
from mult import psi_samples, ft
u=np.linspace(-10,6,1600); t=2.0**u
def kap(name,m,tt=t,uu=u):
    x,p,dx=psi_samples(name,10); ph=ft(x,p,dx,tt); p2=ft(x,p,dx,tt/2)
    Th=0.5*p2*np.exp(-1j*np.pi*m*tt)*(1-np.exp(-1j*np.pi*tt)); return np.conj(ph)*Th
for m in range(6,12):
    print('coif3 m',m, np.round(np.trapezoid(kap('coif3',m),u),3))
# translation-only ripple for aligned db4 (m=4) and sym4 (m=3)
for name,m in [('haar',0),('db4',4),('sym4',3)]:
    s=np.linspace(0,1,21)[:-1]; js=np.arange(-12,9)
    MT=np.array([np.sum(kap(name,m,2.0**(ss+js),None)) for ss in s])
    print(name,'aligned T-only Im range',np.round(MT.imag.min(),3),np.round(MT.imag.max(),3),' Re range',np.round(MT.real.min(),3),np.round(MT.real.max(),3))
# integrand for chart
g=np.linspace(-2,2,33); tt=2.0**g
for name,m in [('haar',0),('db4',0),('db4',4)]:
    print(name,m,np.round(kap(name,m,tt,None).imag,3).tolist())
print(np.round(g,3).tolist())
