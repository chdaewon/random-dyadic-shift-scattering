"""Independent check of Prop 3.1 for Haar: closed form vs definition at +/- xi, and sign of the H-component
on a discrete implementation (H sin = -cos with symbol -i sgn)."""
import numpy as np
from scipy.integrate import quad
from shift import Shift
hhat=lambda x: np.exp(-1j*np.pi*x)*2j*np.sin(np.pi*x/2)**2/(np.pi*x)
That=lambda x: 0.5*hhat(x/2)*(1-np.exp(-1j*np.pi*x))          # m=0
m0_def=lambda x: np.conj(hhat(x))*That(x)
m0_cf=lambda x: 8j*np.sin(np.pi*x/2)**3*np.sin(np.pi*x/4)**2/(np.pi**2*x**2)
for x in [0.3,1.0,2.7,-0.3,-1.0,-2.7]:
    print('xi=%5.2f def=%s cf=%s'%(x,np.round(m0_def(x),10),np.round(m0_cf(x),10)))
k_pos=quad(lambda t: m0_cf(t).imag/t,0,200,limit=4000)[0]+quad(lambda t: m0_cf(t).imag/t,200,np.inf,limit=200)[0]
k_neg=quad(lambda t: m0_cf(-t).imag/t,0,200,limit=4000)[0]+quad(lambda t: m0_cf(-t).imag/t,200,np.inf,limit=200)[0]
print('Im kappa: xi>0 %.9f  xi<0 %.9f  pi/(8ln2)=%.9f'%(k_pos/np.log(2),k_neg/np.log(2),np.pi/(8*np.log(2))))
# discrete: E_beta S^J sin(2 pi xi x) vs Im m_J(xi) cos(2 pi xi x), xi in units of coarsest cell
N=4096; lmax=6; lmin=1; P=2**lmax; S=Shift('haar',N,lmax,lmin); x=np.arange(N)
for k in [48,96,80]:
    f=np.sin(2*np.pi*k*x/N); acc=np.zeros(N)
    for b in range(P): acc+=S.at(f,b)
    acc/=P; xi=k*P/N   # cycles per coarsest cell
    mJ=sum(m0_cf(xi*2.0**-j) for j in range(0,lmax-lmin))   # scales j=0..J, J=lmax-lmin-1
    c=np.cos(2*np.pi*k*x/N); s_=np.sin(2*np.pi*k*x/N)
    print('xi=%.2f  <ES sin,cos>/|cos|^2=%.5f  <ES sin,sin>/|sin|^2=%.5f   Im m_J=%.5f Re m_J=%.1e'%(xi,acc@c/(c@c),acc@s_/(s_@s_),mJ.imag,mJ.real))
