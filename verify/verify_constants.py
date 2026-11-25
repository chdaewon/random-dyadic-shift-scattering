"""Independent check of DSSN multiplier constants.
psi-hat is computed exactly from the filter (infinite product), so no wavefun sampling error."""
import numpy as np, pywt, json
from pathlib import Path
OUT=Path(__file__).with_name('verify_constants.json')
from scipy.integrate import quad
LN2=np.log(2)

def haar_m0(t):
    t=np.asarray(t,float)
    return 8j*np.sin(np.pi*t/2)**3*np.sin(np.pi*t/4)**2/(np.pi**2*t**2)

def make_psihat(name,K=40):
    w=pywt.Wavelet(name); lo=np.array(w.rec_lo); hi=np.array(w.rec_hi); n=np.arange(len(lo))
    def H(x): return (lo[None,:]*np.exp(-2j*np.pi*np.outer(x,n))).sum(1)/np.sqrt(2)
    def G(x): return (hi[None,:]*np.exp(-2j*np.pi*np.outer(x,n))).sum(1)/np.sqrt(2)
    def phihat(x):
        x=np.asarray(x,float).ravel(); out=np.ones(x.size,complex)
        for k in range(1,K+1): out*=H(x/2**k)
        return out
    def psihat(x):
        x=np.asarray(x,float).ravel(); return G(x/2)*phihat(x/2)
    return psihat,len(lo)-1

def m0_factory(name,m):
    if name=='haar' and m==0: return haar_m0
    ph,L=make_psihat(name)
    def m0(t):
        t=np.asarray(t,float); sh=t.shape; t=t.ravel()
        th=0.5*ph(t/2)*np.exp(-1j*np.pi*m*t)*(1-np.exp(-1j*np.pi*t))
        return (np.conj(ph(t))*th).reshape(sh)
    return m0

def kappa(m0,umin=-25,umax=14,n=200001):
    u=np.linspace(umin,umax,n); t=2.0**u      # dt/t = ln2 du
    return np.trapezoid(m0(t),u)               # (1/ln2)*ln2

def mT(m0,s,jr=45):
    j=np.arange(-jr,jr+1); return m0(2.0**(s[:,None]+j[None,:])).sum(1)

def ripples(m0,k,Qs=(1,2,4,5),n=2000):
    s=np.linspace(0,1,n,endpoint=False); out={}
    for Q in Qs:
        MQ=np.mean([mT(m0,(s+(q+.5)/Q)%1) for q in range(Q)],axis=0)
        out[Q]=float(np.max(np.abs(MQ-k))/abs(k.imag))
    return out

res={}
# Haar closed form
I=quad(lambda y: np.sin(y)**3*(1-np.cos(y))/y**3,0,np.inf,limit=2000)[0]
res['haar_integral_sin3(1-cos)/y^3']=I; res['pi/8']=np.pi/8
m0=haar_m0; k=kappa(m0); s=np.linspace(0,1,4000,endpoint=False); MT=mT(m0,s)
res['haar']=dict(kappa=[k.real,k.imag],exact=np.pi/(8*LN2),c0=-1/k.imag,
   ImmT_range=[float(MT.imag.min()),float(MT.imag.max())],ripple=ripples(m0,k))
# check psihat against pywt wavefun for sym4 (orientation)
ph,L=make_psihat('sym4'); phi,psi,x=pywt.Wavelet('sym4').wavefun(level=14)
dx=x[1]-x[0]; xi=np.array([0.3,0.7,1.1,1.9])
num=np.array([np.sum(psi*np.exp(-2j*np.pi*x*z))*dx for z in xi])
res['sym4_psihat_check_maxdiff']=float(np.max(np.abs(num-ph(xi))))
for name,ms in [('db2',range(0,3)),('db4',range(0,7)),('sym4',range(0,7)),('coif3',range(0,17)),('sym6',range(0,11))]:
    rows={}
    for m in ms:
        kk=kappa(m0_factory(name,m),n=60001); rows[m]=[kk.real,kk.imag]
    mstar=max(rows,key=lambda m: abs(rows[m][1]))
    m0s=m0_factory(name,mstar); ks=kappa(m0s)
    rows_naive=rows[0]
    res[name]=dict(L=pywt.Wavelet(name).dec_len-1,m_star=mstar,kappa=[ks.real,ks.imag],
        abs_c0=1/abs(ks.imag),re_over_im=abs(ks.real)/abs(ks.imag),
        naive_m0_kappa=rows_naive,naive_abs_c0=1/abs(rows_naive[1]),
        ripple=ripples(m0s,ks),all_m=rows)
    print(name,'done',mstar,ks)
json.dump(res,open(OUT,'w'),indent=1,default=float)
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk!='all_m'}) for k,v in res.items()},indent=1,default=float))
