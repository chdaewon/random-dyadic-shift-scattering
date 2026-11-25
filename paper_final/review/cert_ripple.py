"""Certified (interval-arithmetic) range of Im m_T over one octave for Haar, and certified ripples of the
Q-point dilation quadrature M_Q(xi)=(1/Q) sum_q m_T(2^{(q+1/2)/Q} xi).
f(s)=Im m_T(2^s)=sum_j phi(2^{s-j}),  phi(t)=8 sin^3(pi t/2) sin^2(pi t/4)/(pi^2 t^2)  (Re m_T = 0).
Enclosure on I=[s0,s1]: f(I) in f(mid) + f'(I)*[-h/2,h/2] (mean value form), all in mpmath.iv;
terms with t outside [2^-20,2^20] are bounded analytically:  |phi|<=pi^3 t^3/16, |t phi'|<=7 pi^3 t^3/16 (small t);
|phi|<=8/(pi^2 t^2), |t phi'|<=8(2pi+2/t)/(pi^2 t) (large t).  Branch and bound for max and min."""
from mpmath import iv, mp
import json, sys
iv.dps=30
PI=iv.pi; a=PI/2; b=PI/4; LN2=iv.log(2)
def phi(t):  return 8*iv.sin(a*t)**3*iv.sin(b*t)**2/(PI**2*t**2)
def tdphi(t):
    sa,ca,sb,cb=iv.sin(a*t),iv.cos(a*t),iv.sin(b*t),iv.cos(b*t)
    d=(3*a*sa**2*ca*sb**2+2*b*sa**3*sb*cb)/t**2-2*sa**3*sb**2/t**3
    return 8*t*d/PI**2
JLO,JHI=-21,21                       # 2^{s-j} for s in [0,1+1], j in [JLO,JHI] covers [2^-21,2^23]
def tail(deriv):
    # small t: j>JHI, t<=2^{2-j}; large t: j<JLO, t>=2^{-j}
    small=sum((7 if deriv else 1)*PI**3*(iv.mpf(2)**(2-j))**3/16 for j in range(JHI+1,JHI+40))*2
    large=sum((8*(2*PI+2)/(PI**2*iv.mpf(2)**(-j))) if deriv else 8/(PI**2*iv.mpf(2)**(-2*j)) for j in range(JLO-40,JLO))*2
    T=small+large; return iv.mpf([-T.b,T.b])
TAIL0,TAIL1=tail(False),tail(True)
def f(S):      return sum(phi(iv.mpf(2)**(S-j)) for j in range(JLO,JHI+1))+TAIL0
def df(S):     return LN2*(sum(tdphi(iv.mpf(2)**(S-j)) for j in range(JLO,JHI+1))+TAIL1)
def fQ(S,Q):   return sum(f(S+iv.mpf(q)/Q+iv.mpf(1)/(2*Q)) for q in range(Q))/Q
def dfQ(S,Q):  return sum(df(S+iv.mpf(q)/Q+iv.mpf(1)/(2*Q)) for q in range(Q))/Q
def encl(lo,hi,Q):
    m=(lo+hi)/2; h=(hi-lo)/2; I=iv.mpf([lo,hi])
    v=fQ(iv.mpf(m),Q); d=dfQ(I,Q)
    return v, v+d*iv.mpf([-h,h])
def extremum(Q,sign,tol=2e-8,n0=64):
    period=1.0/Q
    boxes=[(k*period/n0,(k+1)*period/n0) for k in range(n0)]
    best=None; it=0
    while True:
        it+=1; ev=[]
        for lo,hi in boxes:
            v,E=encl(lo,hi,Q)
            pv=v.a if sign>0 else -v.b          # certified lower bound of sign*f at a point
            up=E.b if sign>0 else -E.a          # certified upper bound of sign*f on box
            ev.append((lo,hi,pv,up)); best=pv if best is None or pv>best else best
        keep=[(lo,hi,up) for lo,hi,pv,up in ev if up>=best]
        gap=max(up for _,_,up in keep)-best
        if gap<tol or it>40:
            return (best, max(up for _,_,up in keep), it, len(keep))
        boxes=[]
        for lo,hi,up in keep:
            m=(lo+hi)/2; boxes+= [(lo,m),(m,hi)]
out={}
kap=PI/(8*LN2)
for Q in [1,2,4,5]:
    mx=extremum(Q,+1); mn=extremum(Q,-1)
    F=lambda x: float(x.mid) if hasattr(x,'mid') else float(x)   # 30-digit endpoints -> float (rel. err 1e-16)
    fmax=(F(mx[0]),F(mx[1])); fmin=(F(-mn[1]),F(-mn[0])); ka,kb=F(kap.a),F(kap.b)
    rip_lo=max(fmax[0]-kb, ka-fmin[1])/kb; rip_hi=max(fmax[1]-ka, kb-fmin[0])/ka
    out[Q]=dict(max=list(fmax),min=list(fmin),ripple=[rip_lo,rip_hi],iters=[mx[2],mn[2]])
    print(Q,out[Q],flush=True)
pts={str(x):[float(v.a),float(v.b)] for x,v in [(1.0,f(iv.mpf(0))),(1.5,f(iv.log(iv.mpf(1.5))/LN2))]}
print('Im m_T(1), Im m_T(3/2):',pts)
json.dump(dict(Q=out,points=pts,kappa=[float(kap.a),float(kap.b)]),open('cert_ripple.json','w'),indent=1)
