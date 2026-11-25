import numpy as np, pywt
SQ=np.sqrt(2)
class Shift:
    """Discrete periodic first-order shift S^J with wavelet w.
    source levels l = lmin+1 .. lmax (coarse=lmax <-> doc scale j=0), target level l-1."""
    def __init__(s,w,N,lmax,lmin,m=0):
        s.w=pywt.Wavelet(w); s.N=N; s.lmax=lmax; s.lmin=lmin; s.m=m
    def _dec(s,f): return pywt.wavedec(f,s.w,mode='periodization',level=s.lmax)
    def _rec(s,c): return pywt.waverec(c,s.w,mode='periodization')
    # coeffs list: c[0]=cA_lmax, c[i]=cD_{lmax-i+1}; level l -> index lmax-l+1
    def idx(s,l): return s.lmax-l+1
    def apply(s,f):
        c=s._dec(f); o=[np.zeros_like(x) for x in c]
        for l in range(s.lmin+1,s.lmax+1):
            d=c[s.idx(l)]; t=o[s.idx(l-1)]; n=len(t)
            t[:]+=np.roll(np.ravel(np.column_stack([d,-d]))/SQ, s.m)
        return s._rec(o)
    def adj(s,g):
        c=s._dec(g); o=[np.zeros_like(x) for x in c]
        for l in range(s.lmin+1,s.lmax+1):
            t=np.roll(c[s.idx(l-1)],-s.m); o[s.idx(l)]+= (t[0::2]-t[1::2])/SQ
        return s._rec(o)
    def at(s,f,b):   # shifted lattice by b samples: T_b S T_{-b}
        return np.roll(s.apply(np.roll(f,-b)),b)
    def at_adj(s,g,b): return np.roll(s.adj(np.roll(g,-b)),b)
