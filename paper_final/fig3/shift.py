import numpy as np, pywt
SQ=np.sqrt(2)
class Shift:
    """Discrete periodic first-order shift S^J with wavelet w (copied from DSSN_deliverables/3_scripts/shift.py)."""
    def __init__(s,w,N,lmax,lmin,m=0):
        s.w=pywt.Wavelet(w); s.N=N; s.lmax=lmax; s.lmin=lmin; s.m=m
    def _dec(s,f): return pywt.wavedec(f,s.w,mode='periodization',level=s.lmax)
    def _rec(s,c): return pywt.waverec(c,s.w,mode='periodization')
    def idx(s,l): return s.lmax-l+1
    def apply(s,f):
        c=s._dec(f); o=[np.zeros_like(x) for x in c]
        for l in range(s.lmin+1,s.lmax+1):
            d=c[s.idx(l)]; t=o[s.idx(l-1)]
            t[:]+=np.roll(np.ravel(np.column_stack([d,-d]))/SQ, s.m)
        return s._rec(o)
    def at(s,f,b): return np.roll(s.apply(np.roll(f,-b)),b)
