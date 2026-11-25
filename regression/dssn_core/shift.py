
import numpy as np, pywt

class Shift:
    """Periodic first-order dyadic wavelet shift using PyWavelets periodization.

    Detail levels are numbered 1..lmax in PyWavelets convention. The target of
    level l is level l-1; since there is no detail level 0 in wavedec, the loop
    starts at max(lmin+1, 2). This makes lmin=0 safe while preserving all valid
    detail-to-detail channels.
    """
    def __init__(self,wavelet='haar',N=1024,lmax=6,lmin=0,m=0,remove_identity=0.0):
        self.w=pywt.Wavelet(wavelet); self.wavelet=wavelet; self.N=int(N)
        self.lmax=int(lmax); self.lmin=int(lmin); self.m=int(m); self.remove_identity=float(remove_identity)
    def _dec(self,f): return pywt.wavedec(np.asarray(f),self.w,mode='periodization',level=self.lmax)
    def _rec(self,c): return pywt.waverec(c,self.w,mode='periodization')[:self.N]
    def idx(self,l): return self.lmax-int(l)+1
    def levels(self): return range(max(self.lmin+1,2), self.lmax+1)
    def apply(self,f):
        f=np.asarray(f); c=self._dec(f); o=[np.zeros_like(x) for x in c]
        for l in self.levels():
            d=c[self.idx(l)]; t=o[self.idx(l-1)]
            t[:]+=np.roll(np.ravel(np.column_stack([d,-d]))/np.sqrt(2), self.m)
        y=self._rec(o)
        return y-self.remove_identity*f if self.remove_identity else y
    def adj(self,g):
        g=np.asarray(g); c=self._dec(g); o=[np.zeros_like(x) for x in c]
        for l in self.levels():
            t=np.roll(c[self.idx(l-1)],-self.m); o[self.idx(l)]+=(t[0::2]-t[1::2])/np.sqrt(2)
        y=self._rec(o)
        return y-self.remove_identity*g if self.remove_identity else y
    def at(self,f,b): return np.roll(self.apply(np.roll(np.asarray(f),-int(b))),int(b))
    def at_adj(self,g,b): return np.roll(self.adj(np.roll(np.asarray(g),-int(b))),int(b))
    def period_offsets(self): return range(2**self.lmax)

def cycle_average(S,f,offsets=None,modulus=False):
    if offsets is None: offsets=S.period_offsets()
    offsets=list(offsets); acc=np.zeros(S.N)
    for b in offsets:
        y=S.at(f,b); acc += np.abs(y) if modulus else y
    return acc/len(offsets)

def monte_carlo_average(S,f,M=1024,seed=0,modulus=False):
    rng=np.random.default_rng(seed); P=2**S.lmax; acc=np.zeros(S.N)
    for b in rng.integers(0,P,size=int(M)):
        y=S.at(f,int(b)); acc += np.abs(y) if modulus else y
    return acc/int(M)
