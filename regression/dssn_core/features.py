
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from .shift import Shift, cycle_average, monte_carlo_average

def periodic_cubic(f, coords):
    N=len(f); x=np.arange(N); cs=CubicSpline(np.r_[x,N],np.r_[f,f[0]],bc_type='periodic')
    return cs(np.mod(coords,N))

def trig_resample(f, coords, chunk=512):
    """Band-limited periodic interpolation by direct Fourier evaluation."""
    f=np.asarray(f); N=len(f); coords=np.mod(np.asarray(coords,float),N)
    F=np.fft.fft(f)/N; k=np.fft.fftfreq(N)*N
    out=np.empty(coords.shape,dtype=complex)
    for a in range(0,coords.size,chunk):
        c=coords.ravel()[a:a+chunk]
        out.ravel()[a:a+chunk]=np.exp(2j*np.pi*np.outer(c,k)/N).dot(F)
    return out.real.reshape(coords.shape)

def resample_periodic(f, scale=1.0, method='fft'):
    x=np.arange(len(f)); coords=x/float(scale)
    if method in ('fft','bandlimited'): return trig_resample(f,coords)
    if method in ('cubic','spline'): return periodic_cubic(f,coords)
    raise ValueError('method must be fft/bandlimited or cubic/spline')

def warp_periodic(f,tau,method='cubic'):
    x=np.arange(len(f)); coords=x-np.asarray(tau)
    return trig_resample(f,coords) if method in ('fft','bandlimited') else periodic_cubic(f,coords)

def gaussian_pool(f,sigma): return gaussian_filter1d(f, sigma=max(0.6,float(sigma)), mode='wrap')

def dssn_feature(f,wavelet='haar',m=0,lmax=6,lmin=0,offsets='cycle',M=64,seed=0,pool_sigma=8,include_zero=True,remove_identity=0.0):
    S=Shift(wavelet,len(f),lmax,lmin,m,remove_identity=remove_identity)
    if offsets=='cycle': ch1=cycle_average(S,f,modulus=True)
    else: ch1=monte_carlo_average(S,f,M=M,seed=seed,modulus=True)
    out=[gaussian_pool(ch1,pool_sigma)]
    if include_zero: out.insert(0,gaussian_pool(f,pool_sigma))
    return np.concatenate(out)

def dilation_quadrature_feature(f,wavelet='haar',m=0,lmax=6,lmin=0,Q=4,resample_method='fft',**kwargs):
    acc=None
    for q in range(int(Q)):
        r=2**((q+0.5)/int(Q))
        fr=resample_periodic(f,r,method=resample_method)
        y=dssn_feature(fr,wavelet=wavelet,m=m,lmax=lmax,lmin=lmin,offsets='cycle',**kwargs)
        acc=y if acc is None else acc+y
    return acc/int(Q)
