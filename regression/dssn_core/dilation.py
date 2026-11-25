'''Signal-level dilation quadrature A_Q (A4).  dssn_core 0.2.0.

A_Q f = (1/Q) sum_q R_{r_q}^{-1} E_beta S_beta R_{r_q} f,  r_q = 2^((q+1/2)/Q).
R_r is realised as resampling of the periodic signal onto a circle of M_q = round(r_q N / 2^lmax) 2^lmax samples
(one full period of the dilated function, so there is no seam); R_r^{-1} resamples back to N samples.
The effective ratio r_eff = M_q/N differs from r_q by at most 2^lmax/(2 r_q N) (relative).
'''
import numpy as np
from scipy.signal import resample as _sp_resample
from .features import periodic_cubic
from .shift import Shift, cycle_average

DEFAULT_METHOD = 'fft'


def resample_length(x, M, method=DEFAULT_METHOD):
    '''Periodic resampling of x (length N) to M samples over the same period.'''
    x = np.asarray(x); N = len(x); M = int(M)
    if M == N:
        return x.copy()
    if method in ('fft', 'bandlimited'):
        return _sp_resample(x, M)
    if method in ('spline3', 'cubic', 'spline'):
        return periodic_cubic(x, np.arange(M) * (N / M))
    raise ValueError('method must be fft or spline3')


def node_lengths(N, Q, lmax):
    r = 2.0 ** ((np.arange(int(Q)) + 0.5) / int(Q)); P = 2 ** int(lmax)
    M = np.maximum(np.rint(r * N / P), 1).astype(int) * P
    return r, M


def continuous_to_discrete_offset(name, m):
    from .multipliers import discrete_to_continuous_offset
    info = discrete_to_continuous_offset(name, 0, return_info=True)
    return int(m) - int(info['shift'])


def dilation_quadrature_shift(f, wavelet='sym4', m=3, Q=4, method=DEFAULT_METHOD, lmax=10, lmin=0,
                              offsets=None, remove_identity=0.0, m_is_discrete=False, return_info=False):
    '''Linear operator A_Q (no modulus, no pooling).  m is the CONTINUOUS child offset unless m_is_discrete.
    offsets=None -> exhaustive average over all 2^lmax offsets (E_beta).  Returns A_Q f (same length as f).'''
    f = np.asarray(f); N = len(f)
    md = int(m) if m_is_discrete else continuous_to_discrete_offset(wavelet, m)
    r, M = node_lengths(N, Q, lmax)
    parts = [f.real, f.imag] if np.iscomplexobj(f) else [f]
    outs = []
    for x in parts:
        acc = np.zeros(N)
        for Mq in M:
            g = resample_length(x, int(Mq), method)
            S = Shift(wavelet, int(Mq), lmax, lmin, md, remove_identity=remove_identity)
            y = cycle_average(S, g, offsets=offsets)
            acc += resample_length(y, N, method)
        outs.append(acc / int(Q))
    out = outs[0] if len(outs) == 1 else outs[0] + 1j * outs[1]
    if return_info:
        return out, dict(r_nominal=r, r_eff=M / N, M=M, m_discrete=md)
    return out
