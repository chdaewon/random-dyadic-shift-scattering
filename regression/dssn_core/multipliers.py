"""Symbols (Fourier multipliers) of the averaged dyadic shift.  dssn_core 0.2.0.

Changes w.r.t. 0.1.x (see ../README.md):
  A1  Haar / db1 psi is sampled at cell midpoints x=(k+1/2)dx (was left end points).
  A2  symbol grid default n=320, grid size is an argument, n % Q != 0 -> ValueError.
  A5  discrete_to_continuous_offset searches the integer offset instead of hard-coding +3.
  A6  np.trapz -> trapezoid (compat).
"""
import numpy as np, pywt
from functools import lru_cache
from .compat import trapezoid
from .shift import Shift, cycle_average

DEFAULT_GRID_N = 320          # = lcm(1,2,4,5) * 16 ; divisible by Q in {1,2,4,5}
DEFAULT_J_MIN, DEFAULT_J_MAX = -12, 8


def _check_grid(n, Qs):
    n = int(n)
    Qs = [Qs] if np.isscalar(Qs) else list(Qs)
    for Q in Qs:
        if n % int(Q) != 0:
            raise ValueError(f'grid size n={n} is not divisible by Q={int(Q)} (n % Q != 0)')
    return n


def psi_samples(name, level=11, haar_sampling='midpoint'):
    """Samples of the unit-L2 wavelet.  Haar/db1: midpoint rule x=(k+1/2)dx (0.2.0);
    haar_sampling='left' reproduces the 0.1.x left-end-point sampling."""
    if name in ('haar', 'db1'):
        n = 2 ** int(level); dx = 1.0 / n
        if haar_sampling == 'midpoint':
            x = (np.arange(n) + 0.5) * dx
        elif haar_sampling in ('left', 'legacy'):
            x = np.arange(n) * dx
        else:
            raise ValueError('haar_sampling must be midpoint or left')
        p = np.where(x < 0.5, 1.0, -1.0)
        return x, p, dx
    w = pywt.Wavelet(name); phi, psi, x = w.wavefun(level=level); dx = x[1] - x[0]
    return x, psi / np.sqrt(np.sum(psi ** 2) * dx), dx


def ft(x, p, dx, xi, chunk=512):
    xi = np.atleast_1d(np.asarray(xi, float)); out = np.empty(xi.shape, dtype=complex)
    for a in range(0, len(xi), chunk):
        z = xi[a:a + chunk]
        out[a:a + chunk] = (p[:, None] * np.exp(-2j * np.pi * x[:, None] * z[None, :])).sum(axis=0) * dx
    return out


def m0(name, m, xi, level=11, haar_sampling='midpoint'):
    x, p, dx = psi_samples(name, level, haar_sampling); xi = np.asarray(xi, float)
    ph = ft(x, p, dx, xi); ph2 = ft(x, p, dx, xi / 2)
    return np.conj(ph) * 0.5 * ph2 * np.exp(-1j * np.pi * int(m) * xi) * (1 - np.exp(-1j * np.pi * xi))


def kappa(name='haar', m=0, level=11, u_min=-12, u_max=7, n=2200, haar_sampling='midpoint'):
    u = np.linspace(u_min, u_max, n)
    return trapezoid(m0(name, m, 2.0 ** u, level, haar_sampling), u)


def translation_symbol(name='haar', m=0, s_grid=None, level=11, n=DEFAULT_GRID_N,
                       j_min=DEFAULT_J_MIN, j_max=DEFAULT_J_MAX, haar_sampling='midpoint'):
    """Translation-only symbol m_T(2^s), s in [0,1) (log2 phase), summed over scales j in [j_min, j_max]."""
    if s_grid is None:
        s_grid = np.linspace(0, 1, int(n), endpoint=False)
    s = np.asarray(s_grid, float); js = np.arange(j_min, j_max + 1)
    xi = 2.0 ** (s[:, None] + js[None, :])
    v = m0(name, m, xi.ravel(), level, haar_sampling).reshape(xi.shape).sum(axis=1)
    return s, v


def _quad_average(MT, Q):
    n = len(MT); idx = (np.arange(n)[:, None] + (np.arange(Q) * (n // Q))[None, :]) % n
    return MT[idx].mean(axis=1)


def dilation_quadrature_symbol(name='haar', m=0, Q=4, s_grid=None, level=11, n=DEFAULT_GRID_N, **kw):
    """Q-point dilation quadrature of the translation symbol; nodes at s + q/Q (exact index n/Q)."""
    if s_grid is not None:
        n = len(s_grid)
    n = _check_grid(n, int(Q))
    s, MT = translation_symbol(name, m, s_grid=s_grid, level=level, n=n, **kw)
    return s, _quad_average(MT, int(Q))


def q_ripples(name='haar', m=0, Qlist=(1, 2, 4, 5), level=10, n=DEFAULT_GRID_N,
              kappa_kwargs=None, **kw):
    """Relative max deviation max|M_Q - kappa| / |Im kappa| for each Q in Qlist.  Returns (kappa, [ripple_Q])."""
    n = _check_grid(n, Qlist)
    kk = dict(level=level, u_min=-10, u_max=6, n=1600)
    if kappa_kwargs: kk.update(kappa_kwargs)
    hs = kw.get('haar_sampling', 'midpoint'); kk.setdefault('haar_sampling', hs)
    kap = kappa(name, m, **kk)
    s, MT = translation_symbol(name, m, level=level, n=n, **kw)
    rows = [float(np.max(np.abs(_quad_average(MT, int(Q)) - kap)) / max(1e-12, abs(kap.imag))) for Q in Qlist]
    return kap, rows


def best_offset(name, search=None):
    if search is None:
        L = {'haar': 1, 'db2': 3, 'db4': 7, 'sym4': 7, 'coif3': 17}.get(name, pywt.Wavelet(name).dec_len - 1)
        search = range(max(1, L + 2))
    vals = [(m, kappa(name, m, level=10, u_min=-10, u_max=6, n=1400)) for m in search]
    return max(vals, key=lambda z: (abs(z[1].imag), -abs(z[1].real)))


# ----------------------------------------------------------------------------------------------
# A5: discrete <-> continuous child-offset correspondence by search
# ----------------------------------------------------------------------------------------------
def discrete_symbol(name='haar', md=0, N=4096, lmax=10, lmin=0):
    """Impulse response h of the offset-averaged (cycle-spun) discrete shift; A e_k = fft(h)[k] e_k."""
    S = Shift(name, N, lmax, lmin, md); d = np.zeros(N); d[0] = 1.0
    return cycle_average(S, d)


def dtft(h, f):
    """Symbol sum_n h[n] exp(-2 pi i f n / N) at (possibly fractional) frequencies f (cycles per N samples)."""
    h = np.asarray(h); N = len(h); n = np.arange(N); f = np.atleast_1d(np.asarray(f, float))
    out = np.empty(f.shape, complex)
    for a in range(0, f.size, 256):
        out.ravel()[a:a + 256] = np.exp(-2j * np.pi * np.outer(f.ravel()[a:a + 256], n) / N) @ h
    return out


def continuous_symbol_at_k(name, m, ks, N, level=10, xi_min=2.0 ** -10, xi_max=64.0, ls=range(-14, 20)):
    """All-scale translation symbol sum_l m0(k 2^l / N) with continuous child offset m (xi restricted to [xi_min, xi_max])."""
    ks = np.asarray(ks, float); xi = np.outer(ks, 2.0 ** np.array(list(ls), float)) / N
    mask = (xi >= xi_min) & (xi <= xi_max); vals = np.zeros(xi.shape, complex)
    vals[mask] = m0(name, m, xi[mask], level)
    return vals.sum(axis=1)


@lru_cache(maxsize=None)
def _d2c(name, md, N, lmax, smin, smax):
    ks = np.arange(24, 101, 4)
    h = discrete_symbol(name, md, N, lmax); H = np.fft.fft(h)[ks]
    errs = {}
    for s in range(smin, smax + 1):
        C = continuous_symbol_at_k(name, md + s, ks, N)
        errs[s] = float(np.max(np.abs(H - C)))
    order = sorted(errs, key=errs.get)
    return order[0], errs[order[0]], order[1], errs[order[1]], float(np.max(np.abs(H)))


def discrete_to_continuous_offset(name, md, N=4096, lmax=10, search=(-8, 8), return_info=False):
    """Continuous child offset m such that the translation-averaged DISCRETE symbol of Shift(name, m_d=md)
    equals the CONTINUOUS all-scale symbol m_T with offset m (integer shift searched in `search`).
    Matching: max_k |sym_discrete(k) - m_T,cont(k)| over a band of integer frequencies."""
    s, e, s2, e2, hmax = _d2c(name, int(md), int(N), int(lmax), int(search[0]), int(search[1]))
    m = int(md) + s
    if return_info:
        return dict(name=name, md=int(md), m=m, shift=s, err=e, second_shift=s2, second_err=e2, sym_max=hmax)
    return m
