'''A7 regression tests for dssn_core 0.2.0 (no pytest).  Run:  python run_tests.py
Writes tests_result.csv (test, quantity, expected, tolerance, measured, pass).  Exit code 1 if any non-record row fails.'''
import sys, time, csv, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dssn_core
from dssn_core import multipliers as mu
from dssn_core.shift import Shift, cycle_average, monte_carlo_average
from dssn_core.dilation import dilation_quadrature_shift as dqs

ROWS = []


def add(test, quantity, expected, tol, measured, mode='abs'):
    if mode == 'record':
        ok = 'record'
    elif mode == 'exact':
        ok = bool(measured == expected)
    elif mode == 'rel':
        ok = bool(abs(measured - expected) <= tol * abs(expected))
    else:
        ok = bool(abs(measured - expected) <= tol)
    ROWS.append(dict(test=test, quantity=quantity, expected=expected, tolerance=tol, measured=measured, **{'pass': ok}))
    print('%-34s %-36s exp=%-10s meas=%-12s %s' % (test[:34], quantity[:36], expected, '%.6g' % measured if isinstance(measured, float) else measured, ok), flush=True)


KW10 = dict(level=10, u_min=-10, u_max=6, n=1600)
rng = np.random.default_rng(0)
t0 = time.time()
add('version', '__version__', '0.2.0', 'exact', dssn_core.__version__, 'exact')

for nm in ('haar', 'sym4'):
    S = Shift(nm, 1024, 6, 0, 0)
    f, g = rng.standard_normal(1024), rng.standard_normal(1024)
    v = abs(np.dot(S.apply(f), g) - np.dot(f, S.adj(g))) / (np.linalg.norm(f) * np.linalg.norm(g))
    add('adjoint', '|<Sf,g>-<f,S*g>|/(|f||g|) ' + nm, 0.0, 1e-10, float(v))
    add('constant', 'max|S 1| ' + nm, 0.0, 1e-10, float(np.max(np.abs(S.apply(np.ones(1024))))))

kh = mu.kappa('haar', 0)
add('Haar Im kappa', 'Im kappa', float(np.pi / (8 * np.log(2))), 1e-5, float(kh.imag))
add('Haar |Re kappa|', '|Re kappa|', 0.0, 1e-8, float(abs(kh.real)))

_, MT = mu.translation_symbol('haar', 0)
add('Haar Im m_T range', 'min Im m_T', 0.5103, 5e-4, float(MT.imag.min()))
add('Haar Im m_T range', 'max Im m_T', 0.6208, 5e-4, float(MT.imag.max()))

_, rp = mu.q_ripples('haar', 0, (1, 2, 4, 5))
for Q, e, r in zip((1, 2, 4, 5), (9.93, 1.12, 0.327, 0.060), rp):
    add('Haar ripple', 'Q=%d ripple %%' % Q, e, 0.03, float(100 * r), 'rel')
try:
    mu.q_ripples('haar', 0, (1, 2, 4, 5), n=65)
    add('grid ValueError', 'n=65 raises', True, 'exact', False, 'exact')
except ValueError:
    add('grid ValueError', 'n=65 raises', True, 'exact', True, 'exact')

ks = mu.kappa('sym4', 3, **KW10)
add('sym4 m=3 kappa', 'Re', -0.0074, 5e-4, float(ks.real)); add('sym4 m=3 kappa', 'Im', 0.4610, 5e-4, float(ks.imag))
kd = mu.kappa('db4', 4, **KW10)
add('db4 m=4 kappa', 'Re', 0.0104, 5e-4, float(kd.real)); add('db4 m=4 kappa', 'Im', -0.4566, 5e-4, float(kd.imag))

for nm, e in (('db4', 43.18), ('sym4', 85.74)):
    k0 = mu.kappa(nm, 0, **KW10)
    add('unaligned |c0| m=0', '|c0|=1/|Im kappa| ' + nm, e, 0.01, float(1 / abs(k0.imag)), 'rel')

N, LMAX, Q = 16384, 8, 4
re_, im_ = [], []
for k in range(192, 769, 96):
    e = np.exp(2j * np.pi * k * np.arange(N) / N)
    kh_ = np.vdot(e, dqs(e, 'sym4', 3, Q, 'fft', lmax=LMAX)) / np.vdot(e, e)
    re_.append(kh_.real); im_.append(kh_.imag)
re_m, im_m = float(np.mean(re_)), float(np.mean(im_))
l12, l13 = (mu.kappa('sym4', 3, level=l, u_min=-10, u_max=6, n=1600) for l in (12, 13))
kR = 2 * l13 - l12
T1 = 'signal kappa [ref=level-10 symbol]'
T2 = 'signal kappa [ref=converged symbol]'
add(T1, 'Re', float(ks.real), 5e-4, re_m); add(T1, 'Im*sgn k', float(ks.imag), 5e-4, im_m)
add(T2, 'Re', float(kR.real), 5e-4, re_m); add(T2, 'Im*sgn k', float(kR.imag), 5e-4, im_m)

for nm in ('db4', 'sym4'):
    for md in (0, 1):
        add('offset map', '%s m_d=%d -> m' % (nm, md), md + 3, 'exact', int(mu.discrete_to_continuous_offset(nm, md)), 'exact')

Nn = 4096; Sx = Shift('haar', Nn, 6, 0, 0); errs = []
for sd in range(5):
    r = np.random.default_rng(sd); F = np.fft.fft(r.standard_normal(Nn)); F[64:-63] = 0
    f = np.fft.ifft(F).real
    ta = time.time(); ex = cycle_average(Sx, f); t_ex = time.time() - ta
    ta = time.time(); mc = monte_carlo_average(Sx, f, M=1024, seed=4 + sd); t_mc = time.time() - ta
    errs.append(np.linalg.norm(mc - ex) / np.linalg.norm(ex))
add('exhaustive vs MC(M=1024)', 'rel L2 mean of 5 inputs (report ~0.017)', 0.017, 'record', float(np.mean(errs)), 'record')
add('exhaustive vs MC(M=1024)', 'time exhaustive s (report 0.0046)', 0.0046, 'record', float(t_ex), 'record')
add('exhaustive vs MC(M=1024)', 'time MC s (report 0.069)', 0.069, 'record', float(t_mc), 'record')

with open(HERE / 'tests_result.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['test', 'quantity', 'expected', 'tolerance', 'measured', 'pass']); w.writeheader(); w.writerows(ROWS)
nf = [r for r in ROWS if r['pass'] is False]
print('\n%d rows, %d fail, %.0fs' % (len(ROWS), len(nf), time.time() - t0))
for r in nf: print('FAIL:', r['test'], '|', r['quantity'], r['expected'], r['measured'])
sys.exit(1 if nf else 0)
