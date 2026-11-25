'''A3: robustness of the Haar Q=5 dilation-quadrature ripple (about 0.060%) under changes of scale range, kappa interval, sampling level, grid and ripple definition.'''
import sys, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dssn_core import multipliers as mu

Qs = (1, 2, 4, 5)
KEX = 1j * np.pi / (8 * np.log(2))
rows = []


def rip(level=10, n=320, j_min=-12, j_max=8, sampling='midpoint', kap=None, kkw=None):
    s, MT = mu.translation_symbol('haar', 0, level=level, n=n, j_min=j_min, j_max=j_max, haar_sampling=sampling)
    if kap is None:
        kk = dict(level=level, u_min=-10, u_max=6, n=1600, haar_sampling=sampling)
        if kkw: kk.update(kkw)
        kap = mu.kappa('haar', 0, **kk)
    return [100 * float(np.max(np.abs(mu._quad_average(MT, Q) - kap)) / abs(kap.imag)) for Q in Qs], kap, MT


def add(group, setting, vals, note=''):
    rows.append(dict(group=group, setting=setting, Q1=vals[0], Q2=vals[1], Q4=vals[2], Q5=vals[3], note=note))
    print('%-14s %-34s %8.3f %8.3f %8.4f %8.4f %s' % (group, setting, vals[0], vals[1], vals[2], vals[3], note))


v, kap, MT = rip(); add('base', 'j[-12,8] k[-10,6] lvl10 n=320', v)
for a, b in ((-8, 4), (-10, 6), (-14, 10), (-16, 12), (-20, 16), (-30, 24)):
    add('scale sum', 'j in [%d,%d]' % (a, b), rip(j_min=a, j_max=b)[0])
for a, b in ((-6, 4), (-8, 5), (-12, 8), (-16, 10), (-20, 14)):
    add('kappa interval', 'u in [%d,%d]' % (a, b), rip(kkw=dict(u_min=a, u_max=b, n=100 * (b - a)))[0])
add('kappa exact', 'kappa = i pi/(8 ln2)', rip(kap=KEX)[0])
for lv in (8, 9, 11, 12, 13):
    add('sampling lvl', 'level %d' % lv, rip(level=lv)[0])
add('sampling', 'left end point (0.1.x), lvl10', rip(sampling='left')[0], 'not the 0.2.0 default')
for n in (640, 1280, 3200):
    add('grid n', 'n=%d' % n, rip(n=n)[0])
add('combined', 'j[-20,16] exact kappa lvl12 n=3200', rip(level=12, n=3200, j_min=-20, j_max=16, kap=KEX)[0])
# alternative ripple definitions at base
im = MT.imag
for Q in Qs:
    MQ = mu._quad_average(MT, Q).imag
    rows.append(dict(group='definition', setting='Q=%d (max-min)/(2 mean)' % Q, Q1=float('nan'), Q2=float('nan'), Q4=float('nan'), Q5=float('nan'),
                     note='%.4f%% vs max|M-k|/Im k = %.4f%%' % (100 * (MQ.max() - MQ.min()) / (2 * MQ.mean()), 100 * np.max(np.abs(MQ - kap.imag)) / kap.imag)))
print('definition (max-min)/(2 mean) for Q=5: %s' % rows[-1]['note'])
# evaluation grid: M_Q(s) evaluated directly on a plotting grid, not on the index grid
for N in (50, 100, 128, 200, 256, 500, 2000):
    s = np.linspace(0, 1, N)
    out = []
    for Q in Qs:
        acc = 0
        for q in range(Q):
            acc = acc + mu.translation_symbol('haar', 0, s_grid=s + q / Q, level=10)[1]
        out.append(100 * float(np.max(np.abs(acc / Q - kap)) / abs(kap.imag)))
    add('plot grid', 's=linspace(0,1,%d)' % N, out)
# discrete implementation: cycle-averaged discrete shift on a padded circle, symbol at fractional frequency
N0 = 4096; L = 4 * N0; lmax = 10
h = mu.discrete_symbol('haar', 0, L, lmax); hc = np.fft.fftshift(h); n_ = np.arange(-L // 2, L // 2)
ks = np.arange(48, 96.0001, 0.25)
for lab, ks_ in (('band k4096 in [48,96]', ks), ('band k4096 in [32,64]', np.arange(32, 64.0001, 0.25))):
    out = []; out2 = []
    for Q in Qs:
        rq = 2.0 ** ((np.arange(Q) + 0.5) / Q)
        MQ = np.mean([np.exp(-2j * np.pi * np.outer(ks_ / N0 / r, n_)) @ hc for r in rq], axis=0)
        out.append(100 * float(np.max(np.abs(MQ - kap)) / abs(kap.imag)))
        out2.append(100 * float((MQ.imag.max() - MQ.imag.min()) / (2 * MQ.imag.mean())))
    add('discrete L=4N0', lab + ' vs exact kappa', out)
    add('discrete L=4N0', lab + ' (max-min)/(2 mean)', out2)
with open(HERE / 'A3_candidates.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['group', 'setting', 'Q1', 'Q2', 'Q4', 'Q5', 'note']); w.writeheader(); w.writerows(rows)
