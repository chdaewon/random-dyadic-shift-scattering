'''A1: Haar / db1 psi sampling, left end point (0.1.x) vs midpoint (0.2.0): Re kappa at sampling levels 10-13.'''
import sys, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dssn_core import multipliers as mu

rows = []
for name in ('haar', 'db1'):
    prev = {}
    for lv in (10, 11, 12, 13):
        for samp in ('left', 'midpoint'):
            k = mu.kappa(name, 0, level=lv, u_min=-10, u_max=6, n=1600, haar_sampling=samp)
            ratio = (prev[samp] / k.real) if (samp in prev and k.real != 0) else float('nan')
            prev[samp] = k.real
            rows.append(dict(wavelet=name, level=lv, sampling=samp, re_kappa=float(k.real), im_kappa=float(k.imag),
                             re_over_im_percent=100 * float(k.real) / float(k.imag),
                             reduction_vs_previous_level=ratio))
            print('%-5s level %2d %-8s Re=%+.3e Im=%.9f Re/Im=%.3e%% ratio_prev=%.3f' % (name, lv, samp, k.real, k.imag, 100 * k.real / k.imag, ratio))
with open(HERE / 'A1_haar_midpoint.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('exact Im = pi/(8 ln2) = %.9f' % (np.pi / (8 * np.log(2))))
