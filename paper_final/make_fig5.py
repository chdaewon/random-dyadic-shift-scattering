"""Figure 5: lattice average (flat in J) versus a single lattice (grows like sqrt(J+1)).
Reads the JSON outputs in ../theory_J (or a directory given as the first argument)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p for p in [Path(__file__).resolve().parent.parent / 'theory_J'] if p.exists())
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # validated categorical slots 1-3
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MULTI_NORM = 0.7469   # ||tau'||_inf / eps for the multi-scale warp (seed 11, periods 2^4..2^12)

def worst(fname, norm=1.0, lzname=None):
    """Lanczos values (lz_*.json) where available; remaining J from power iteration (lower bounds)."""
    pw = {r['J']: r['sigma_over_eps'] / norm for r in json.load(open(D / fname))}
    lz = {}
    if lzname and (D / lzname).exists():
        lz = {r['J']: r['sigma_over_eps'] / norm for r in json.load(open(D / lzname))}
    Js = sorted(set(pw) | set(lz))
    return (np.array(Js), np.array([lz.get(j, pw.get(j)) for j in Js]), np.array([j in lz for j in Js]))

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
                     'font.family': 'serif', 'mathtext.fontset': 'cm'})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)

series = [('Haar, smooth warp', 'ej_worst_haar_N8192.json', 'lz_haar_smooth_eps0.1.json', 1.0, BLUE, 'o'),
          ('sym4, smooth warp', 'ej_worst_sym4_N8192.json', 'lz_sym4_smooth_eps0.1.json', 1.0, ORANGE, 's'),
          ('Haar, multi-scale warp', 'ej_worst_haar_N8192_eps0.05_multi.json', 'lz_haar_multi_eps0.05.json', MULTI_NORM, AQUA, '^')]
for lab, f, fl, nrm, c, mk in series:
    J, y, isl = worst(f, nrm, fl)
    a.plot(J, y, color=c, lw=1.5, label=lab)
    a.plot(J[isl], y[isl], ls='none', color=c, marker=mk, ms=4.5, mec='white', mew=0.8)
    a.plot(J[~isl], y[~isl], ls='none', color=c, marker=mk, ms=4.5, mfc='white', mec=c, mew=1.0)
a.plot([], [], ls='none', marker='o', ms=4.5, mfc='white', mec=MUTED, label='power iteration (lower bound)')
a.set_ylim(0, 3.2); a.set_xlim(-0.3, 8.3)
a.set_xlabel('$J$ (number of scales $-1$)')
a.set_ylabel(r'$\sigma_{\max}/\|\tau^\prime\|_\infty$')
a.set_title('(a) lattice average: worst case', fontsize=9, loc='left', color=INK)
a.legend(frameon=False, fontsize=7.5, loc='lower right')

d = json.load(open(D / 'ej_bmo_sym4_smooth.json'))
for eps, mk, ls in [(0.1, 'o', '-'), (0.2, 's', '-')]:
    rr = sorted([r for r in d if r['kind'] == 'smooth' and r['eps'] == eps], key=lambda r: r['J'])
    J = np.array([r['J'] for r in rr]); y = np.array([r['bmo'] for r in rr]) / eps
    c = np.sum(y * np.sqrt(J + 1)) / np.sum(J + 1)
    Jf = np.linspace(0, 8, 100)
    b.plot(Jf, c * np.sqrt(Jf + 1), color=MUTED, lw=0.9, ls='--')
    b.plot(J, y, color=ORANGE, lw=1.5, marker=mk, ms=4.5, mec='white', mew=0.8,
           label=rf'sym4, $\varepsilon={eps}$')
    b.annotate(rf'$\varepsilon={eps}$', (J[-1], y[-1]), xytext=(4, 0), textcoords='offset points',
               va='center', fontsize=7.5, color=INK)
b.plot([], [], color=MUTED, lw=0.9, ls='--', label=r'fit $c\sqrt{J+1}$')
b.set_xlim(-0.3, 9.2); b.set_ylim(0, 0.7)
b.set_xlabel('$J$')
b.set_ylabel(r'$\|\tilde G_\beta^{*}1\|_{\mathrm{BMO}}/\varepsilon$')
b.set_title('(b) one lattice: grows like $\\sqrt{J+1}$', fontsize=9, loc='left', color=INK)
b.legend(frameon=False, fontsize=7.5, loc='upper left')
for ax in (a, b):
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
out = Path(__file__).resolve().parent / 'figs' / 'fig5_uniform.pdf'
fig.savefig(out); fig.savefig(out.with_suffix('.png'), dpi=200)
print('wrote', out)
