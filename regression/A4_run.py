'''A4: signal-level dilation quadrature A_Q. khat(k)=<A_Q e_k,e_k>/||e_k||^2, sym4 (m=3), Q=4.
Outputs: A4_signal_level.csv (tones), A4_method_comparison.csv (wideband, 5 seeds), A4_summary.json.
References: kappa level 10 (default; the value used in an earlier draft), level 13, Richardson 2k13-k12 (first-order dx error removed).'''
import sys, csv, json, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dssn_core import multipliers as mu
from dssn_core.dilation import dilation_quadrature_shift as dqs

NAME, MC, Q, TOL = 'sym4', 3, 4, 5e-4
CONFIGS = [(16384, 8, 192, 768, 24, ('fft', 'spline3'), 'main'), (4096, 10, 48, 192, 12, ('fft',), 'coarse_nodes')]


def tone(N, lmax, k, method):
    e = np.exp(2j * np.pi * k * np.arange(N) / N)
    return np.vdot(e, dqs(e, NAME, MC, Q, method, lmax=lmax)) / N


def band(N, seed, klo, khi):
    F = np.fft.fft(np.random.default_rng(seed).standard_normal(N))
    k = np.abs(np.fft.fftfreq(N, 1.0 / N)); F[(k < klo) | (k > khi)] = 0
    return np.fft.ifft(F).real


def sym_apply(f, kap):
    k = np.fft.fftfreq(len(f), 1.0 / len(f))
    return np.fft.ifft(np.fft.fft(f) * (kap.real + 1j * kap.imag * np.sign(k))).real


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


if __name__ == '__main__':
    t0 = time.time()
    kl = {l: mu.kappa(NAME, MC, level=l, u_min=-10, u_max=6, n=1600) for l in (10, 12, 13)}
    kr = 2 * kl[13] - kl[12]; k10 = kl[10]
    print('kappa L10 %.5f%+.5fi  L13 %.5f%+.5fi  Richardson %.5f%+.5fi' % (k10.real, k10.imag, kl[13].real, kl[13].imag, kr.real, kr.imag))
    summ = dict(tol=TOL, kappa_level10=[k10.real, k10.imag], kappa_level13=[kl[13].real, kl[13].imag], kappa_richardson=[kr.real, kr.imag], configs={})
    rows = []
    for N, lmax, klo, khi, st, methods, tag in CONFIGS:
        for meth in methods:
            re, im = [], []
            for k in range(klo, khi + 1, st):
                v = tone(N, lmax, k, meth); re.append(v.real); im.append(v.imag)
                rows.append(dict(config=tag, N=N, lmax=lmax, Q=Q, method=meth, k=k, re=v.real, im_sgnk=v.imag))
            for k in (-(klo + khi) // 3, -(klo + khi) // 2):
                v = tone(N, lmax, k, meth)
                rows.append(dict(config=tag, N=N, lmax=lmax, Q=Q, method=meth, k=k, re=v.real, im_sgnk=v.imag * np.sign(k)))
            r, i = float(np.mean(re)), float(np.mean(im))
            d = dict(re_mean=r, im_mean=i, re_std=float(np.std(re)), im_std=float(np.std(im)), n_tones=len(re))
            for nm, ref in (('level10', k10), ('level13', kl[13]), ('richardson', kr)):
                d['dre_' + nm] = r - ref.real; d['dim_' + nm] = i - ref.imag
                d['pass_' + nm] = bool(abs(r - ref.real) <= TOL and abs(i - ref.imag) <= TOL)
            summ['configs'][tag + '_' + meth] = d
            print(tag, meth, 'Re %.5f Im %.5f | dRe(L10) %.2e dRe(Rich) %.2e dIm(Rich) %.2e | pass L10 %s Rich %s' % (r, i, d['dre_level10'], d['dre_richardson'], d['dim_richardson'], d['pass_level10'], d['pass_richardson']))
    # wideband comparison fft vs spline3, N=16384, lmax=8, 5 seeds
    N, lmax = 16384, 8; cmp_rows = []
    for seed in range(5):
        for kind in ('band', 'full'):
            f = band(N, seed, 192, 768) if kind == 'band' else np.random.default_rng(100 + seed).standard_normal(N)
            yf = dqs(f, NAME, MC, Q, 'fft', lmax=lmax); ys = dqs(f, NAME, MC, Q, 'spline3', lmax=lmax)
            ref = sym_apply(f, kr)
            cmp_rows.append(dict(seed=seed, input=kind, N=N, lmax=lmax, Q=Q, fft_vs_spline_rel=rel(yf, ys), fft_vs_symbol_rel=rel(yf, ref), spline3_vs_symbol_rel=rel(ys, ref), symbol_norm_ratio=float(np.linalg.norm(ref) / np.linalg.norm(f))))
    for c in cmp_rows: print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()})
    for fn, rr in (('A4_signal_level.csv', rows), ('A4_method_comparison.csv', cmp_rows)):
        with open(HERE / fn, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    for inp in ('band', 'full'):
        sel = [c for c in cmp_rows if c['input'] == inp]
        summ['wideband_' + inp] = {k: float(np.mean([c[k] for c in sel])) for k in ('fft_vs_spline_rel', 'fft_vs_symbol_rel', 'spline3_vs_symbol_rel')}
    json.dump(summ, open(HERE / 'A4_summary.json', 'w'), indent=1)
    print('done %.0fs' % (time.time() - t0))
