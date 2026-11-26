# Random Dyadic Shift Scattering: code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24518588.svg)](https://doi.org/10.5281/zenodo.24518588) [![arXiv](https://img.shields.io/badge/arXiv-2611.33907-b31b1b.svg)](https://arxiv.org/abs/2611.33907)

Scripts and numerical outputs for the paper

> Daewon Chung, *Random Dyadic Shift Scattering: Scale-Uniform Deformation Stability and Phase-Aligned Hilbert Multipliers*, preprint, [arXiv:2611.33907](https://arxiv.org/abs/2611.33907) (submitted).

Every table and figure of the paper that involves a computation can be regenerated from this repository. The JSON, CSV and `.tex` files in the repository are the outputs used in the paper, so the tables can also be checked without rerunning the expensive computations.

## Requirements

Python 3.11 or later with NumPy, SciPy, PyWavelets, Matplotlib and mpmath:

```
pip install -r requirements.txt
```

The final computations used Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, PyWavelets 1.8.0, mpmath 1.4.1 and Matplotlib 3.11.2. All computations run on a CPU in double precision, except where interval arithmetic (mpmath) is stated. Random inputs use fixed seeds, which are set in the scripts.

## Layout

| Folder | Contents |
|---|---|
| `verify/` | Aligned-generator constants (Table 1), Corollary 3.3 |
| `theory_J/` | Deformation experiments: worst cases, Lanczos, extensions, BMO, affine symbols |
| `paper_final/` | Scripts that turn the outputs into the paper's tables and figures, and the resulting `tab_*.tex` and `figs/*.pdf` |
| `paper_final/review/` | Certified ripples (interval arithmetic) and independent recomputation of Table 1 |
| `paper_final/fig3/` | Figure 3 (covariance error against `||tau'||` and `||tau||`) |
| `early_scripts/` | Figures 1 and 2, and early implementation checks |
| `regression/` | The reference implementation `dssn_core` 0.2.0 and its 31 regression checks |

Scripts are run from their own folder, for example `cd theory_J && python ej_worst.py ...`. Scripts in `paper_final/` read the JSON files in `../theory_J` by default; another folder can be passed as the first argument.

## Which script produces which result

Numbers refer to the submitted version of the paper. Tables S1 and S2 are in its supplementary material, which is also appended to the arXiv version.

| Result in the paper | Script | Output |
|---|---|---|
| Table 1 (aligned generators, ripples), Corollary 3.3 | `verify/verify_constants.py` | `verify_constants.json` |
| Table 1, independent recomputation; sign check of Proposition 3.1(c) | `paper_final/review/recompute_tab1.py`, `check_sign.py` | `recompute_tab1.json` |
| Haar ripples, certified (Corollary 3.3, Section 3) | `paper_final/review/cert_ripple.py` | `cert_ripple.json` |
| Figure 1 (averaged Haar multiplier) | `early_scripts/mult.py`, `e1_discrete.py`, then `figs_en.py` | `paper_final/figs/fig1_en.pdf` |
| Figure 2 (lattice dependence; Hilbert-coefficient integrand) | `early_scripts/e2_transl.py`, `e8.py`, `e9.py`, then `figs_en.py` | `fig2_en.pdf` (left), `fig4_en.pdf` (right) |
| Table 2 (typical inputs) and linear-defect checks | `theory_J/ej_check.py`, `ej_run.py`, `ej_linear.py` | `ej_run.json`, `ej_linear.json` |
| Table S1 (supplementary; channels, second layer, dilation averaging) | `theory_J/ej_ext.py layers\|dil <gen> <Js>`, then `paper_final/make_tab_ext.py` | `ej_ext_*.json`, `tab_ext.tex` |
| Table 3 and Figure 4(a) (worst-case inputs) | `theory_J/lz_worst.py <gen> <eps> <warp> <Js> <steps> <full-Js>` (Lanczos), `theory_J/ej_worst.py` (power iteration), then `paper_final/make_tab_worst.py` | `lz_*.json`, `ej_worst_*.json`, `tab_worst.tex` |
| Table 4 and Figure 4(b) (single-lattice BMO norm) | `theory_J/ej_bmo.py` | `ej_bmo_*.json` |
| Table 5 (affine warps, exact averaged constants) | `theory_J/ej_affine_symbol.py` | `ej_affine_symbol_*.json` |
| Table S2 (supplementary; Lanczos against full matrix) | `theory_J/lz_worst.py` with `<full-Js>` | `lz_haar_smooth_eps0.1.json`, `lz_sym4_smooth_eps0.1.json` |
| Remark 5.7 (`sigma_eps`, exact single-lattice norms) | `theory_J/ej_single_lowerbound.py` | `ej_single_lowerbound_*.json` |
| Low-frequency control runs | `theory_J/ej_lowfreq2.py` | `ej_lowfreq2_*.json` |
| Figure 3 (deformation, aligned generators) | `paper_final/fig3/e3_deform_aligned.py`, then `make_fig3.py` | `e3_aligned.json`, `fig3_en.pdf` |
| Figure 4 | `paper_final/make_fig5.py` | `paper_final/figs/fig5_uniform.pdf` |
| Implementation checks | `regression/run_tests.py` | `regression/tests_result.csv` |

The file names `fig4_en.pdf` and `fig5_uniform.pdf` are historical: they are the right panel of Figure 2 and Figure 4 of the paper.

## Running times

On one CPU core: the table and figure scripts in `paper_final/` take seconds; `verify/verify_constants.py` about 7 minutes; `regression/run_tests.py` under a minute (it exits with code 1 because of the one expected failure explained in `regression/README.md`); the scripts in `early_scripts/` under a minute each. The worst-case computations in `theory_J/` (power iteration and Lanczos at `N = 8192`, up to `J = 11`) can take hours per configuration; their outputs are included.

## License

MIT (see `LICENSE`). If you use this code, please cite the paper and the archived code (see `CITATION.cff`):

> D. Chung, *Random Dyadic Shift Scattering: code and data*, Zenodo (2026), https://doi.org/10.5281/zenodo.24518588
