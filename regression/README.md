# dssn_core 0.2.0 and regression checks

`dssn_core/` is the reference implementation of the dyadic shift `S_beta`, its exhaustive and Monte Carlo lattice averages, the averaged multipliers and the dilation quadrature. `run_tests.py` runs 31 checks against known values (no pytest needed):

```
python run_tests.py      # writes tests_result.csv
```

Columns of `tests_result.csv`: test, quantity, expected, tolerance, measured, pass. Rows marked `record` are timings and are not pass/fail.

## The one failing row

The row `signal kappa [ref=level-10 symbol], Re` fails, and this is expected. The real part of the symbol-level constant `kappa` for sym4 (child offset `m = 3`) depends on the PyWavelets `wavefun` level used to sample `psi`:

| level | 10 | 11 | 12 | 13 | Richardson |
|---|---|---|---|---|---|
| Re kappa | -0.00736 | -0.00933 | -0.01032 | -0.01081 | -0.01130 |
| Im kappa | 0.46098 | 0.46099 | 0.46100 | 0.46100 | 0.46100 |

This is a first-order sampling error: the distance to the limit halves with each level. The signal-level measurement, -0.01131, agrees with the converged value, and the row `signal kappa [ref=converged symbol]` passes. The imaginary part (the Hilbert coefficient) and db4 are not affected. The constants in Table 1 of the paper do not use sampled wavelets: `verify/verify_constants.py` computes them from the infinite-product spectrum.

## Other scripts

| Script | Output | Content |
|---|---|---|
| `A1_run.py` | `A1_haar_midpoint.csv` | Haar `Re kappa` at sampling levels 10–13, left-end-point against midpoint sampling |
| `A3_run.py` | `A3_candidates.csv` | Robustness of the Haar `Q = 5` ripple (about 0.060%) |
| `A4_run.py` | `A4_*.csv`, `A4_summary.json` | Signal-level dilation quadrature `A_Q` against the symbol-level prediction |
| `A5_run.py` | `A5_offset_map.csv` | Integer shift between discrete and continuous child offsets |
| `make_env_report.py` | `env_report_A.txt` | Environment of the reported run |
