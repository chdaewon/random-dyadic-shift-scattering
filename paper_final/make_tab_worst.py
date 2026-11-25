"""Builds tab_worst.tex: Lanczos values (lz_*.json) where available, else power-iteration lower bounds (ej_worst_*.json, marked)."""
import json, sys
from pathlib import Path
D=Path(sys.argv[1]) if len(sys.argv)>1 else next(p for p in [Path(__file__).resolve().parent.parent/'theory_J', Path('.')] if p.exists())
MULTI_NORM=0.7469
def sci(v):
    m,e=f'{v:.0e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
rows=[('Haar','smooth','0.1','ej_worst_haar_N8192.json','lz_haar_smooth_eps0.1.json',1.0),
      ('Haar','smooth','0.3','ej_worst_haar_N8192_eps0.3.json','lz_haar_smooth_eps0.3.json',1.0),
      ('Haar','smooth','0.02','ej_worst_haar_N8192_eps0.02_smooth.json','lz_haar_smooth_eps0.02.json',1.0),
      ('sym4','smooth','0.1','ej_worst_sym4_N8192.json','lz_sym4_smooth_eps0.1.json',1.0),
      ('sym4','smooth','0.3','ej_worst_sym4_N8192_eps0.3.json','lz_sym4_smooth_eps0.3.json',1.0),
      ('Haar','multi','0.1','ej_worst_haar_N8192_eps0.1_multi.json','lz_haar_multi_eps0.1.json',MULTI_NORM),
      ('Haar','multi','0.05','ej_worst_haar_N8192_eps0.05_multi.json','lz_haar_multi_eps0.05.json',MULTI_NORM),
      ('Haar','multi','0.02','ej_worst_haar_N8192_eps0.02_multi.json','lz_haar_multi_eps0.02.json',MULTI_NORM)]
Js=list(range(9)); out=[r'\begin{table}[h]\centering\small\setlength{\tabcolsep}{4pt}',
 r'\begin{tabular}{lll'+'c'*len(Js)+'}\\toprule',
 r'gen. & warp & $\eps$ & '+' & '.join(f'$J={j}$' if j==0 else str(j) for j in Js)+r'\\\midrule']
prev=None; maxres=0; summary={}
for g,w,e,fp,fl,nrm in rows:
    pw={r['J']:r['sigma_over_eps']/nrm for r in json.load(open(D/fp))} if (D/fp).exists() else {}
    lz={r['J']:r for r in json.load(open(D/fl))} if (D/fl).exists() else {}
    if prev is not None and prev!=(g,w): out.append(r'\addlinespace[2pt]')
    prev=(g,w); cells=[]
    for j in Js:
        if j in lz:
            cells.append(f"{lz[j]['sigma_over_eps']/nrm:.3f}"); maxres=max(maxres,lz[j]['resid']/lz[j]['theta'])
        elif j in pw: cells.append(f"{pw[j]:.3f}$^\\ast$")
        else: cells.append('---')
    out.append(f'{g} & {w} & {e} & '+' & '.join(cells)+r'\\'); summary[(g,w,e)]=(lz,pw)
out+=[r'\bottomrule\end{tabular}',
 r'\caption{Worst-case linear defect $\sigma_{\max}/\norm{\tau^\prime}_\infty$ of the lattice-averaged shift ($2^{13}$ samples, finest cell 8 samples). Unmarked entries are Lanczos values, with relative residual at most '+sci(maxres)+r' (\cref{ssec:num-worst}); entries marked ${}^\ast$ are power-iteration lower bounds. For the multi-scale warp, $\norm{\tau^\prime}_\infty=0.747\,\eps$. Every row saturates in $J$; the three multi-scale Haar rows agree to within $3\%$ at every $J$.}\label{tab:worst}',
 r'\end{table}']
Path(__file__).resolve().parent.joinpath('tab_worst.tex').write_text('\n'.join(out)+'\n'); print('\n'.join(out))
# comparison
for k,(lz,pw) in summary.items():
    print(k,[(j,round(lz[j]['sigma_over_eps'],4),round(pw.get(j,float('nan'))*(0.7469 if k[1]=='multi' else 1),4)) for j in sorted(lz)])
