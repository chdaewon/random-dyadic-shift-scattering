"""Table tab:ext from ej_ext_layers_*.json and ej_ext_dil_*.json."""
import json, sys, numpy as np
from pathlib import Path
DATA=Path(sys.argv[1]) if len(sys.argv)>1 else next(p for p in [Path(__file__).resolve().parent.parent/"theory_J", Path(".")] if p.exists())
rows=[]
def f4(v): return '---' if v is None else ('%.3f'%v)[1:] if v<1 else '%.2f'%v
lines=[]
stats={}
for w,W in [('haar','Haar'),('sym4','sym4')]:
    L={(r['J'],r['input']):r for r in json.load(open(DATA/f'ej_ext_layers_{w}.json'))}
    D={(r['J'],r['input']):r for r in json.load(open(DATA/f'ej_ext_dil_{w}.json'))}
    for inp,IN in [('white0','white'),('pink0','pink')]:
        r1=[L[(J,inp)]['sep']/np.sqrt(L[(J,inp)]['Pact']) for J in range(7)]
        r2=[None]+[L[(J,inp)]['sep2']/np.sqrt(L[(J,inp)]['P2act']) for J in range(1,7)]
        rd=[D[(J,inp)]['dil']/L[(J,inp)]['sum'] for J in range(7)]
        q=[r2[J]/r1[J] for J in range(1,7)]; e=[L[(J,inp)]['E2']/L[(J,inp)]['E1'] for J in range(1,7)]
        stats[(w,inp)]=dict(r1=(min(r1),max(r1)),ratio=(min(q),max(q)),E2E1=(min(e),max(e)),dil=(min(rd),max(rd)))
        lines.append(f"{W} & {IN} & $\\rho_{{\\rm sep}}$ & "+' & '.join(f4(v) for v in r1)+r'\\')
        lines.append(f" &  & $\\rho_2$ & "+' & '.join(f4(v) for v in r2)+r'\\')
        lines.append(f" &  & $e_\\Pi/e$ & "+' & '.join(f4(v) for v in rd)+r'\\')
    lines[-1]+=r'\midrule' if w=='haar' else ''
tex=r"""\begin{table}[t]\centering\small
\begin{tabular}{lll ccccccc}\toprule
gen. & input & & $J=0$ & 1 & 2 & 3 & 4 & 5 & 6\\\midrule
"""+'\n'.join(lines)+r"""\bottomrule
\end{tabular}
\caption{Scale-separated channels, second layer and dilation averaging ($N_{\rm s}=2^{14}$, finest cell 16 samples, smooth warp, $\eps=0.1$). $\rho_{\rm sep}=\mathcal E_{\rm sep}(f)/\norm{P_{[0,J]}f}$ is the error of \cref{thm:sep} per unit of input energy in the active scale window; $\rho_2$ is the second-layer error $\big(\sum_p\norm{\Psi'_{2,p}(L_\tau f)-L_\tau\Psi'_{2,p}f}^2\big)^{1/2}$ per unit of energy of the first-layer outputs in the second-layer windows ($j_2<j_1$); $e_\Pi/e$ is the covariance error of the dilation-averaged channel $\Psi'_{1,\Pi}$, with $\Pi$ uniform on $\{1,\frac87,\frac43,\frac85\}$, divided by that of the translation-averaged channel $\Psi_1'$.}\label{tab:ext}
\end{table}
"""
Path(__file__).resolve().parent.joinpath('tab_ext.tex').write_text(tex); print(tex)
for k,v in stats.items(): print(k,{a:(round(b[0],3),round(b[1],3)) for a,b in v.items()})
