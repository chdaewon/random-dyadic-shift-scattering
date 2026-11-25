'''A5: automatic search of the integer shift between discrete and continuous child offsets.'''
import sys, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dssn_core import multipliers as mu

rows = []
for name in ('db4', 'sym4', 'db2', 'coif3', 'sym6'):
    for md in (0, 1, 2):
        info = mu.discrete_to_continuous_offset(name, md, return_info=True)
        rows.append(dict(wavelet=name, m_discrete=md, m_continuous=info['m'], shift=info['shift'], match_err=info['err'],
                         second_best_shift=info['second_shift'], second_best_err=info['second_err'],
                         margin_ratio=info['second_err'] / max(info['err'], 1e-300), symbol_max=info['sym_max'],
                         is_plus3=int(info['shift'] == 3)))
        print('%-5s md=%d -> m=%d (shift %+d) err=%.2e  2nd shift %+d err=%.2e' % (name, md, info['m'], info['shift'], info['err'], info['second_shift'], info['second_err']))
with open(HERE / 'A5_offset_map.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
