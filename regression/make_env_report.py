import sys, platform, subprocess
from pathlib import Path
import numpy, scipy, pywt, sympy, pandas, matplotlib, torch

out = Path(__file__).resolve().parents[1] / 'env_report_A.txt'
L = [f'python {platform.python_version()} on {platform.platform()}',
     f'numpy {numpy.__version__}', f'scipy {scipy.__version__}', f'pywavelets {pywt.__version__}',
     f'torch {torch.__version__} (cuda available: {torch.cuda.is_available()}, cuda {torch.version.cuda})',
     f'sympy {sympy.__version__}', f'pandas {pandas.__version__}', f'matplotlib {matplotlib.__version__}',
     f'np.trapezoid available: {hasattr(numpy, "trapezoid")}']
try:
    import jupyter_core, nbformat, nbconvert
    L += [f'jupyter_core {jupyter_core.__version__}', f'nbformat {nbformat.__version__}', f'nbconvert {nbconvert.__version__}']
except Exception as e:
    L.append(f'jupyter modules: {e}')
if torch.cuda.is_available():
    L.append('GPU: ' + torch.cuda.get_device_name(0) + f', {torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MiB')
try:
    L.append(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=20).stdout.strip())
except Exception as e:
    L.append(f'nvidia-smi: {e}')
for mod in ('pytest',):
    try:
        __import__(mod); L.append(f'{mod}: present')
    except ImportError:
        L.append(f'{mod}: absent (as expected)')
L.append('network/installs: none used')
out.write_text('\n'.join(L) + '\n')
print('\n'.join(L))
