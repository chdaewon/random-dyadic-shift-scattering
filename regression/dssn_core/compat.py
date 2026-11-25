"""numpy compatibility helpers (A6)."""
import numpy as np

# np.trapz was removed in numpy 2.x (and deprecated in 2.0); np.trapezoid exists from numpy 2.0.
trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
