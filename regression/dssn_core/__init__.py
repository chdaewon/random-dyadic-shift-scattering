__version__ = "0.2.0"
from .shift import Shift, cycle_average, monte_carlo_average
from .multipliers import (kappa, q_ripples, best_offset, discrete_to_continuous_offset,
                          dilation_quadrature_symbol, translation_symbol, psi_samples, discrete_symbol)
from .features import dssn_feature, dilation_quadrature_feature, warp_periodic, resample_periodic
from .dilation import dilation_quadrature_shift, node_lengths, resample_length
from .compat import trapezoid
