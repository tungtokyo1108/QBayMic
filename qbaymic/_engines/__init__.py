"""Internal engine modules — the validated QBayMic implementations.

These modules are the exact, scientifically-validated implementations used in
the paper (barrier theory, the six clustering methods, and the hyperparameter
searches). They are vendored here UNMODIFIED. Because they use flat top-level
imports among themselves (e.g. ``from DMM_SVVS_Variational_v2 import ...``), we
add this directory to ``sys.path`` on import so those imports resolve without
editing any engine file. Public users should not import from here directly; use
the top-level :mod:`qbaymic` API instead.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
