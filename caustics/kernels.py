"""2D radial density-estimation kernels, normalized to unit area over the disk r<=R.

Each kernel is a function of u = r/R in [0, 1] and returns the *density*
K(u).  Normalization condition: integral over the disk of R^2 K(r/R) dA = 1,
i.e. 2*pi * integral_0^1 K(u) u du = 1.

The estimator uses these as proper probability densities (not raw weights),
so the photon count cancels correctly in flat illumination: doubling the
number of photons halves their individual power and the estimate is unchanged.
"""
from math import exp, pi


_EPAN_K = 2.0 / pi            # int 2*pi*(2/pi)*(1-u^2)*u du over [0,1] == 1
_BOX_K = 1.0 / pi            # int 2*pi*(1/pi)*u du over [0,1] == 1
_GAUSS_S2 = 0.25             # sigma^2 for the truncated gaussian
_GAUSS_C = _GAUSS_S2 * (1.0 - exp(-1.0 / _GAUSS_S2))
_GAUSS_K = 1.0 / (pi * _GAUSS_C)


def box(u):
    if u > 1.0:
        return 0.0
    return _BOX_K


def epanechnikov(u):
    if u > 1.0:
        return 0.0
    return _EPAN_K * (1.0 - u * u)


def gaussian(u):
    if u > 1.0:
        return 0.0
    return _GAUSS_K * exp(-(u * u) / _GAUSS_S2)


KERNELS = {
    "box": box,
    "epanechnikov": epanechnikov,
    "gaussian": gaussian,
}


def kernel_integral(name, steps=4000):
    """Numerically verify 2*pi*int_0^1 K(u) u du == 1 (used by tests)."""
    k = KERNELS[name]
    total = 0.0
    du = 1.0 / steps
    for i in range(steps):
        u = (i + 0.5) * du
        total += k(u) * u * du
    return 2.0 * pi * total
