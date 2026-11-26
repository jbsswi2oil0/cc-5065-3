"""Kernel density estimation of irradiance from a photon map.

Irradiance estimate (per color channel):

    E(x) = (1 / R^2) * sum_p Phi_p * K(|x - x_p| / R)

K is a unit-area 2D density (see kernels.py), so the factor 1/R^2 is the
correct area normalization.  Only photons sharing the query surface id and
whose incoming direction is compatible with the query normal are accepted:
this prevents the caustic from leaking through a thin wall onto its back
face, and rejects grazing photons that would otherwise smear the boundary.
"""
from math import sqrt

from . import vec3
from .kernels import KERNELS

NORMAL_COS_THRESHOLD = 0.5
UNDER_SAMPLE_PHOTONS = 8
BUDGET_MIN_PHOTONS = 2


def estimate(photon_map, point, normal, surface_id, radius,
             kernel="epanechnikov", max_radius=None):
    """Estimate irradiance; returns irradiance(rgb), radius used, n_eff.

    radius:       initial (minimum) search radius.
    max_radius:   if given, the radius grows until UNDER_SAMPLE_PHOTONS
                  compatible photons are found or this cap is reached.
    """
    kfun = KERNELS[kernel]
    r = float(radius)
    cap = float(max_radius) if max_radius is not None else r
    photons = []
    # Grow in discrete steps; each step re-queries the hash grid so photons
    # outside the initial cell can still be borrowed on the same surface.
    while True:
        candidates = photon_map.neighborhood(point, r, surface_id)
        photons = [p for p in candidates
                   if vec3.dot(vec3.mul(p.direction, -1.0), normal)
                   > NORMAL_COS_THRESHOLD]
        if len(photons) >= UNDER_SAMPLE_PHOTONS or r >= cap - 1e-9:
            break
        r = min(cap, r * 1.5)

    n_eff = len(photons)
    irradiance = [0.0, 0.0, 0.0]
    if n_eff:
        inv_r2 = 1.0 / (r * r)
        for ph in photons:
            dist = sqrt((ph.pos[0] - point[0]) ** 2
                        + (ph.pos[1] - point[1]) ** 2
                        + (ph.pos[2] - point[2]) ** 2)
            kv = kfun(dist / r)
            if kv <= 0.0:
                continue
            w = kv * inv_r2
            for c in range(3):
                irradiance[c] += ph.power[c] * w

    undersampled = n_eff < UNDER_SAMPLE_PHOTONS
    budget_exhausted = n_eff < BUDGET_MIN_PHOTONS
    return {
        "irradiance": tuple(irradiance),
        "radius": r,
        "n_eff": n_eff,
        "undersampled": undersampled,
        "budget_exhausted": budget_exhausted,
    }
