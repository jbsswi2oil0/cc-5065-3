"""Refinement logic driven by estimator footprints.

The estimator hands every query point its kernel radius (footprint) and the
number of compatible photons it found (n_eff).  The refiner turns those into
explicit masks so the UI can tell three situations apart:

* bright enough        - >= target photons, normal estimate
* genuinely dark       - footprint already at the cap, still ~0 photons,
                          and no neighbors carry energy either
* undersampled         - few photons but the footprint could still grow,
                          or local energy exists while this cell is noisy
"""
from .estimator import UNDER_SAMPLE_PHOTONS


def classify(view, target_photons=UNDER_SAMPLE_PHOTONS):
    n = view["grid_n"]
    n_eff = view["n_eff"]
    total = view["total"]
    exhausted = view["budget_exhausted"]
    labels = [["ok" for _ in range(n)] for _ in range(n)]

    def neighbor_energy(iy, ix):
        e = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j, i = iy + dy, ix + dx
                if 0 <= j < n and 0 <= i < n:
                    e += sum(total[j][i])
        return e

    for iy in range(n):
        for ix in range(n):
            if n_eff[iy][ix] >= target_photons:
                continue
            local = sum(total[iy][ix])
            if exhausted[iy][ix] and local <= 1e-12 and neighbor_energy(iy, ix) <= 1e-12:
                labels[iy][ix] = "dark"
            else:
                labels[iy][ix] = "undersampled"
    return labels


def refinement_stats(view):
    n = view["grid_n"]
    labels = classify(view)
    counts = {"ok": 0, "dark": 0, "undersampled": 0}
    for row in labels:
        for lab in row:
            counts[lab] += 1
    total_px = n * n
    return {
        "labels": labels,
        "counts": counts,
        "fraction_undersampled": counts["undersampled"] / total_px,
        "fraction_dark": counts["dark"] / total_px,
    }
