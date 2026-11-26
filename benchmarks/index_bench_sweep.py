#!/usr/bin/env python3
"""Robust query-strategy sweep for the photon-index migration decision.

Runs the renderer's actual query points against each candidate index over a
grid of photon budgets, radii and seeds, verifying exactness against a
brute-force oracle and reporting aggregate wall-clock time.  This backs the
``DEFAULT_QUERY_MODE`` choice with measurements rather than a single sample.

Run:
    python3 benchmarks/index_bench_sweep.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caustics.photon import trace_photons
from caustics.renderer import GRID_N
from caustics.scene import FRONT_ID, build_scene
from caustics.spatial import (BruteForceIndex, HashPhotonIndex, KdPhotonIndex)

BUDGETS = (10000, 40000, 160000)
RADII = (0.06, 0.12, 0.25, 0.45)
SEEDS = (12345, 7, 99)
QUERY_STRIDE = 6  # sample every 6th grid point -> 25x25 = 625 queries / radius


def _queries():
    n = GRID_N
    pts = []
    for iy in range(0, n, QUERY_STRIDE):
        z = -2.6 + 5.2 * (iy + 0.5) / n
        for ix in range(0, n, QUERY_STRIDE):
            x = -2.6 + 5.2 * (ix + 0.5) / n
            pts.append((x, 0.6, z))
    return pts


def _build(cls, photons, mode=None):
    if mode:
        index = KdPhotonIndex(0.08, query_mode=mode)
    elif cls is BruteForceIndex:
        index = BruteForceIndex()
    else:
        index = cls(0.08)
    for ph in photons:
        index.add(ph)
    return index


def _time_queries(index, pts, radii):
    t0 = time.perf_counter()
    n = 0
    sigs = []
    for rad in radii:
        for p in pts:
            res = index.neighborhood(p, rad, FRONT_ID)
            n += len(res)
            sigs.append(frozenset(id(q) for q in res))
    return time.perf_counter() - t0, n, tuple(sigs)


def main():
    pts = _queries()
    totals = {"hash_grid": [], "kdtree:sphere": [], "kdtree:bucket": []}
    mismatches = 0

    for seed in SEEDS:
        for budget in BUDGETS:
            tracer = trace_photons(build_scene(), budget, seed=seed)
            photons = [ph for ph in tracer["caustic"].photons
                       if ph.surface_id == FRONT_ID]

            oracle = _build(BruteForceIndex, photons)
            _t, _n, ref = _time_queries(oracle, pts, RADII)

            for label, cls, mode in (
                ("hash_grid", HashPhotonIndex, None),
                ("kdtree:sphere", KdPhotonIndex, "sphere"),
                ("kdtree:bucket", KdPhotonIndex, "bucket"),
            ):
                idx = _build(cls, photons, mode)
                dt, _cnt, sigs = _time_queries(idx, pts, RADII)
                totals[label].append(dt)
                if sigs != ref:
                    mismatches += 1
                    print(f"  MISMATCH {label} seed={seed} N={budget}",
                          file=sys.stderr)

    print(f"budgets={BUDGETS} radii={RADII} seeds={SEEDS} "
          f"queries/cfg={len(pts) * len(RADII)}")
    print(f"{'strategy':<16}{'mean_s':>9}{'median_s':>10}{'max_s':>9}")
    for label, xs in totals.items():
        print(f"{label:<16}{statistics.mean(xs):>9.4f}"
              f"{statistics.median(xs):>10.4f}{max(xs):>9.4f}")
    if mismatches:
        print(f"\n{mismatches} exactness mismatches vs brute force!", file=sys.stderr)
        return 1
    sm = statistics.mean(totals["kdtree:sphere"])
    bm = statistics.mean(totals["kdtree:bucket"])
    print(f"\nkdtree sphere vs bucket mean: {sm:.4f}s vs {bm:.4f}s -> "
          f"{'sphere' if sm <= bm else 'bucket'} wins")
    print("All index/mode combinations matched the brute-force oracle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
