#!/usr/bin/env python3
"""Empirical benchmark: pick the kd-tree query strategy on real data.

The migration brief requires the exact-radius ("sphere") strategy vs. the
spatial-bucket coarse query + distance re-filter ("bucket") decision to be
made from measurement, not assumption.  This script traces the *same* photon
cloud (same seed) once, then issues the renderer's real per-pixel queries
against both kd-tree query modes and the legacy hash grid, checking:

  * set equality against a brute-force oracle (exactness is a precondition,
    not a performance trade-off), and
  * wall-clock time and candidate count (the actual cost).

Each photon map is indexed independently, exactly as the renderer queries
direct and caustic photons separately.

Run:
    python3 benchmarks/index_bench.py [--photons 40000] [--seed 12345]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caustics.photon import trace_photons
from caustics.renderer import _grid_points
from caustics.scene import RECEIVER_IDS, build_scene
from caustics.spatial import (BruteForceIndex, HashPhotonIndex, KdPhotonIndex)


def _queries():
    """A representative slice of the (point, radius, surface_id) queries."""
    radii = (0.06, 0.12, 0.25)
    out = []
    for sid in RECEIVER_IDS:
        grid = _grid_points(sid)
        for rad in radii:
            for row in grid[::5]:
                for point, _normal in row[::5]:
                    out.append((point, rad, sid))
    return out


def _signature(photons):
    return frozenset(id(p) for p in photons)


def _fill(index, photons):
    for ph in photons:
        index.add(ph)
    return index


def _run(index, queries):
    t0 = time.perf_counter()
    total = 0
    sigs = []
    for point, rad, sid in queries:
        res = index.neighborhood(point, rad, sid)
        total += len(res)
        sigs.append(_signature(res))
    return time.perf_counter() - t0, total, sigs


def _bench_map(photons, queries, cell=0.08):
    strategies = {
        "brute_force": _fill(BruteForceIndex(), photons),
        "hash_grid": _fill(HashPhotonIndex(cell), photons),
        "kdtree:sphere": _fill(KdPhotonIndex(cell, query_mode="sphere"),
                               photons),
        "kdtree:bucket": _fill(KdPhotonIndex(cell, query_mode="bucket"),
                               photons),
    }
    out = {}
    oracle = None
    for label, index in strategies.items():
        dt, total, sigs = _run(index, queries)
        out[label] = (dt, total, sigs)
        if label == "brute_force":
            oracle = sigs
    return out, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    tracer = trace_photons(build_scene(), args.photons, seed=args.seed)
    queries = _queries()

    agg = {}
    for map_name in ("caustic", "direct"):
        metrics, oracle = _bench_map(tracer[map_name].photons, queries)
        print(f"\n[{map_name} map] queries={len(queries)}")
        print(f"  {'strategy':<16}{'time_s':>9}{'candidates':>12}"
              f"{'matches_brute':>15}")
        for label, (dt, total, sigs) in metrics.items():
            print(f"  {label:<16}{dt:>9.4f}{total:>12d}"
                  f"{str(sigs == oracle):>15}")
        for label in ("hash_grid", "kdtree:sphere", "kdtree:bucket"):
            if metrics[label][2] != oracle:
                print(f"  WARNING: {label} disagrees with brute force!",
                      file=sys.stderr)
                return 1
        for label, (dt, total, _s) in metrics.items():
            a = agg.setdefault(label, [0.0, 0])
            a[0] += dt
            a[1] += total

    ks, kb = agg["kdtree:sphere"][0], agg["kdtree:bucket"][0]
    winner = "sphere" if ks <= kb else "bucket"
    print("\nTotals across both maps:")
    for label, (dt, total) in agg.items():
        print(f"  {label:<16}{dt:>9.4f}s{total:>12d} candidates")
    print(f"\nFaster query strategy: {winner} "
          f"(sphere={ks:.4f}s, bucket={kb:.4f}s)")
    print("All strategies return the exact inclusive-radius set on this workload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
