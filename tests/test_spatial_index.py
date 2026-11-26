"""Neighborhood-index migration tests.

Pins the contract that the old hash grid and the migrated kd-tree must both
honour: exact inclusive-radius membership, per-surface isolation, agreement
with a brute-force oracle (analytic radius shells), the boundary rule, and
that the estimator's *effective sample* (irradiance / n_eff / radius) is
unchanged by the migration rather than merely "close enough".

Run:  python3 -m unittest discover -s tests -v
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caustics import run_experiment
from caustics.estimator import estimate
from caustics.photon import Photon, trace_photons
from caustics.scene import BACK_ID, FRONT_ID, build_scene
from caustics.spatial import (BOUNDARY_RULE, DEFAULT_BACKEND, INDEX_VERSION,
                              NEIGHBORHOOD_INDEX_MIGRATION, BruteForceIndex,
                              HashPhotonIndex, KdPhotonIndex, make_index)

DOWN = (0.0, -1.0, 0.0)


def _fill(index, photons):
    for ph in photons:
        index.add(ph)
    return index


def _all_indexes(photons, cell=0.08):
    return {
        "brute_force": _fill(BruteForceIndex(), photons),
        "hash_grid": _fill(HashPhotonIndex(cell), photons),
        "kdtree:sphere": _fill(KdPhotonIndex(cell, "sphere"), photons),
        "kdtree:bucket": _fill(KdPhotonIndex(cell, "bucket"), photons),
    }


def _signature(result):
    return frozenset(id(p) for p in result)


class TestExactnessAgainstBruteForce(unittest.TestCase):
    def _cloud(self, n=400, seed=1):
        rng = random.Random(seed)
        photons = []
        for _ in range(n):
            rid = FRONT_ID if rng.random() < 0.6 else BACK_ID
            photons.append(Photon((rng.uniform(-2, 2), 0.6, rng.uniform(-2, 2)),
                                  DOWN, (1.0, 1.0, 1.0), rid))
        return photons

    def test_random_queries_match_oracle_all_indexes(self):
        photons = self._cloud()
        indexes = _all_indexes(photons)
        rng = random.Random(7)
        for _ in range(300):
            q = (rng.uniform(-2.4, 2.4), 0.6, rng.uniform(-2.4, 2.4))
            rad = rng.choice((0.02, 0.1, 0.3, 0.75))
            sid = FRONT_ID if rng.random() < 0.5 else BACK_ID
            ref = _signature(indexes["brute_force"].neighborhood(q, rad, sid))
            for label, idx in indexes.items():
                self.assertEqual(_signature(idx.neighborhood(q, rad, sid)), ref,
                                 msg=f"{label} diverged at {q} r={rad}")

    def test_analytic_radius_shell(self):
        # Photons exactly on a ring of radius R about the origin, plus one
        # just outside.  All four indices must return *exactly* the on-shell
        # photons (inclusive boundary) and never the exterior one.
        R = 0.5
        pts = []
        for k in range(8):
            a = 2.0 * math.pi * k / 8
            pts.append((R * math.cos(a), 0.0, R * math.sin(a)))
        outside = (R + 1e-6, 0.0, 0.0)
        photons = [Photon(p, DOWN, (1.0, 1.0, 1.0), FRONT_ID) for p in pts]
        photons.append(Photon(outside, DOWN, (1.0, 1.0, 1.0), FRONT_ID))
        indexes = _all_indexes(photons)
        on_shell = {id(p) for p in photons[:-1]}
        for label, idx in indexes.items():
            got = {id(p) for p in idx.neighborhood((0.0, 0.0, 0.0), R, FRONT_ID)}
            self.assertEqual(got, on_shell, msg=f"{label} boundary rule broken")

    def test_coincident_photons_all_returned(self):
        # Several photons share one position; every index must return all of
        # them (no dedup, no single-hit shortcuts).
        p = (0.123, 0.6, -0.456)
        photons = [Photon(p, DOWN, (1.0, 1.0, 1.0), FRONT_ID) for _ in range(5)]
        indexes = _all_indexes(photons)
        for label, idx in indexes.items():
            got = idx.neighborhood(p, 1e-9, FRONT_ID)
            self.assertEqual(len(got), 5, msg=f"{label} dropped coincident")

    def test_excluded_just_outside_each_axis(self):
        R = 0.3
        photons, inside_ids, outside_ids = [], set(), set()
        for axis in range(3):
            d = [0.0, 0.0, 0.0]
            d[axis] = R
            ph = Photon(tuple(d), DOWN, (1.0,) * 3, FRONT_ID)
            photons.append(ph)
            inside_ids.add(id(ph))
            d2 = [0.0, 0.0, 0.0]
            d2[axis] = R * (1 + 1e-6)
            ph2 = Photon(tuple(d2), DOWN, (1.0,) * 3, FRONT_ID)
            photons.append(ph2)
            outside_ids.add(id(ph2))
        indexes = _all_indexes(photons)
        for label, idx in indexes.items():
            got = {id(p) for p in idx.neighborhood((0.0, 0.0, 0.0), R, FRONT_ID)}
            self.assertEqual(got, inside_ids, msg=f"{label} axis boundary")
            self.assertFalse(got & outside_ids)


class TestSurfaceIsolation(unittest.TestCase):
    def test_cross_surface_never_borrowed(self):
        photons = [Photon((0.0, 0.6, 0.0), DOWN, (1.0,) * 3, FRONT_ID),
                   Photon((0.0, 0.4, 0.0), DOWN, (1.0,) * 3, BACK_ID)]
        indexes = _all_indexes(photons)
        # Radius 0.1: the back photon (0.2 away) is out of range of the front
        # query point, so surface isolation must report zero for BACK_ID.
        for label, idx in indexes.items():
            self.assertEqual(len(idx.neighborhood((0.0, 0.6, 0.0), 0.1,
                                                  FRONT_ID)), 1, label)
            self.assertEqual(len(idx.neighborhood((0.0, 0.4, 0.0), 0.1,
                                                  BACK_ID)), 1, label)
            self.assertEqual(len(idx.neighborhood((0.0, 0.6, 0.0), 0.1,
                                                  BACK_ID)), 0, label)


class TestPhotonMapAliasCompatibility(unittest.TestCase):
    def test_photon_map_points_at_migrated_index(self):
        from caustics.photon import PhotonMap
        self.assertIs(PhotonMap, KdPhotonIndex)
        self.assertEqual(DEFAULT_BACKEND, "kdtree")

    def test_make_index_legacy_backend(self):
        legacy = make_index("hash_grid", 0.1)
        self.assertIsInstance(legacy, HashPhotonIndex)
        new = make_index("kdtree", 0.1)
        self.assertIsInstance(new, KdPhotonIndex)


class TestEstimatorParity(unittest.TestCase):
    """The estimator's effective sample must match across the migration.

    The migrated and legacy indices return the *same photon set* per query,
    but not necessarily in the same order, so the irradiance sum can differ by
    float rounding (~1e-16 relative).  These tests measure that deviation and
    bound it; they do not silently treat "approximately equal" as identical.
    """

    REL_TOL = 1e-12

    def _render_views(self, backend):
        tracer = trace_photons(build_scene(), 20000, seed=3,
                               index_backend=backend)
        from caustics.renderer import render
        return render(tracer, 0.1, 0.5, "epanechnikov", True)

    def _max_rel_deviation(self, a, b):
        worst = 0.0
        for row_a, row_b in zip(a, b):
            for pa, pb in zip(row_a, row_b):
                for ca, cb in zip(pa, pb):
                    denom = max(abs(ca), abs(cb), 1e-300)
                    worst = max(worst, abs(ca - cb) / denom)
        return worst

    def test_per_pixel_n_eff_and_radius_identical(self):
        old = self._render_views("hash_grid")
        new = self._render_views("kdtree")
        for sid in (FRONT_ID, BACK_ID):
            # Discrete estimator outputs (counts, footprints) must be exactly
            # equal: these are not float-noise quantities.
            self.assertEqual(old[sid]["n_eff"], new[sid]["n_eff"], sid)
            self.assertEqual(old[sid]["radius"], new[sid]["radius"], sid)
            # Continuous irradiance: same photon set, different summation
            # order.  Measure the deviation and bound it tightly.
            worst_c = self._max_rel_deviation(old[sid]["caustic"],
                                              new[sid]["caustic"])
            worst_d = self._max_rel_deviation(old[sid]["direct"],
                                              new[sid]["direct"])
            self.assertLess(worst_c, self.REL_TOL,
                            msg=f"caustic rel dev {worst_c:.2e} on {sid}")
            self.assertLess(worst_d, self.REL_TOL,
                            msg=f"direct rel dev {worst_d:.2e} on {sid}")

    def test_estimate_reports_same_effective_sample(self):
        tracer = trace_photons(build_scene(), 8000, seed=11)
        old = _fill(HashPhotonIndex(0.08), tracer["caustic"].photons)
        new = _fill(KdPhotonIndex(0.08, "sphere"), tracer["caustic"].photons)
        normal = (0.0, 1.0, 0.0)
        for pt in ((0.0, 0.6, 0.0), (0.4, 0.6, -0.3), (1.0, 0.6, 1.0)):
            eo = estimate(old, pt, normal, FRONT_ID, 0.12, "epanechnikov", 0.5)
            en = estimate(new, pt, normal, FRONT_ID, 0.12, "epanechnikov", 0.5)
            # Same effective sample by construction.
            self.assertEqual(eo["n_eff"], en["n_eff"], pt)
            self.assertEqual(eo["radius"], en["radius"], pt)
            self.assertEqual(eo["undersampled"], en["undersampled"], pt)
            self.assertEqual(eo["budget_exhausted"], en["budget_exhausted"], pt)
            # And the neighborhood sets themselves are identical.
            self.assertEqual(
                {id(p) for p in old.neighborhood(pt, 0.5, FRONT_ID)},
                {id(p) for p in new.neighborhood(pt, 0.5, FRONT_ID)}, pt)
            for a, b in zip(eo["irradiance"], en["irradiance"]):
                denom = max(abs(a), abs(b), 1e-300)
                self.assertLess(abs(a - b) / denom, self.REL_TOL, pt)


class TestMigrationMetadata(unittest.TestCase):
    def test_tracer_reports_version_and_boundary(self):
        tracer = trace_photons(build_scene(), 2000, seed=1)
        meta = tracer["index"]
        self.assertEqual(meta["version"], INDEX_VERSION)
        self.assertEqual(meta["migration"], NEIGHBORHOOD_INDEX_MIGRATION)
        self.assertEqual(meta["backend"], "kdtree")
        self.assertEqual(meta["query_mode"], "sphere")
        self.assertEqual(meta["boundary_rule"], BOUNDARY_RULE)
        self.assertIn("inclusive", BOUNDARY_RULE)

    def test_legacy_backend_still_selectable_and_matches(self):
        a = trace_photons(build_scene(), 3000, seed=5,
                          index_backend="hash_grid")
        b = trace_photons(build_scene(), 3000, seed=5,
                          index_backend="kdtree")
        self.assertEqual(a["caustic"].count(), b["caustic"].count())
        self.assertEqual(a["direct"].count(), b["direct"].count())
        self.assertEqual(a["index"]["backend"], "hash_grid")
        self.assertEqual(b["index"]["backend"], "kdtree")


class TestRunExperimentThreading(unittest.TestCase):
    def test_index_backend_plumbed_through(self):
        r = run_experiment(n_photons=4000, radius=0.15, max_radius=0.5,
                           index_backend="kdtree")
        self.assertEqual(r["tracer_summary"]["index"]["backend"], "kdtree")
        r2 = run_experiment(n_photons=4000, radius=0.15, max_radius=0.5,
                            index_backend="hash_grid")
        self.assertEqual(r2["tracer_summary"]["index"]["backend"], "hash_grid")


if __name__ == "__main__":
    unittest.main()


class TestDiagnostics(unittest.TestCase):
    """Spot / estimation anomaly semantics used by the UI status and CLI."""

    def _payload(self, n=6000):
        r = run_experiment(n_photons=n, radius=0.12, max_radius=0.45)
        return r

    def test_low_budget_flags_estimation_anomaly(self):
        r = self._payload(n=1000)
        codes = {i["code"] for i in r["diagnostics"]["issues"]}
        self.assertTrue(codes & {"undersampled_warn", "undersampled_high",
                                 "budget_exhausted"})
        self.assertNotEqual(r["diagnostics"]["status"], "ok")

    def test_high_budget_is_clean(self):
        r = self._payload(n=40000)
        self.assertEqual(r["diagnostics"]["status"], "ok")
        self.assertEqual(r["diagnostics"]["count"], 0)

    def test_healthy_render_has_no_spot_anomaly(self):
        r = self._payload(n=20000)
        spot_issues = [i for i in r["diagnostics"]["issues"]
                       if i["area"] == "spot"]
        self.assertEqual(spot_issues, [])

    def test_issue_shape(self):
        r = self._payload(n=1000)
        for it in r["diagnostics"]["issues"]:
            self.assertIn(it["severity"], ("warn", "bad"))
            self.assertIn(it["area"], ("spot", "estimate"))
            self.assertIsInstance(it["code"], str)
            self.assertIsInstance(it["message"], str)
