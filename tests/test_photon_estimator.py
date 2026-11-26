"""Tests for kernel normalization, estimator area normalization, normal
compatibility, thin-wall leakage isolation, and energy conservation.

Run:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caustics import run_experiment
from caustics.estimator import estimate
from caustics.kernels import KERNELS, kernel_integral
from caustics.photon import Photon, PhotonMap
from caustics.scene import BACK_ID, FRONT_ID


class TestKernels(unittest.TestCase):
    def test_unit_area(self):
        for name in KERNELS:
            self.assertAlmostEqual(kernel_integral(name), 1.0, places=3)

    def test_zero_outside_support(self):
        for k in KERNELS.values():
            self.assertEqual(k(1.0001), 0.0)


def _flat_map(surface_id, n_side, span, power, direction=(0.0, -1.0, 0.0),
              y=0.6, cell=0.5):
    """A uniform n_side x n_side photon grid over a (2*span)^2 area."""
    pm = PhotonMap(cell)
    for iz in range(n_side):
        z = -span + 2.0 * span * (iz + 0.5) / n_side
        for ix in range(n_side):
            x = -span + 2.0 * span * (ix + 0.5) / n_side
            pm.add(Photon((x, y, z), direction, (power,) * 3, surface_id))
    return pm


class TestEstimatorNormalization(unittest.TestCase):
    def test_flat_field_density_is_phi_over_area(self):
        # Uniform 2D photon density over area A = (2*span)^2: estimated
        # irradiance must approach total_power/A regardless of N (no
        # 'more photons => brighter' normalization bug).
        span = 2.0
        area = (2.0 * span) ** 2
        normal = (0.0, 1.0, 0.0)
        readings = []
        for n_side in (16, 32, 64):
            n = n_side * n_side
            power = 1.0 / n
            pm = _flat_map(FRONT_ID, n_side, span, power)
            est = estimate(pm, (0.0, 0.6, 0.0), normal, FRONT_ID,
                           0.4, "epanechnikov", 0.4)
            readings.append(sum(est["irradiance"]) / 3.0)
        expected = 1.0 / area  # total power = N * (1/N) = 1
        for r in readings:
            self.assertAlmostEqual(r, expected, delta=0.02)
        # The three budgets must agree to within sampling noise.
        self.assertAlmostEqual(readings[0], readings[2], delta=0.01)

    def test_power_scaling_linear(self):
        pm1 = _flat_map(FRONT_ID, 32, 2.0, 1.0 / (32 * 32))
        pm2 = _flat_map(FRONT_ID, 32, 2.0, 4.0 / (32 * 32))
        normal = (0.0, 1.0, 0.0)
        e1 = estimate(pm1, (0.0, 0.6, 0.0), normal, FRONT_ID, 0.4,
                      "epanechnikov", 0.4)["irradiance"]
        e2 = estimate(pm2, (0.0, 0.6, 0.0), normal, FRONT_ID, 0.4,
                      "epanechnikov", 0.4)["irradiance"]
        for c in range(3):
            self.assertAlmostEqual(e2[c] / e1[c], 4.0, places=6)


class TestNormalCompatibility(unittest.TestCase):
    def test_grazing_and_wrong_normal_rejected(self):
        # Photon travelling upward (+y) is incompatible with a +y query
        # normal (it would be leaving the surface, not arriving).
        pm = PhotonMap(0.5)
        for i in range(50):
            pm.add(Photon((0.05 * (i - 25), 0.6, 0.0), (0.0, 1.0, 0.0),
                          (0.1,) * 3, FRONT_ID))
        normal = (0.0, 1.0, 0.0)
        est = estimate(pm, (0.0, 0.6, 0.0), normal, FRONT_ID, 1.5,
                       "epanechnikov", 1.5)
        self.assertEqual(est["n_eff"], 0)
        self.assertTrue(est["budget_exhausted"])


class TestWallLeakage(unittest.TestCase):
    def test_no_caustic_on_back_face(self):
        r = run_experiment(n_photons=8000, radius=0.15, max_radius=0.5)
        back = r["views"][BACK_ID]
        self.assertEqual(back["photon_count"], 0)
        self.assertEqual(back["metrics"]["caustic_irradiance"], 0.0)

    def test_surface_ids_are_separate_grids(self):
        pm = PhotonMap(0.2)
        pm.add(Photon((0.0, 0.6, 0.0), (0.0, -1.0, 0.0), (1.0,) * 3, FRONT_ID))
        front = pm.neighborhood((0.0, 0.6, 0.0), 0.5, FRONT_ID)
        back = pm.neighborhood((0.0, 0.4, 0.0), 0.5, BACK_ID)
        self.assertEqual(len(front), 1)
        self.assertEqual(len(back), 0)


class TestEnergyConservation(unittest.TestCase):
    def test_fate_bookkeeping_closes(self):
        r = run_experiment(n_photons=6000, radius=0.15, max_radius=0.5)
        s = r["tracer_summary"]
        self.assertAlmostEqual(s["bookkeeping_residual"], 0.0, places=6)

    def test_fixed_radius_energy_stable_with_budget(self):
        vals = []
        for n in (3000, 12000):
            r = run_experiment(n_photons=n, radius=0.2, max_radius=0.2,
                               adaptive=False)
            vals.append(r["views"][FRONT_ID]
                        ["metrics"]["caustic_irradiance"])
        self.assertAlmostEqual(vals[1] / vals[0], 1.0, delta=0.05)


class TestPaneMovement(unittest.TestCase):
    def test_spot_follows_pane(self):
        rl = run_experiment(pane_x=-1.0, n_photons=10000, radius=0.15,
                            max_radius=0.5, seed=7)
        rr = run_experiment(pane_x=1.0, n_photons=10000, radius=0.15,
                            max_radius=0.5, seed=7)
        xl = rl["views"][FRONT_ID]["metrics"]["spot"]["centroid"][0]
        xr = rr["views"][FRONT_ID]["metrics"]["spot"]["centroid"][0]
        self.assertLess(xl, -0.4)
        self.assertGreater(xr, 0.4)
        self.assertGreater(xr - xl, 1.0)


class TestFocusing(unittest.TestCase):
    def test_lens_concentrates_light_into_a_caustic(self):
        # Without any glass there can be no caustic photons; with the lens
        # the focused caustic peak must clearly exceed the diffuse direct
        # illumination level at the same (small) kernel radius.
        from caustics.photon import trace_photons
        from caustics.renderer import render
        from caustics.scene import build_scene

        no_glass = build_scene()
        no_glass["primitives"] = [p for p in no_glass["primitives"]
                                  if p.material.kind == "diffuse"]
        tr0 = trace_photons(no_glass, 20000, seed=5)
        self.assertEqual(tr0["caustic"].count(), 0)

        tr1 = trace_photons(build_scene(), 20000, seed=5)
        view = render(tr1, 0.06, 0.06, adaptive=False)["wall.front"]
        peak = lambda img: max(sum(px) / 3.0 for row in img for px in row)
        self.assertGreater(tr1["caustic"].count(), 2000)
        self.assertGreater(peak(view["caustic"]),
                           2.5 * peak(view["direct"]))


if __name__ == "__main__":
    unittest.main()
