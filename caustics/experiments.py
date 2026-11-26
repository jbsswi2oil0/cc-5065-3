"""Experiment harness: budgets, convergence, bias and energy bookkeeping."""
from .diagnostics import diagnose, summarize_issues
from .photon import trace_photons
from .refine import refinement_stats
from .renderer import render
from .scene import build_scene
from .spatial import DEFAULT_BACKEND, DEFAULT_QUERY_MODE


def total_irradiance(view):
    """Integrate irradiance over the sampled view area (power proxy)."""
    n = view["grid_n"]
    cell_area = (2.0 * view["view_hx"] / n) * (2.0 * view["view_hz"] / n)
    total = 0.0
    for row in view["total"]:
        for rgb in row:
            total += sum(rgb)
    return total * cell_area / 3.0


def caustic_irradiance(view):
    n = view["grid_n"]
    cell_area = (2.0 * view["view_hx"] / n) * (2.0 * view["view_hz"] / n)
    total = 0.0
    for row in view["caustic"]:
        for rgb in row:
            total += sum(rgb)
    return total * cell_area / 3.0


def peak_spot(view):
    """Centroid and peak of the caustic channel (focused spot position)."""
    n = view["grid_n"]
    sx = sz = sw = peak = 0.0
    peak_xy = (n // 2, n // 2)
    for iy in range(n):
        z = -view["view_hz"] + 2.0 * view["view_hz"] * (iy + 0.5) / n
        for ix in range(n):
            x = -view["view_hx"] + 2.0 * view["view_hx"] * (ix + 0.5) / n
            w = sum(view["caustic"][iy][ix])
            if w > peak:
                peak = w
                peak_xy = (ix, iy)
            sx += x * w
            sz += z * w
            sw += w
    if sw <= 1e-15:
        return {"centroid": (0.0, 0.0), "peak": 0.0, "peak_xy": peak_xy}
    return {
        "centroid": (sx / sw, sz / sw),
        "peak": peak,
        "peak_xy": peak_xy,
    }


def smoothness_scale(view):
    """Mean estimator footprint actually used on lit pixels (smoothing scale)."""
    radii, weights = [], []
    for iy, row in enumerate(view["radius"]):
        for ix, r in enumerate(row):
            e = sum(view["total"][iy][ix])
            if e > 1e-12:
                radii.append(r)
                weights.append(e)
    if not radii:
        return 0.0
    sw = sum(weights)
    return sum(r * w for r, w in zip(radii, weights)) / sw


def run_experiment(pane_x=0.0, pane_z=0.0, n_photons=40000, radius=0.12,
                   max_radius=0.45, kernel="epanechnikov", adaptive=True,
                   seed=12345, index_backend=DEFAULT_BACKEND,
                   index_query_mode=DEFAULT_QUERY_MODE):
    scene = build_scene(pane_x, pane_z)
    tracer = trace_photons(scene, n_photons, seed=seed,
                           index_backend=index_backend,
                           index_query_mode=index_query_mode)
    views = render(tracer, radius, max_radius, kernel, adaptive)
    result = {"views": views, "tracer_summary": _summary(tracer)}
    for sid, view in views.items():
        stats = refinement_stats(view)
        view["refine"] = {
            "counts": stats["counts"],
            "fraction_undersampled": stats["fraction_undersampled"],
            "fraction_dark": stats["fraction_dark"],
        }
        view["metrics"] = {
            "total_irradiance": total_irradiance(view),
            "caustic_irradiance": caustic_irradiance(view),
            "smoothness_scale": smoothness_scale(view),
            "spot": peak_spot(view),
        }
    result["diagnostics"] = summarize_issues(diagnose(
        views["wall.front"], views["wall.back"], result["tracer_summary"]))
    return result


def _summary(tracer):
    emitted = sum(tracer["emitted_energy"])
    stored = sum(tracer["stored_energy"])
    absorbed = sum(tracer["absorbed_energy"])
    escaped = sum(tracer["escaped_energy"])
    alive = sum(tracer["alive_energy"])
    # Final-fate conservation: every emitted photon ends absorbed, escaped,
    # or still alive at the bounce limit.
    residual = (emitted - absorbed - escaped - alive) / (emitted + 1e-30)
    return {
        "n_photons": tracer["n_photons"],
        "per_photon_power": tracer["per_photon_power"],
        "emitted_energy": tracer["emitted_energy"],
        "stored_energy": tracer["stored_energy"],
        "absorbed_energy": tracer["absorbed_energy"],
        "escaped_energy": tracer["escaped_energy"],
        "alive_energy": tracer["alive_energy"],
        "stored_fraction": stored / (emitted + 1e-30),
        "bookkeeping_residual": residual,
        "direct_photons": tracer["direct"].count(),
        "caustic_photons": tracer["caustic"].count(),
        "caustic_by_surface": {
            "wall.front": tracer["caustic"].count("wall.front"),
            "wall.back": tracer["caustic"].count("wall.back"),
        },
        "index": tracer["index"],
    }


def convergence(budgets, **kwargs):
    """Render the same scene at several photon budgets.

    Correctly normalized estimators keep roughly constant mean energy while
    variance (pixel-to-pixel noise in lit areas) falls ~ 1/sqrt(N).
    """
    rows = []
    for n in budgets:
        r = run_experiment(n_photons=n, **kwargs)
        front = r["views"]["wall.front"]
        rows.append({
            "n_photons": n,
            "front_total_energy": front["metrics"]["total_irradiance"],
            "front_caustic_energy": front["metrics"]["caustic_irradiance"],
            "front_peak": front["metrics"]["spot"]["peak"],
            "smoothness_scale": front["metrics"]["smoothness_scale"],
            "undersampled_fraction": front["refine"]["fraction_undersampled"],
            "bookkeeping_residual": r["tracer_summary"]["bookkeeping_residual"],
            "caustic_back": r["tracer_summary"]["caustic_by_surface"]["wall.back"],
        })
    base = rows[0]["front_caustic_energy"]
    for row in rows:
        row["caustic_energy_ratio"] = (row["front_caustic_energy"] / base
                                       if base > 0 else 0.0)
    return rows
