"""Render the two receiver faces by querying the photon maps.

Direct illumination comes from the *direct* photon map (photons whose first
hit was diffuse) and the caustic from the *caustic* map (photons that
underwent a specular event first).  They are stored in independent maps and
added exactly once per pixel, so a light path is never counted twice.
"""
from .estimator import estimate
from .scene import (BACK_ID, FRONT_ID, RECEIVER_IDS, WALL_BOTTOM, WALL_HX,
                    WALL_HZ, WALL_TOP)

GRID_N = 150
VIEW_HX = 2.6
VIEW_HZ = 2.6


def _grid_points(surface_id):
    n = GRID_N
    if surface_id == FRONT_ID:
        # Geometric outward normal of the box top face; the estimator tests
        # dot(-photon.direction, normal), which is positive for downward rays.
        y, normal = WALL_TOP, (0.0, 1.0, 0.0)
    else:
        y, normal = WALL_BOTTOM, (0.0, -1.0, 0.0)
    pts = []
    for iy in range(n):
        z = -VIEW_HZ + 2.0 * VIEW_HZ * (iy + 0.5) / n
        row = []
        for ix in range(n):
            x = -VIEW_HX + 2.0 * VIEW_HX * (ix + 0.5) / n
            row.append(((x, y, z), normal))
        pts.append(row)
    return pts


def render(tracer, radius=0.12, max_radius=0.45, kernel="epanechnikov",
           adaptive=True):
    direct_map = tracer["direct"]
    caustic_map = tracer["caustic"]
    cap = max_radius if adaptive else None
    views = {}

    for surface_id in RECEIVER_IDS:
        grid = _grid_points(surface_id)
        n = GRID_N
        direct_img = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
        caustic_img = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
        total_img = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
        n_eff_map = [[0 for _ in range(n)] for _ in range(n)]
        radius_map = [[0.0 for _ in range(n)] for _ in range(n)]
        under_map = [[False for _ in range(n)] for _ in range(n)]
        exhaust_map = [[False for _ in range(n)] for _ in range(n)]

        for iy, row in enumerate(grid):
            for ix, (point, normal) in enumerate(row):
                dr = estimate(direct_map, point, normal, surface_id,
                              radius, kernel, cap)
                cr = estimate(caustic_map, point, normal, surface_id,
                              radius, kernel, cap)
                d_e = dr["irradiance"]
                c_e = cr["irradiance"]
                direct_img[iy][ix] = list(d_e)
                caustic_img[iy][ix] = list(c_e)
                total_img[iy][ix] = [d_e[k] + c_e[k] for k in range(3)]
                # The caustic channel drives refinement decisions.
                n_eff_map[iy][ix] = cr["n_eff"]
                radius_map[iy][ix] = cr["radius"]
                under_map[iy][ix] = cr["undersampled"]
                exhaust_map[iy][ix] = cr["budget_exhausted"]

        views[surface_id] = {
            "grid_n": n,
            "view_hx": VIEW_HX,
            "view_hz": VIEW_HZ,
            "direct": direct_img,
            "caustic": caustic_img,
            "total": total_img,
            "n_eff": n_eff_map,
            "radius": radius_map,
            "undersampled": under_map,
            "budget_exhausted": exhaust_map,
            "photon_count": caustic_map.count(surface_id),
            "direct_count": direct_map.count(surface_id),
        }
    return views
