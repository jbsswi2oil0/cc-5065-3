"""Stdlib-only HTTP application for the photon-estimation experiment.

Endpoints
  GET  /                 static single-page UI
  POST /api/render       trace photons + density-estimate both receiver faces
  POST /api/convergence  render the same scene across several photon budgets
  GET  /api/defaults     default control values

The render response carries, per face, the direct/caustic/total image arrays,
the per-pixel effective photon count, footprint (radius) and the refiner's
dark/undersampled masks, plus global energy bookkeeping.  The UI exposes the
photon budget and kernel controls and feeds the estimator's failure flags
(undersampled / budget-exhausted) back as overlay warnings.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .experiments import (convergence, caustic_irradiance, peak_spot,
                          smoothness_scale, total_irradiance, _summary)
from .photon import trace_photons
from .refine import refinement_stats
from .renderer import render
from .scene import build_scene

DEFAULT_BUDGET = 40000
MAX_BUDGET = 300000
DEFAULTS = {
    "budget": DEFAULT_BUDGET,
    "maxBudget": MAX_BUDGET,
    "paneX": 0.0,
    "paneZ": 0.0,
    "radius": 0.12,
    "maxRadius": 0.45,
    "kernel": "epanechnikov",
    "adaptive": True,
    "seed": 12345,
}


def _numeric(body, key, default, lo=None, hi=None):
    val = body.get(key, default)
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def _photon_points(photon_map, surface_id, cap=4000):
    count = 0
    for ph in photon_map.photons:
        if ph.surface_id == surface_id:
            count += 1
    stride = max(1, count // cap)
    pts = []
    seen = 0
    for ph in photon_map.photons:
        if ph.surface_id != surface_id:
            continue
        if seen % stride == 0:
            pts.append([round(ph.pos[0], 4), round(ph.pos[2], 4)])
        seen += 1
        if len(pts) >= cap:
            break
    return pts


def build_payload(body):
    budget = int(_numeric(body, "budget", DEFAULT_BUDGET, 200, MAX_BUDGET))
    pane_x = _numeric(body, "paneX", 0.0, -2.0, 2.0)
    pane_z = _numeric(body, "paneZ", 0.0, -2.0, 2.0)
    radius = _numeric(body, "radius", 0.12, 0.02, 1.0)
    max_radius = _numeric(body, "maxRadius", 0.45, radius, 1.5)
    kernel = body.get("kernel", "epanechnikov")
    if kernel not in ("box", "epanechnikov", "gaussian"):
        kernel = "epanechnikov"
    adaptive = bool(body.get("adaptive", True))
    seed = int(_numeric(body, "seed", 12345, 1, 10 ** 9))

    scene = build_scene(pane_x, pane_z)
    tracer = trace_photons(scene, budget, seed=seed)
    views_raw = render(tracer, radius, max_radius, kernel, adaptive)
    summary = _summary(tracer)

    views = {}
    for sid, view in views_raw.items():
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
        views[sid] = {
            "gridN": view["grid_n"],
            "viewHx": view["view_hx"],
            "viewHz": view["view_hz"],
            "direct": view["direct"],
            "caustic": view["caustic"],
            "total": view["total"],
            "nEff": view["n_eff"],
            "radius": view["radius"],
            "undersampled": view["undersampled"],
            "budgetExhausted": view["budget_exhausted"],
            "labels": stats["labels"],
            "refine": view["refine"],
            "metrics": view["metrics"],
            "photonCount": view["photon_count"],
            "directCount": view["direct_count"],
            "causticPoints": _photon_points(tracer["caustic"], sid),
        }
    return {"views": views, "summary": summary,
            "params": {"budget": budget, "paneX": pane_x, "paneZ": pane_z,
                       "radius": radius, "maxRadius": max_radius,
                       "kernel": kernel, "adaptive": adaptive, "seed": seed}}


def build_convergence(body):
    raw = body.get("budgets")
    if not isinstance(raw, list) or not raw:
        raw = [5000, 20000, 80000]
    budgets = []
    for b in raw[:6]:
        try:
            b = int(float(b))
        except (TypeError, ValueError):
            continue
        if 200 <= b <= MAX_BUDGET:
            budgets.append(b)
    if not budgets:
        budgets = [5000, 20000, 80000]
    rows = convergence(
        budgets,
        pane_x=_numeric(body, "paneX", 0.0, -2.0, 2.0),
        pane_z=_numeric(body, "paneZ", 0.0, -2.0, 2.0),
        radius=_numeric(body, "radius", 0.12, 0.02, 1.0),
        max_radius=_numeric(body, "maxRadius", 0.45, 0.02, 1.5),
        kernel=body.get("kernel", "epanechnikov"),
        adaptive=bool(body.get("adaptive", True)),
        seed=int(_numeric(body, "seed", 12345, 1, 10 ** 9)),
    )
    return {"rows": rows, "budgets": budgets}


class Handler(BaseHTTPRequestHandler):
    server_version = "PhotonEstimator/1.0"

    def log_message(self, fmt, *args):  # quiet by default
        pass

    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(
            payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/defaults":
            self._send(200, DEFAULTS)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid JSON body"})
            return
        try:
            if self.path == "/api/render":
                self._send(200, build_payload(body))
            elif self.path == "/api/convergence":
                self._send(200, build_convergence(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # surface failures to the UI
            self._send(500, {"error": type(exc).__name__,
                             "detail": str(exc)})


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


# HTML is injected at import time to keep the server a single module.
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "..", "web", "index.html"), "rb") as fh:
    INDEX_HTML = fh.read()
