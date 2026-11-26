"""Pure-stdlib photon mapping renderer for focused (caustic) light spots."""
from .experiments import convergence, run_experiment
from .photon import trace_photons
from .renderer import render
from .scene import build_scene

__all__ = [
    "build_scene",
    "trace_photons",
    "render",
    "run_experiment",
    "convergence",
]
