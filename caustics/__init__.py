"""Pure-stdlib photon mapping renderer for focused (caustic) light spots."""
from .experiments import convergence, run_experiment
from .photon import trace_photons
from .renderer import render
from .scene import build_scene
from .spatial import (BOUNDARY_RULE, INDEX_VERSION,
                      NEIGHBORHOOD_INDEX_MIGRATION, make_index)

__all__ = [
    "build_scene",
    "trace_photons",
    "render",
    "run_experiment",
    "convergence",
    "make_index",
    "INDEX_VERSION",
    "NEIGHBORHOOD_INDEX_MIGRATION",
    "BOUNDARY_RULE",
]
