"""Photon path tracing and a photon map with indexed neighborhood queries.

Every photon carries an explicit RGB power (flux).  At dielectric events the
Fresnel reflectance drives a *random* reflection/refraction choice and the
power is folded by the transported fraction (F on reflection, 1-F on
transmission), which keeps every path energy-conserving per interaction and
avoids the explosive weights of divide-by-probability sampling.  Direct
photons (first diffuse hit) and caustic photons (diffuse hit after >=1
specular event) are stored in separate maps, which lets direct lighting and
the caustic estimator be combined without double-counting.
"""
import math
import random

from . import vec3
from .materials import fresnel_schlick, refract
from .scene import AIR_IOR, RECEIVER_IDS
from .spatial import (BOUNDARY_RULE, DEFAULT_BACKEND, DEFAULT_QUERY_MODE,
                      INDEX_VERSION, NEIGHBORHOOD_INDEX_MIGRATION,
                      KdPhotonIndex, make_index)

EPS = 1.0e-4
MAX_BOUNCES = 24
RR_SURVIVE = 0.7


class Photon:
    __slots__ = ("pos", "direction", "power", "surface_id")

    def __init__(self, pos, direction, power, surface_id):
        self.pos = pos
        self.direction = direction
        self.power = power
        self.surface_id = surface_id


# ``PhotonMap`` keeps its historic name and constructor so existing call sites
# and tests keep working; it now resolves to the migrated kd-tree index.  The
# pre-migration hash grid is still reachable as ``HashPhotonIndex`` (pass
# ``index_backend="hash_grid"`` to trace_photons) for side-by-side comparison.
PhotonMap = KdPhotonIndex


def _nearest_hit(primitives, o, d, tmax=1.0e6):
    best = None
    for prim in primitives:
        hit = prim.intersect(o, d, EPS, tmax)
        if hit is not None and (best is None or hit.t < best.t):
            best = hit
    return best


def _cosine_hemisphere(n, rng):
    r = math.sqrt(rng.random())
    theta = 2.0 * math.pi * rng.random()
    ux = r * math.cos(theta)
    uy = r * math.sin(theta)
    uz = math.sqrt(max(0.0, 1.0 - ux * ux - uy * uy))
    if abs(n[0]) > 0.9:
        t1 = (0.0, 1.0, 0.0)
    else:
        t1 = (1.0, 0.0, 0.0)
    t1 = vec3.normalize(vec3.sub(t1, vec3.mul(n, vec3.dot(t1, n))))
    t2 = vec3.cross(n, t1)
    return vec3.normalize((
        t1[0] * ux + t2[0] * uy + n[0] * uz,
        t1[1] * ux + t2[1] * uy + n[1] * uz,
        t1[2] * ux + t2[2] * uy + n[2] * uz,
    ))


def trace_photons(scene, n_photons, beam_power=1.0, seed=12345,
                  grid_cell=0.08, index_backend=DEFAULT_BACKEND,
                  index_query_mode=DEFAULT_QUERY_MODE):
    """Emit `n_photons` beam photons; return direct and caustic photon maps.

    ``index_backend`` selects the neighborhood index: ``"kdtree"`` (migrated
    default), ``"hash_grid"`` (legacy) or ``"brute_force"`` (audit oracle).
    """
    rng = random.Random(seed)
    prims = scene["primitives"]
    glass = scene["glass"]
    from .scene import BEAM_HX, BEAM_HZ, EMIT_Y

    direct = make_index(index_backend, grid_cell, query_mode=index_query_mode)
    caustic = make_index(index_backend, grid_cell, query_mode=index_query_mode)
    w = (beam_power / n_photons,) * 3

    emitted_energy = [0.0, 0.0, 0.0]
    stored_energy = [0.0, 0.0, 0.0]   # sum of power at every diffuse storage
    absorbed_energy = [0.0, 0.0, 0.0]  # final fate: energy removed from path
    escaped_energy = [0.0, 0.0, 0.0]   # final fate: rays that leave the scene
    alive_energy = [0.0, 0.0, 0.0]     # final fate: cut by bounce limit

    for _ in range(n_photons):
        ox = (2.0 * rng.random() - 1.0) * BEAM_HX
        oz = (2.0 * rng.random() - 1.0) * BEAM_HZ
        o = (ox, EMIT_Y, oz)
        d = (0.0, -1.0, 0.0)
        power = list(w)
        inside = False
        specular_bounces = 0
        terminated = False

        for _bounce in range(MAX_BOUNCES):
            hit = _nearest_hit(prims, o, d)
            if hit is None:
                for k in range(3):
                    escaped_energy[k] += power[k]
                terminated = True
                break
            mat = hit.prim.material
            if mat.kind == "dielectric":
                n = hit.normal
                if vec3.dot(d, n) > 0.0:
                    n = vec3.mul(n, -1.0)  # n now points against the ray
                ior_from = glass.ior if inside else AIR_IOR
                ior_to = AIR_IOR if inside else glass.ior
                cos_i = -vec3.dot(d, n)
                fr = min(1.0, max(0.0,
                                  fresnel_schlick(cos_i, ior_from, ior_to)))
                # Weight folding: the chosen branch keeps the physically
                # transported fraction, so power never explodes on rare
                # events and every path conserves energy per interaction.
                if rng.random() < fr:
                    d = vec3.reflect(d, n)
                    # The transmitted (1-F) branch is not followed; book it.
                    for k in range(3):
                        absorbed_energy[k] += power[k] * (1.0 - fr)
                    power = [p * fr for p in power]
                else:
                    td = refract(d, n, ior_from, ior_to)
                    if td is None:  # total internal reflection: keep all
                        d = vec3.reflect(d, n)
                    else:
                        d = td
                        inside = not inside
                        # The reflected F branch is not followed; book it.
                        for k in range(3):
                            absorbed_energy[k] += power[k] * fr
                        power = [p * (1.0 - fr) for p in power]
                specular_bounces += 1
                o = hit.point
                continue

            # Diffuse hit: store a photon.  On the receiver the surface
            # reflects albedo and absorbs (1-albedo); only that loss is
            # booked here, the reflected power continues the path.  The
            # ground (and the bounce-limit case) absorbs everything.
            target = caustic if specular_bounces > 0 else direct
            target.add(Photon(hit.point, d, tuple(power), hit.surface_id))
            for k in range(3):
                stored_energy[k] += power[k]

            albedo = mat.albedo
            if hit.surface_id not in RECEIVER_IDS or _bounce >= MAX_BOUNCES - 1:
                for k in range(3):
                    absorbed_energy[k] += power[k]
                terminated = True
                break
            for k in range(3):
                absorbed_energy[k] += power[k] * (1.0 - albedo[k])
            if rng.random() >= RR_SURVIVE:
                # Russian roulette ends the path; its remaining carried power
                # is then considered absorbed at this event.
                for k in range(3):
                    absorbed_energy[k] += power[k] * albedo[k]
                terminated = True
                break
            power = [power[k] * albedo[k] for k in range(3)]
            d = _cosine_hemisphere(hit.normal, rng)
            o = hit.point

        if not terminated:
            # Bounce limit reached while still bouncing (e.g. trapped in glass).
            for k in range(3):
                alive_energy[k] += power[k]

        for k in range(3):
            emitted_energy[k] += w[k]

    return {
        "direct": direct,
        "caustic": caustic,
        "emitted_energy": emitted_energy,
        "stored_energy": stored_energy,
        "absorbed_energy": absorbed_energy,
        "escaped_energy": escaped_energy,
        "alive_energy": alive_energy,
        "per_photon_power": w,
        "n_photons": n_photons,
        "index": {
            "backend": index_backend,
            "query_mode": (index_query_mode if index_backend == "kdtree"
                           else "bucket_refilter"),
            "version": INDEX_VERSION,
            "migration": NEIGHBORHOOD_INDEX_MIGRATION,
            "boundary_rule": BOUNDARY_RULE,
        },
    }
