"""Geometric primitives with ray intersection and per-face surface ids."""
from math import sqrt

from . import vec3


class Hit:
    __slots__ = ("t", "point", "normal", "prim", "surface_id")

    def __init__(self, t, point, normal, prim, surface_id):
        self.t = t
        self.point = point
        self.normal = normal
        self.prim = prim
        self.surface_id = surface_id


class Sphere:
    def __init__(self, center, radius, material, surface_id="lens"):
        self.center = center
        self.r = radius
        self.material = material
        self.surface_id = surface_id

    def intersect(self, o, d, tmin, tmax):
        oc = vec3.sub(o, self.center)
        b = vec3.dot(oc, d)
        c = vec3.dot(oc, oc) - self.r * self.r
        disc = b * b - c
        if disc < 0.0:
            return None
        sq = sqrt(disc)
        for t in (-b - sq, -b + sq):
            if tmin < t < tmax:
                p = vec3.scale_add(o, t, d)
                n = vec3.normalize(vec3.sub(p, self.center))
                return Hit(t, p, n, self, self.surface_id)
        return None


# Face order: (axis index, sign) -> outward normal and label
_FACES = (
    (0, -1, (-1.0, 0.0, 0.0), "west"),
    (0, 1, (1.0, 0.0, 0.0), "east"),
    (1, -1, (0.0, -1.0, 0.0), "back"),
    (1, 1, (0.0, 1.0, 0.0), "front"),
    (2, -1, (0.0, 0.0, -1.0), "south"),
    (2, 1, (0.0, 0.0, 1.0), "north"),
)


class Plane:
    """Horizontal plane y = constant; used as a photon-killing ground."""

    def __init__(self, y, material, surface_id="ground"):
        self.y = y
        self.material = material
        self.surface_id = surface_id

    def intersect(self, o, d, tmin, tmax):
        if abs(d[1]) < 1e-12:
            return None
        t = (self.y - o[1]) / d[1]
        if not (tmin < t < tmax):
            return None
        p = vec3.scale_add(o, t, d)
        n = (0.0, 1.0, 0.0)
        return Hit(t, p, n, self, self.surface_id)


class Box:
    """Axis-aligned box; every one of the 6 faces keeps its own surface_id."""

    def __init__(self, lo, hi, material, prefix="wall"):
        self.lo = tuple(lo)
        self.hi = tuple(hi)
        self.material = material
        self.prefix = prefix

    def intersect(self, o, d, tmin, tmax):
        t_enter, t_exit = tmin, tmax
        enter_face = exit_face = None
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                if o[axis] < self.lo[axis] or o[axis] > self.hi[axis]:
                    return None
                continue
            inv = 1.0 / d[axis]
            t1 = (self.lo[axis] - o[axis]) * inv
            t2 = (self.hi[axis] - o[axis]) * inv
            near, far = (t1, t2) if t1 < t2 else (t2, t1)
            near_face = axis * 2 + (1 if t1 >= t2 else 0)
            far_face = axis * 2 + (0 if t1 >= t2 else 1)
            if near > t_enter:
                t_enter, enter_face = near, near_face
            if far < t_exit:
                t_exit, exit_face = far, far_face
            if t_enter > t_exit:
                return None
        if tmin < t_enter < tmax:
            t = t_enter
            face = _FACES[enter_face]
        elif tmin < t_exit < tmax:
            # Ray started inside the box: the exit face normal points against
            # the ray, which is exactly the geometric (outward) normal.
            t = t_exit
            face = _FACES[exit_face]
        else:
            return None
        p = vec3.scale_add(o, t, d)
        return Hit(t, p, face[2], self, f"{self.prefix}.{face[3]}")
