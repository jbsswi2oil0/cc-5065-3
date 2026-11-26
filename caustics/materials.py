"""Material model and dielectric Fresnel / Snell physics."""
from math import sqrt

from . import vec3


class Material:
    def __init__(self, name, kind, ior=1.0, albedo=(0.8, 0.8, 0.8)):
        self.name = name
        # kind: "diffuse" | "dielectric"
        self.kind = kind
        self.ior = ior
        self.albedo = albedo


def fresnel_schlick(cos_i, ior_from, ior_to):
    """Unpolarized Fresnel reflectance for the given cosine of incidence."""
    eta = ior_from / ior_to
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return 1.0  # total internal reflection
    cos_t = sqrt(1.0 - sin2_t)
    r_s = ((ior_from * cos_i - ior_to * cos_t) /
           (ior_from * cos_i + ior_to * cos_t)) ** 2
    r_p = ((ior_to * cos_i - ior_from * cos_t) /
           (ior_to * cos_i + ior_from * cos_t)) ** 2
    return 0.5 * (r_s + r_p)


def refract(d, n, ior_from, ior_to):
    """Refracted direction; returns None on total internal reflection."""
    eta = ior_from / ior_to
    cos_i = -vec3.dot(d, n)
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return None
    cos_t = sqrt(1.0 - sin2_t)
    # d_t = eta * d + (eta*cos_i - cos_t) * n
    t = vec3.add(vec3.mul(d, eta), vec3.mul(n, eta * cos_i - cos_t))
    return vec3.normalize(t)
