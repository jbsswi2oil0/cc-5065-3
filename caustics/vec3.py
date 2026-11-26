"""Minimal 3D vector helpers using plain tuples (x, y, z)."""
from math import sqrt


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a):
    return sqrt(dot(a, a))


def normalize(a):
    n = length(a)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def scale_add(a, s, b):
    """a + s * b"""
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def reflect(d, n):
    k = 2.0 * dot(d, n)
    return (d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2])
