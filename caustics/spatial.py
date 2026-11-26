"""Spatial indices for photon neighborhood queries.

The photon estimator needs, for a query point and a world-space radius R, the
set of photons *of one surface id* whose Euclidean distance to the query is
``d <= R`` (the boundary is **inclusive**; see ``BOUNDARY_RULE``).

Historic implementation (still retained for regression / comparison)
--------------------------------------------------------------------
``HashPhotonIndex`` is the original uniform hash grid: it widens the query to
the integer cell range covering the sphere's axis-aligned bounding box
("bucket coarse query") and then rejects individual photons by exact distance
("distance re-filter").  Its cell range is derived with ``floor`` and the exact
test uses ``<=``, so it is exact and boundary-inclusive.

Migration target (default)
--------------------------
``KdPhotonIndex`` is a balanced median-split 3-D kd-tree, built lazily per
surface id.  Two query strategies are provided and both return the *same
exact* inclusive set:

``query_mode="sphere"``
    "Exact radius filtering": prune by the squared distance from the query to
    each node's bounding box, then apply the exact ``d^2 <= R^2`` test in the
    leaves.  The bounding-box test is ``<=`` so a photon sitting exactly on the
    sphere boundary at a box corner can never be pruned away before the exact
    test.

``query_mode="bucket"``
    "Spatial-bucket coarse query followed by distance re-filter": gather every
    photon in the leaves whose bounding box overlaps the sphere's AABB, then
    re-filter that candidate list by ``d^2 <= R^2``.  Coarser pruning; useful
    when a caller wants a candidate superset.

The default mode is chosen empirically (``DEFAULT_QUERY_MODE`` below, see the
benchmark in ``benchmarks/index_bench.py``), not assumed.

Units and boundary contract
---------------------------
* ``radius`` is a world-space distance in the same units as ``photon.pos``.
* Squared distances are compared internally to avoid ``sqrt`` rounding.
* ``BOUNDARY_RULE`` records the inclusive ``<=`` rule; every backend and both
  query modes obey it, and ``tests/test_spatial_index.py`` pins the behaviour
  with analytic radius shells and coincident photons.
"""
import math

INDEX_VERSION = "2.0"
NEIGHBORHOOD_INDEX_MIGRATION = "kdtree-2026-11"
BOUNDARY_RULE = "distance <= radius (inclusive)"
DEFAULT_BACKEND = "kdtree"

# Chosen from benchmarks/index_bench_sweep.py (mean over 3 budgets x 4 radii x
# 3 seeds, 2500 queries/config): kdtree:sphere 0.826s vs kdtree:bucket 0.879s,
# both bit-for-bit equal to the brute-force oracle.  This is also structurally
# expected: a box that is farther than R from the query cannot contain an
# in-sphere photon, so the sphere test's kept-node set is a subset of the AABB
# bucket test's — sphere prunes at least as much, and skips materialising the
# candidate list.  Re-run the sweep to revisit the choice.
DEFAULT_QUERY_MODE = "sphere"

_KD_LEAF_SIZE = 16


def _bbox_dist2(lo, hi, x, y, z):
    """Squared distance from point to an axis-aligned box (0 if inside)."""
    if x < lo[0]:
        dx = lo[0] - x
    elif x > hi[0]:
        dx = x - hi[0]
    else:
        dx = 0.0
    if y < lo[1]:
        dy = lo[1] - y
    elif y > hi[1]:
        dy = y - hi[1]
    else:
        dy = 0.0
    if z < lo[2]:
        dz = lo[2] - z
    elif z > hi[2]:
        dz = z - hi[2]
    else:
        dz = 0.0
    return dx * dx + dy * dy + dz * dz


def _bbox_overlaps(lo, hi, qlo, qhi):
    """Inclusive AABB overlap test (touching counts as overlap)."""
    return not (lo[0] > qhi[0] or hi[0] < qlo[0]
                or lo[1] > qhi[1] or hi[1] < qlo[1]
                or lo[2] > qhi[2] or hi[2] < qlo[2])


class _KdNode:
    __slots__ = ("lo", "hi", "axis", "split", "left", "right", "items")

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.axis = -1
        self.split = 0.0
        self.left = None
        self.right = None
        self.items = None


def _build_kd(idxs, photons, leaf_size):
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for i in idxs:
        p = photons[i].pos
        for a in range(3):
            v = p[a]
            if v < lo[a]:
                lo[a] = v
            if v > hi[a]:
                hi[a] = v
    node = _KdNode(lo, hi)
    if len(idxs) <= leaf_size:
        node.items = idxs
        return node
    extent = [hi[a] - lo[a] for a in range(3)]
    axis = 0 if extent[0] >= extent[1] and extent[0] >= extent[2] else (
        1 if extent[1] >= extent[2] else 2)
    ordered = sorted(idxs, key=lambda i: photons[i].pos[axis])
    mid = len(ordered) // 2
    node.axis = axis
    node.split = photons[ordered[mid]].pos[axis]
    node.left = _build_kd(ordered[:mid], photons, leaf_size)
    node.right = _build_kd(ordered[mid:], photons, leaf_size)
    return node


class _BaseIndex:
    """Shared bookkeeping: photons list, per-surface ids, counts."""

    backend = "base"
    query_mode = "n/a"
    version = INDEX_VERSION

    def __init__(self):
        self.photons = []
        self._by_surface = {}

    def add(self, photon):
        idx = len(self.photons)
        self.photons.append(photon)
        self._by_surface.setdefault(photon.surface_id, []).append(idx)

    def count(self, surface_id=None):
        if surface_id is None:
            return len(self.photons)
        return len(self._by_surface.get(surface_id, ()))

    def neighborhood(self, center, radius, surface_id):  # pragma: no cover
        raise NotImplementedError


class BruteForceIndex(_BaseIndex):
    """O(P) exact scan. Reference oracle for audits; never used for rendering."""

    backend = "brute_force"
    query_mode = "exact_scan"

    def neighborhood(self, center, radius, surface_id):
        idxs = self._by_surface.get(surface_id)
        if not idxs:
            return []
        cx, cy, cz = center
        r2 = radius * radius
        photons = self.photons
        out = []
        for i in idxs:
            p = photons[i].pos
            dx = p[0] - cx
            dy = p[1] - cy
            dz = p[2] - cz
            if dx * dx + dy * dy + dz * dz <= r2:
                out.append(photons[i])
        return out


class HashPhotonIndex(_BaseIndex):
    """Original uniform hash grid: bucket coarse query + distance re-filter."""

    backend = "hash_grid"
    query_mode = "bucket_refilter"

    def __init__(self, cell_size):
        super().__init__()
        self.cell = float(cell_size)
        self.grids = {}  # surface_id -> dict[(ix,iy,iz)] -> list[idx]

    def add(self, photon):
        idx = len(self.photons)
        super().add(photon)
        grid = self.grids.setdefault(photon.surface_id, {})
        c = self.cell
        key = (int(math.floor(photon.pos[0] / c)),
               int(math.floor(photon.pos[1] / c)),
               int(math.floor(photon.pos[2] / c)))
        grid.setdefault(key, []).append(idx)

    def neighborhood(self, center, radius, surface_id):
        """Return photons of one surface id within ``radius`` of ``center``."""
        grid = self.grids.get(surface_id)
        if grid is None:
            return []
        c = self.cell
        x0 = int(math.floor((center[0] - radius) / c))
        x1 = int(math.floor((center[0] + radius) / c))
        y0 = int(math.floor((center[1] - radius) / c))
        y1 = int(math.floor((center[1] + radius) / c))
        z0 = int(math.floor((center[2] - radius) / c))
        z1 = int(math.floor((center[2] + radius) / c))
        r2 = radius * radius
        out = []
        photons = self.photons
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                for iz in range(z0, z1 + 1):
                    bucket = grid.get((ix, iy, iz))
                    if not bucket:
                        continue
                    for j in bucket:
                        ph = photons[j]
                        dx = ph.pos[0] - center[0]
                        dy = ph.pos[1] - center[1]
                        dz = ph.pos[2] - center[2]
                        if dx * dx + dy * dy + dz * dz <= r2:
                            out.append(ph)
        return out


class KdPhotonIndex(_BaseIndex):
    """Balanced median-split kd-tree over photon positions, per surface id."""

    backend = "kdtree"

    def __init__(self, cell_size=0.08, query_mode=DEFAULT_QUERY_MODE):
        # cell_size is accepted for signature compatibility with the legacy
        # hash-grid constructor; a kd-tree does not use fixed cells.
        super().__init__()
        if query_mode not in ("sphere", "bucket"):
            raise ValueError("query_mode must be 'sphere' or 'bucket'")
        self.query_mode = query_mode
        self.cell = float(cell_size)
        self._trees = {}

    def add(self, photon):
        super().add(photon)
        self._trees.pop(photon.surface_id, None)

    def _ensure_tree(self, surface_id):
        if surface_id in self._trees:
            return self._trees[surface_id]
        idxs = self._by_surface.get(surface_id)
        if not idxs:
            return None
        tree = _build_kd(list(idxs), self.photons, _KD_LEAF_SIZE)
        self._trees[surface_id] = tree
        return tree

    def neighborhood(self, center, radius, surface_id):
        tree = self._ensure_tree(surface_id)
        if tree is None:
            return []
        cx, cy, cz = center
        r2 = radius * radius
        photons = self.photons
        out = []
        stack = [tree]
        if self.query_mode == "sphere":
            while stack:
                node = stack.pop()
                if _bbox_dist2(node.lo, node.hi, cx, cy, cz) > r2:
                    continue
                if node.items is not None:
                    for i in node.items:
                        p = photons[i].pos
                        dx = p[0] - cx
                        dy = p[1] - cy
                        dz = p[2] - cz
                        if dx * dx + dy * dy + dz * dz <= r2:
                            out.append(photons[i])
                else:
                    stack.append(node.left)
                    stack.append(node.right)
            return out

        # bucket: coarse AABB gather, exact distance re-filter afterwards.
        qlo = (cx - radius, cy - radius, cz - radius)
        qhi = (cx + radius, cy + radius, cz + radius)
        candidates = []
        while stack:
            node = stack.pop()
            if not _bbox_overlaps(node.lo, node.hi, qlo, qhi):
                continue
            if node.items is not None:
                candidates.extend(node.items)
            else:
                stack.append(node.left)
                stack.append(node.right)
        for i in candidates:
            p = photons[i].pos
            dx = p[0] - cx
            dy = p[1] - cy
            dz = p[2] - cz
            if dx * dx + dy * dy + dz * dz <= r2:
                out.append(photons[i])
        return out


_BACKENDS = {
    "kdtree": KdPhotonIndex,
    "hash_grid": HashPhotonIndex,
    "grid": HashPhotonIndex,
    "brute_force": BruteForceIndex,
}


def make_index(backend=DEFAULT_BACKEND, cell_size=0.08, query_mode=None):
    """Build a photon index for ``backend``.

    ``kdtree`` is the migrated default; ``hash_grid`` selects the legacy
    implementation so old configurations keep working.
    """
    if backend not in _BACKENDS:
        raise ValueError("unknown photon index backend: %r" % (backend,))
    if backend in ("kdtree",):
        if query_mode is None:
            query_mode = DEFAULT_QUERY_MODE
        return KdPhotonIndex(cell_size, query_mode=query_mode)
    return _BACKENDS[backend](cell_size)
