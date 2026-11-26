"""Scene: a movable transparent window (thin glass pane + focusing lens)
floating just above two very close, oppositely-facing receiver surfaces.

The receiver is a thin wall box.  Its two large faces share the same plane
geometry but have distinct surface ids, so photons can never be borrowed
across the wall by the density estimator.
"""
from .materials import Material
from .primitives import Box, Plane, Sphere

AIR_IOR = 1.0
GLASS_IOR = 1.5

# Receiver wall (thin slab). Front faces -y, back faces +y.
WALL_HX = 3.0
WALL_HZ = 3.0
WALL_TOP = 0.6
WALL_BOTTOM = 0.4          # thickness 0.2 -> front/back faces 0.2 apart
FRONT_ID = "wall.front"
BACK_ID = "wall.back"
RECEIVER_IDS = (FRONT_ID, BACK_ID)

# Light beam (collimated, travelling -y) emitted from this height/area.
# Wider than the pane so the outer rays bypass the window and reach the wall
# directly (direct photons), while the inner rays pass through the glass and
# form the focused caustic.
EMIT_Y = 5.0
BEAM_HX = 2.2
BEAM_HZ = 2.2

# Window assembly: a thin glass pane with a focusing sphere lens mounted in
# it. The lens is large enough and close enough to the wall that its focal
# point lands just above the receiver, producing a tight, visible caustic.
PANE_HX = 1.8
PANE_HZ = 1.8
PANE_THICK = 0.12
PANE_Y = 2.6
LENS_R = 0.9
LENS_CENTER_Y = 1.8


def build_scene(pane_x=0.0, pane_z=0.0):
    glass = Material("glass", "dielectric", ior=GLASS_IOR)
    receiver = Material("receiver", "diffuse", albedo=(0.75, 0.73, 0.68))
    ground = Material("ground", "diffuse", albedo=(0.1, 0.1, 0.12))

    wall = Box((-WALL_HX, WALL_BOTTOM, -WALL_HZ),
               (WALL_HX, WALL_TOP, WALL_HZ), receiver, prefix="wall")
    pane = Box((pane_x - PANE_HX, PANE_Y - PANE_THICK, pane_z - PANE_HZ),
               (pane_x + PANE_HX, PANE_Y + PANE_THICK, pane_z + PANE_HZ),
               glass, prefix="pane")
    lens = Sphere((pane_x, LENS_CENTER_Y, pane_z), LENS_R,
                  glass, surface_id="lens")
    floor = Plane(-1.0, ground)

    primitives = [wall, pane, lens, floor]
    return {
        "primitives": primitives,
        "glass": glass,
        "receiver": receiver,
        "params": {"pane_x": pane_x, "pane_z": pane_z},
    }
