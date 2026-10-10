"""Blender-render van de opgebouwde kalender (Cycles, CPU, zonder scherm).

Gebruik:  python render.py [uitvoer.png] [--snel]
Vereist de pip-module `bpy` (Blender als Python-module) in dezelfde omgeving.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

import kalender as K
from params import (BIN_INNER_W, CARD_T, CARD_W, DAGEN, FILAMENT, FLOOR, FOOT_H, MAANDEN, PLATE_T, RIM_H,
                    SLOT_FRONT_Y, SLOT_W, TILT_DEG, WALL, WINDOW_Y0)

MM = 0.001


def srgb_to_linear(hexcode, floor=0.0):
    rgb = [int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return [max(floor, c) for c in lin] + [1.0]


def pla(name, hexcode, rough=0.5, floor=0.0, cap=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    col = srgb_to_linear(hexcode, floor)
    b.inputs["Base Color"].default_value = [min(cap, c) for c in col[:3]] + [1.0]
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.35
    return m


def add(name, manifold, mat, matrix=Matrix.Identity(4), bevel=0.0):
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties[:, :3], dtype=float)
    f = np.asarray(mesh.tri_verts, dtype=int)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    ob.matrix_world = Matrix.Scale(MM, 4) @ matrix
    bpy.context.collection.objects.link(ob)
    if bevel:
        mod = ob.modifiers.new("afronding", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(50)
        mod.harden_normals = False
    return ob


def T(x=0.0, y=0.0, z=0.0):
    return Matrix.Translation((x, y, z))


def Rx(deg):
    return Matrix.Rotation(math.radians(deg), 4, "X")


def wood():
    m = bpy.data.materials.new("hout")
    m.use_nodes = True
    nt = m.node_tree
    n = nt.nodes
    bsdf = n["Principled BSDF"]
    coord = n.new("ShaderNodeTexCoord")
    mapping = n.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (2.0, 18.0, 2.0)
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Distortion"].default_value = 2.5
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.16, 0.075, 0.03, 1)
    ramp.color_ramp.elements[1].color = (0.42, 0.22, 0.10, 1)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return m


def mounted(slot_front_y, z_top):
    """Matrix die een liggend gemodelleerd front in een sleuf zet (8° achterover)."""
    lean = Rx(90 - TILT_DEG)
    gap = (SLOT_W - PLATE_T) / 2
    p = (lean @ T(0, 0, PLATE_T)).to_translation()
    return T(0, slot_front_y + gap / math.cos(math.radians(TILT_DEG)) - p.y, z_top) @ lean


def put_card(label, kind, mats, matrix, name):
    for c, m in K.card(label, kind).items():
        add(f"{name}_{c}", m, mats[c], matrix)


def scene_geheel(mats, date):
    for c, m in K.foot().items():
        add("voet", m, mats[c], bevel=0.6)
    front_m = mounted(SLOT_FRONT_Y, FOOT_H)
    front, channels = K.front_fox()
    for c, m in front.items():
        add(f"front_{c}", m, mats[c], front_m, bevel=0.35 if c == K.ORANJE else 0.0)

    # ingestelde datum in de vier kanalen
    kinds = ["dag", "cijfer", "cijfer", "maand"]
    card_y0 = WINDOW_Y0 + RIM_H
    for (ca, cb), label, kind in zip(channels, date, kinds):
        put_card(label, kind, mats, front_m @ T((ca + cb) / 2, card_y0, PLATE_T + 0.05), f"kaart_{label}")

    # overige kaartjes rechtop in de bakken, tekst naar voren
    stacks = [[d for d in DAGEN if d != date[0]], [d for d in "0123" if d != date[1]],
              [d for d in "0123456789" if d != date[2]], [m for m in MAANDEN if m != date[3]]]
    x = -(sum(BIN_INNER_W) + WALL * 3) / 2
    for w, kind, labels in zip(BIN_INNER_W, kinds, stacks):
        for i, label in enumerate(labels):
            y = WALL + 1.0 + i * (CARD_T + 0.15)
            place = T(x + w / 2, y + CARD_T, FLOOR + 0.05) @ Rx(90 + min(10, 2 + 0.6 * i))
            put_card(label, kind, mats, place, f"bak_{label}")
        x += w + WALL


def scene_proef(mats):
    """Sleuvenblok met de kanaalstrook in de middelste sleuf en drie cijferkaartjes erin."""
    sl = K.proef_sleuven()[K.ORANJE]
    add("proef_sleuven", sl, mats[K.ORANJE], T(0, 0, 0), bevel=0.4)
    z_top = K.SLOT_DEPTH + 6
    strip_m = mounted(10.0, z_top)
    add("proef_kanalen", K.proef_kanalen()[K.ORANJE], mats[K.ORANJE], strip_m, bevel=0.3)
    channels, _ = K.layout([CARD_W["cijfer"]] * 3)
    for (ca, cb), label in zip(channels, "012"):
        put_card(label, "cijfer", mats, strip_m @ T((ca + cb) / 2, 4.0 + RIM_H, PLATE_T + 0.05), f"proef_{label}")
    put_card("WOENSDAG", "dag", mats, T(0, -42, 0) @ Matrix.Rotation(math.radians(4), 4, "Z"), "proef_dag")


VIEWS = {
    # naam: (camera, doel, brandpuntsafstand, resolutie)
    "geheel": ((-0.20, -0.60, 0.25), (0.0, 0.03, 0.115), 70, (1100, 1300)),
    "vos": ((-0.08, -0.52, 0.25), (0.0, 0.04, 0.145), 75, (1100, 1100)),
    "proef": ((-0.13, -0.30, 0.17), (0.0, -0.005, 0.025), 70, (1300, 1000)),
}


def main(out, view="geheel", fast=False, date=("ZATERDAG", "1", "0", "OKTOBER")):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    mats = {
        K.ORANJE: pla("oranje", FILAMENT[K.ORANJE], 0.42),
        K.WIT: pla("wit", FILAMENT[K.WIT], 0.6, cap=0.82),
        K.ZWART: pla("zwart", FILAMENT[K.ZWART], 0.6, floor=0.012),
        K.GROEN: pla("groen", FILAMENT[K.GROEN], 0.42),
    }
    if view == "proef":
        scene_proef(mats)
    else:
        scene_geheel(mats, date)

    # tafel en achtergrond
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0.3, 0))
    bpy.context.object.data.materials.append(wood())
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0.75, 0.6), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(pla("muur", "#BFA88F", 0.9))

    world = bpy.data.worlds.new("wereld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.47, 0.40, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.15
    sc.world = world

    def light(name, loc, rot, energy, size, color=(1, 0.96, 0.92)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        ob.rotation_euler = [math.radians(a) for a in rot]
        sc.collection.objects.link(ob)

    light("hoofdlicht", (-0.45, -0.55, 0.65), (55, 0, -38), 18, 0.5)
    light("vullicht", (0.6, -0.35, 0.3), (70, 0, 55), 4, 0.8, (0.92, 0.96, 1))
    light("tegenlicht", (0.1, 0.5, 0.6), (-50, 0, 180), 8, 0.4)

    cam_loc, aim, lens, (rx, ry) = VIEWS[view]
    cam_d = bpy.data.cameras.new("camera")
    cam_d.lens = lens
    cam_d.dof.use_dof = True
    cam_d.dof.aperture_fstop = 5.6
    cam = bpy.data.objects.new("camera", cam_d)
    sc.collection.objects.link(cam)
    cam.location = cam_loc
    target = bpy.data.objects.new("doel", None)
    target.location = aim
    sc.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam_d.dof.focus_object = target
    sc.camera = cam

    sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = -0.55
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24 if fast else 160
    sc.cycles.use_denoising = True
    sc.render.resolution_x = rx
    sc.render.resolution_y = ry
    sc.render.resolution_percentage = 60 if fast else 100
    sc.render.filepath = str(out)
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("render klaar:", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    view = args[0] if args else "geheel"
    out = Path(args[1]) if len(args) > 1 else Path(__file__).resolve().parent.parent / "renders" / f"{view}.png"
    main(out.resolve(), view, fast="--snel" in sys.argv)
