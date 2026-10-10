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
from dieren import ALLE, EIGEN, FANART, PER_SLEUTEL
from params import CARD_W, FILAMENTEN, PLATE_T, RIM_H, SLOT_DEPTH, TILT_DEG, WINDOW_Y0
from talen import TALEN

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


def M(np_matrix):
    return Matrix(np.asarray(np_matrix).tolist())


def materialen():
    """Eén materiaal per Bambu-filament; Matte iets ruwer dan Basic."""
    out = {}
    for code, (naam, materiaal, hexkleur, *_ ) in FILAMENTEN.items():
        rough = 0.62 if "Matte" in materiaal else 0.42
        out[code] = pla(naam, hexkleur, rough, floor=0.012 if code == "11101" else 0.0,
                        cap=0.82 if code == "11100" else 1.0)
    return out


def put_card(label, kind, mats, matrix, name):
    for c, m in K.card(label, kind).items():
        add(f"{name}_{c}", m, mats[c], matrix)


def scene_montage(mats, sleutel="vos", taal="nl", explode=0.0, lift=0.0, dx=0.0):
    """Gemonteerde kalender uit kalender.assembly(); lift schuift het dagkaartje omhoog (mm)."""
    parts, _ = K.assembly(sleutel, taal, explode=explode)
    a = math.radians(TILT_DEG)
    up = Matrix.Translation((0, math.sin(a) * lift, math.cos(a) * lift))
    dag = TALEN[taal]["vandaag"][0]
    bord = PER_SLEUTEL[sleutel].bord
    for n, c, m in parts:
        bevel = 0.6 if n == "voet" else 0.35 if (n == "front" and c == bord) else 0.0
        mat = up if n == f"kanaal {dag}" else Matrix.Identity(4)
        add(n, m, mats[c], T(dx) @ mat, bevel=bevel)


def scene_familie(mats, taal="nl", dieren=EIGEN):
    """De dieren naast elkaar, licht in een boog (fan-art apart: view 'fanart')."""
    n = len(dieren)
    for i, d in enumerate(dieren):
        x = (i - (n - 1) / 2) * 232.0
        y = abs(i - (n - 1) / 2) ** 1.6 * 28.0
        parts, _ = K.assembly(d.sleutel, taal)
        R = Matrix.Rotation(math.radians(-(i - (n - 1) / 2) * 6), 4, "Z")
        for nm, c, m in parts:
            bevel = 0.6 if nm == "voet" else 0.0
            add(nm, m, mats[c], T(x, y) @ R, bevel=bevel)


def scene_proef(mats, taal="nl"):
    """Sleuvenblok met de kanaalstrook in de middelste sleuf en drie cijferkaartjes erin."""
    oranje = mats["10301"]
    add("proef_sleuven", K.proef_sleuven()["proef"], oranje, bevel=0.4)
    strip_m = M(K.mounted(10.0, SLOT_DEPTH + 6))
    add("proef_kanalen", K.proef_kanalen()["proef"], oranje, strip_m, bevel=0.3)
    channels, _ = K.layout([CARD_W["cijfer"]] * 3)
    for (ca, cb), label in zip(channels, "012"):
        put_card(label, "cijfer", mats, strip_m @ T((ca + cb) / 2, 4.0 + RIM_H, PLATE_T + 0.05), f"proef_{label}")
    put_card(TALEN[taal]["proefdag"], "dag", mats, T(0, -42, 0) @ Matrix.Rotation(math.radians(4), 4, "Z"),
             "proef_dag")


def detail_camera():
    """Camera vlak voor de overgang dagkanaal / cijferkanalen, iets van rechtsboven."""
    channels, _ = K.layout([CARD_W[k] for k in K.KINDS])
    cb = channels[0][1]
    p = K.mounted() @ np.array([cb + 2, WINDOW_Y0 + RIM_H + 10, PLATE_T + 2, 1])
    a = math.radians(TILT_DEG)
    out = np.array([0, -math.cos(a), math.sin(a)])
    target = p[:3] * MM
    cam = target + out * 0.20 + np.array([0.06, 0, 0.06])
    return tuple(cam), tuple(target)


VIEWS = {
    # naam: (scène, opties, camera, doel, brandpuntsafstand, resolutie, diafragma)
    "geheel": ("montage", {}, (-0.20, -0.60, 0.25), (0.0, 0.03, 0.115), 70, (1100, 1300), 5.6),
    "variant": ("montage", {}, (-0.20, -0.60, 0.25), (0.0, 0.03, 0.115), 70, (900, 1060), 5.6),
    "vos": ("montage", {}, (-0.08, -0.52, 0.25), (0.0, 0.04, 0.145), 75, (1100, 1100), 5.6),
    "explosie": ("montage", {"explode": 1.0}, (-0.33, -0.56, 0.34), (0.0, 0.0, 0.15), 50, (1200, 1400), 8.0),
    "zijkant": ("montage", {}, (1.10, 0.035, 0.11), (0.0, 0.035, 0.11), 110, (1100, 1000), 11.0),
    "achterkant": ("montage", {}, (0.34, 0.60, 0.33), (0.0, 0.04, 0.085), 60, (1200, 1000), 8.0),
    "detail": ("montage", {"lift": 12.0}, None, None, 100, (1200, 900), 6.3),
    "proef": ("proef", {}, (-0.13, -0.30, 0.17), (0.0, -0.005, 0.025), 70, (1300, 1000), 5.6),
    "familie": ("familie", {}, (0.0, -1.72, 0.50), (0.0, 0.10, 0.10), 42, (2000, 860), 8.0),
    "fanart": ("familie", {"dieren": FANART}, (0.0, -1.40, 0.44), (0.0, 0.10, 0.10), 42, (2000, 900), 8.0),
}


def main(out, view="geheel", fast=False, sleutel="vos", taal="nl", samples=128):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    mats = materialen()
    kind, opts, cam_loc, aim, lens, (rx, ry), fstop = VIEWS[view]
    if kind == "proef":
        scene_proef(mats, taal)
    elif kind == "familie":
        scene_familie(mats, taal, **opts)
    else:
        scene_montage(mats, sleutel, taal, **opts)
    if cam_loc is None:
        cam_loc, aim = detail_camera()

    # tafel en achtergrond
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0.3, 0))
    bpy.context.object.data.materials.append(wood())
    if view not in ("achterkant",):
        bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0.75, 0.6), rotation=(math.radians(90), 0, 0))
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
    if view in ("achterkant", "zijkant"):
        light("achterlicht", (0.45, 0.7, 0.5), (-60, 0, 150), 14, 0.6)

    cam_d = bpy.data.cameras.new("camera")
    cam_d.lens = lens
    cam_d.dof.use_dof = True
    cam_d.dof.aperture_fstop = fstop
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
    sc.cycles.samples = 24 if fast else samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = rx
    sc.render.resolution_y = ry
    sc.render.resolution_percentage = 60 if fast else 100
    sc.render.filepath = str(out)
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("render klaar:", out)


def opties(argv):
    def val(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default
    return val("--dier", "vos"), val("--taal", "nl"), val("--uit", None)


if __name__ == "__main__":
    argv = sys.argv[1:]
    sleutel, taal, uit = opties(argv)
    fast = "--snel" in argv
    root = Path(__file__).resolve().parent.parent
    if "--varianten" in argv:
        # één productfoto per variant, in de map van die variant (fan-art in fanart/)
        from bouw import variant_map
        namen = [a for a in argv if not a.startswith("--")]
        for d in [PER_SLEUTEL[n] for n in namen] if namen else ALLE:
            for tl in TALEN:
                main(variant_map(d, tl) / f"{d.naam[tl]}_{tl}.png", "variant", fast, d.sleutel, tl, samples=64)
    else:
        views = list(VIEWS) if "--alle" in argv else [argv[0] if argv and not argv[0].startswith("--") else "geheel"]
        for view in views:
            if view == "variant":
                continue
            map_ = root / ("fanart" if view == "fanart" else "renders")   # fan-art blijft in zijn eigen map
            naam = "familie" if view == "fanart" else view
            out = Path(uit) if uit and len(views) == 1 else map_ / (
                f"{naam}.png" if view not in ("familie", "fanart", "proef") or taal == "nl" else f"{naam}_{taal}.png")
            main(out.resolve(), view, fast, sleutel, taal)
