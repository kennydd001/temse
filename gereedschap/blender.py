"""Blender-hulpjes (Cycles, CPU, zonder scherm) voor productfoto's van printmodellen.

Gebruik in een render.py:
    from gereedschap import blender as BL
    BL.leeg()
    mats = BL.materialen(FILAMENTEN)              # {code: materiaal}, FILAMENTEN zoals in params.py
    BL.voeg_toe("voet", manifold, mats["11101"])  # manifold in mm
    BL.studio()                                    # houten tafel, muur, licht
    BL.camera((-0.2, -0.45, 0.25), (0, 0.04, 0.05), lens=60)
    BL.render(pad, (1200, 1000), samples=96)

Vereist de pip-module `bpy` (Blender als Python-module).
"""
import math

import bpy
import numpy as np
from mathutils import Matrix

MM = 0.001


def leeg():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def srgb_naar_lineair(hexcode, vloer=0.0):
    rgb = [int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return [max(vloer, c) for c in lin] + [1.0]


def pla(naam, hexcode, ruwheid=0.5, vloer=0.0, plafond=1.0):
    m = bpy.data.materials.new(naam)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    kleur = srgb_naar_lineair(hexcode, vloer)
    b.inputs["Base Color"].default_value = [min(plafond, c) for c in kleur[:3]] + [1.0]
    b.inputs["Roughness"].default_value = ruwheid
    b.inputs["Specular IOR Level"].default_value = 0.35
    return m


def metaal(naam="staal", kleur=(0.62, 0.63, 0.66), ruwheid=0.28):
    m = bpy.data.materials.new(naam)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*kleur, 1.0)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = ruwheid
    return m


def materialen(filamenten):
    """Eén materiaal per Bambu-filament; Matte iets ruwer dan Basic, zwart en wit iets getemperd."""
    out = {}
    for code, (naam, materiaal, hexkleur, *_) in filamenten.items():
        ruw = 0.62 if "Matte" in materiaal else 0.42
        out[code] = pla(naam, hexkleur, ruw, vloer=0.012 if code == "11101" else 0.0,
                        plafond=0.82 if code == "11100" else 1.0)
    return out


def voeg_toe(naam, manifold, mat, matrix=None, afronding=0.0):
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties[:, :3], dtype=float)
    f = np.asarray(mesh.tri_verts, dtype=int)
    me = bpy.data.meshes.new(naam)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.validate()
    me.update()
    ob = bpy.data.objects.new(naam, me)
    ob.data.materials.append(mat)
    ob.matrix_world = Matrix.Scale(MM, 4) @ (matrix if matrix is not None else Matrix.Identity(4))
    bpy.context.collection.objects.link(ob)
    if afronding:
        mod = ob.modifiers.new("afronding", "BEVEL")
        mod.width = afronding
        mod.segments = 2
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(50)
    return ob


def hout():
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


def studio(muur=True, muur_y=0.75, achterlicht=False):
    """Houten tafel (z = 0), muur erachter, drie vlaklampen."""
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0.3, 0))
    bpy.context.object.data.materials.append(hout())
    if muur:
        bpy.ops.mesh.primitive_plane_add(size=8, location=(0, muur_y, 0.6), rotation=(math.radians(90), 0, 0))
        bpy.context.object.data.materials.append(pla("muur", "#BFA88F", 0.9))
    world = bpy.data.worlds.new("wereld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.47, 0.40, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.15
    sc.world = world
    lamp("hoofdlicht", (-0.45, -0.55, 0.65), (55, 0, -38), 18, 0.5)
    lamp("vullicht", (0.6, -0.35, 0.3), (70, 0, 55), 4, 0.8, (0.92, 0.96, 1))
    lamp("tegenlicht", (0.1, 0.5, 0.6), (-50, 0, 180), 8, 0.4)
    if achterlicht:
        lamp("achterlicht", (0.45, 0.7, 0.5), (-60, 0, 150), 14, 0.6)


def lamp(naam, plek, draai, energie, maat, kleur=(1, 0.96, 0.92)):
    ld = bpy.data.lights.new(naam, "AREA")
    ld.energy = energie
    ld.size = maat
    ld.color = kleur
    ob = bpy.data.objects.new(naam, ld)
    ob.location = plek
    ob.rotation_euler = [math.radians(a) for a in draai]
    bpy.context.scene.collection.objects.link(ob)


def camera(plek, doel, lens=60, diafragma=5.6):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("camera")
    cd.lens = lens
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = diafragma
    cam = bpy.data.objects.new("camera", cd)
    sc.collection.objects.link(cam)
    cam.location = plek
    t = bpy.data.objects.new("doel", None)
    t.location = doel
    sc.collection.objects.link(t)
    con = cam.constraints.new("TRACK_TO")
    con.target = t
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cd.dof.focus_object = t
    sc.camera = cam


def render(pad, resolutie, samples=96, snel=False):
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = -0.55
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24 if snel else samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = resolutie
    sc.render.resolution_percentage = 60 if snel else 100
    sc.render.filepath = str(pad)
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print("render klaar:", pad)
