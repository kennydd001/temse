"""Renders van H-02 Moodkin (Blender/Cycles, zonder scherm).

Gebruik:  python render.py [nacht|dag|open|alle] [--snel]
Schrijft naar renders/:
  nacht_<gezicht>.png  in het donker, met een LED-theelichtje binnenin, in elke stand
  gezichten.png        de vier nachtbeelden samen
  dag.png              overdag, schuin van voren
  open.png             deksel met trommel opgetild: zo zet je het lichtje erin

Het lichtje is een gewoon LED-theelichtje (wit potje, plastic vlammetje) met een kleine warme puntlamp in het
vlammetje.
"""
import math
import sys
from pathlib import Path

import bpy
from manifold3d import Manifold

import _pad  # noqa: F401
import lantaarn as L
import params as P
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"
KLEUR = {"schil": P.ORANJE, "trommel": P.GEEL, "deksel": P.ORANJE, "steel": P.GROEN}
VLAM_Z = P.VLOER + 16.0 + 9.0          # mm: potje van 16 mm, vlammetje erboven


def doorschijnend(mat, diepte):
    """De dunne trommel laat een beetje licht door: subsurface over een fractie van een mm (Burley: ruist niet)."""
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Subsurface Weight"].default_value = 1.0
    b.inputs["Subsurface Scale"].default_value = diepte
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.2)
    b.subsurface_method = "BURLEY"


def gloeiend(mat, mate=0.22, kleur=(1.0, 0.32, 0.04)):
    """De dikke schil gloeit zacht: een deel van het licht gaat diffuus door de wand (translucent). Subsurface
    over enkele mm geeft hetzelfde, maar ruist zichtbaar bij een lamp binnenin."""
    nt = mat.node_tree
    tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
    tr.inputs["Color"].default_value = (*kleur, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = mate
    nt.links.new(nt.nodes["Principled BSDF"].outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])


def lichtje(nacht):
    """LED-theelichtje op de vloer van de schil."""
    wit = BL.pla("led", "#F2F0EA", 0.5)
    potje = Manifold.cylinder(16.0, P.LED_D / 2, circular_segments=64).translate([0, 0, P.VLOER])
    BL.voeg_toe("led", potje, wit)
    vlam = Manifold.sphere(1.0, 32).scale([3.2, 3.2, 6.0]).translate([0, 0, VLAM_Z])
    m = bpy.data.materials.new("vlam")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (1.0, 0.75, 0.35, 1)
    b.inputs["Emission Color"].default_value = (1.0, 0.62, 0.22, 1)
    b.inputs["Emission Strength"].default_value = 40.0 if nacht else 4.0
    BL.voeg_toe("vlam", vlam, m)
    if nacht:
        ld = bpy.data.lights.new("led", "POINT")
        ld.energy = 3.0
        ld.shadow_soft_size = 0.003
        ld.color = (1.0, 0.68, 0.32)
        ob = bpy.data.objects.new("led", ld)
        ob.location = (0, 0, VLAM_Z * BL.MM)
        bpy.context.scene.collection.objects.link(ob)


def lantaarn(stand=0, open_=0.0, nacht=True):
    mats = BL.materialen(P.FILAMENTEN)
    if nacht:
        gloeiend(mats[P.ORANJE])
        doorschijnend(mats[P.GEEL], 0.00025)      # niet dieper, anders gloeit de trommel even fel als de gaten
    for naam, m in L.samen(stand, open_).items():
        BL.voeg_toe(naam, m, mats[KLEUR[naam]])
    lichtje(nacht)


def donker():
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0.3, 0))
    bpy.context.object.data.materials.append(BL.hout())
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0.6, 1.0), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(BL.pla("muur", "#8A8079", 0.9))
    world = bpy.data.worlds.new("wereld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.25, 0.3, 0.5, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.02
    bpy.context.scene.world = world
    BL.lamp("maan", (-0.35, 0.45, 0.45), (-55, 0, 200), 1.2, 0.6, (0.55, 0.65, 1.0))   # zwak tegenlicht: deksel en steeltje


def nacht(stand, snel):
    BL.leeg()
    lantaarn(stand)
    donker()
    BL.camera((0.03, -0.30, 0.12), (0.0, 0.0, 0.058), lens=50, diafragma=8.0)
    pad = UIT / f"nacht_{P.GEZICHTEN[stand]}.png"
    bpy.context.scene.cycles.sample_clamp_indirect = 3.0     # geen vuurvliegjes van het lichtje binnenin
    BL.render(pad, (1000, 1000), samples=128, snel=snel)
    return pad


def dag(snel):
    BL.leeg()
    lantaarn(0, nacht=False)
    BL.studio(muur_y=0.5)
    BL.camera((-0.11, -0.38, 0.17), (0.0, 0.0, 0.055), lens=55, diafragma=8.0)
    BL.render(UIT / "dag.png", (1200, 1000), samples=128, snel=snel)


def open_(snel):
    BL.leeg()
    lantaarn(0, open_=58.0, nacht=False)
    BL.studio(muur_y=0.5)
    BL.camera((0.22, -0.34, 0.24), (0.0, 0.0, 0.085), lens=50, diafragma=8.0)
    BL.render(UIT / "open.png", (1000, 1200), samples=128, snel=snel)


def overzicht():
    from PIL import Image, ImageDraw, ImageFont
    beelden = [Image.open(UIT / f"nacht_{n}.png").convert("RGB") for n in P.GEZICHTEN]
    w, h = beelden[0].size
    uit = Image.new("RGB", (2 * w, 2 * h))
    d = ImageDraw.Draw(uit)
    try:
        font = ImageFont.truetype(str(_pad.ROOT / "gereedschap" / "fonts" / "Fredoka-SemiBold.ttf"), 40)
    except OSError:
        font = None
    for i, (n, im) in enumerate(zip(P.GEZICHTEN, beelden)):
        x, y = (i % 2) * w, (i // 2) * h
        uit.paste(im, (x, y))
        d.text((x + 24, y + 18), f"{i * 90}°  {n}", fill=(255, 236, 200), font=font)
    uit.save(UIT / "gezichten.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    wat = [a for a in argv if not a.startswith("--")] or ["alle"]
    UIT.mkdir(exist_ok=True)
    if wat[0] in ("nacht", "alle"):
        standen = [int(a) for a in wat[1:]] or range(4)
        for k in standen:
            nacht(k, snel)
    if wat[0] in ("dag", "alle"):
        dag(snel)
    if wat[0] in ("open", "alle"):
        open_(snel)
    if all((UIT / f"nacht_{n}.png").exists() for n in P.GEZICHTEN):
        overzicht()
