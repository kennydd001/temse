"""Renders van H-01 Schaduwspook (Blender/Cycles, zonder scherm): de echte schaduw op de muur.

Gebruik:  python render.py [sleutel ...|alle] [--snel]
Schrijft naar renders/: <sleutel>_<hoek>.png per schaduw (het beeldje zo gedraaid dat die kijkrichting naar
de lamp wijst) en schaduwen.png met alle schaduwen samen.

Het licht is een puntlamp van 2 mm, 45 cm voor het beeldje en vlak boven de tafel, zoals een gsm-zaklamp.
De muur staat 25 cm achter het beeldje. Zo valt de schaduw op de muur, vanaf de tafel omhoog.
"""
import math
import sys
from pathlib import Path

import bpy

import _pad  # noqa: F401
import beeldje as BJ
import params as P
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"
LAMP = (0.0, -0.45, 0.006)        # m: voor het beeldje, net boven de tafel
MUUR_Y = 0.25


def kamer():
    """Donkere kamer: tafel, lichte muur erachter, één puntlamp en een heel klein beetje omgevingslicht."""
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0.2, 0))
    bpy.context.object.data.materials.append(BL.hout())
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, MUUR_Y, 1.0), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(BL.pla("muur", "#E9E2D6", 0.9))
    world = bpy.data.worlds.new("wereld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.38, 0.5, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.012
    bpy.context.scene.world = world
    ld = bpy.data.lights.new("zaklamp", "POINT")
    ld.energy = 7.0
    ld.shadow_soft_size = 0.002
    ld.color = (1.0, 0.97, 0.92)
    ob = bpy.data.objects.new("zaklamp", ld)
    ob.location = LAMP
    bpy.context.scene.collection.objects.link(ob)


def scene(sleutel, hoek, snel):
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN)
    m = BJ.beeldje(sleutel)
    if hoek:
        m = m.rotate([0, 0, -hoek])            # deze kijkrichting wijst nu naar de lamp
    BL.voeg_toe(sleutel, m, mats[P.VARIANTEN[sleutel][5]])
    kamer()
    BL.camera((-0.24, -0.40, 0.15), (0.01, 0.12, 0.05), lens=38, diafragma=11.0)
    pad = UIT / f"{sleutel}_{hoek:.0f}.png"
    BL.render(pad, (1200, 900), samples=128, snel=snel)
    return pad


def overzicht(sleutels):
    """Alle renders in één beeld, vier per rij; de schaduwen van één variant blijven naast elkaar."""
    from PIL import Image, ImageDraw, ImageFont
    n = max(len(P.VARIANTEN[k][3]) for k in sleutels)
    vakken, r, c = [], 0, 0
    for k in sleutels:
        hoeken = P.VARIANTEN[k][3]
        if c + len(hoeken) > n:
            r, c = r + 1, 0
        vakken += [(k, h, r, c + i) for i, h in enumerate(hoeken)]
        c += len(hoeken)
    eerste = Image.open(UIT / f"{sleutels[0]}_0.png")
    w2, h2 = eerste.size[0] // 2, eerste.size[1] // 2
    uit = Image.new("RGB", (n * w2, (r + 1) * h2), (30, 28, 26))
    d = ImageDraw.Draw(uit)
    try:
        font = ImageFont.truetype(str(_pad.ROOT / "gereedschap" / "fonts" / "Fredoka-SemiBold.ttf"), 26)
    except OSError:
        font = None
    for k, h, r, c in vakken:
        im = Image.open(UIT / f"{k}_{h:.0f}.png").convert("RGB").resize((w2, h2))
        uit.paste(im, (c * w2, r * h2))
        tekst = f"{P.VARIANTEN[k][0]}: " + ("zo neergezet" if h == 0 else f"{h:.0f}° gedraaid")
        d.text((c * w2 + 16, r * h2 + 12), tekst, fill=(255, 255, 255), font=font)
    uit.save(UIT / "schaduwen.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    namen = [a for a in argv if not a.startswith("--")] or ["alle"]
    if namen == ["alle"]:
        namen = list(P.VARIANTEN)
    UIT.mkdir(exist_ok=True)
    for k in namen:
        for h in P.VARIANTEN[k][3]:
            scene(k, h, snel)
    if all((UIT / f"{k}_{h:.0f}.png").exists() for k, v in P.VARIANTEN.items() for h in v[3]):
        overzicht(list(P.VARIANTEN))
