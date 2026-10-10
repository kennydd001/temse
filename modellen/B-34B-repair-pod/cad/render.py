"""Productfoto's van B-34B Repair Pod (Blender/Cycles, zonder scherm).

Gebruik:  python render.py [dicht|half|open|detail|explosie|reeks|alle] [--snel]
Schrijft naar renders/. 'reeks' zet dicht, half en open naast elkaar (renders/reeks.png).
Bits in het dak en schroefjes in de bakjes zijn rekwisieten (niet om te printen).
"""
import math
import random
import sys
from pathlib import Path

from manifold3d import Manifold

import _pad  # noqa: F401
import params as P
import pod
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"

VIEWS = {
    # naam: (hoek, camera, doel, lens, resolutie)
    "dicht": (P.THETA_DICHT, (-0.20, -0.40, 0.23), (0.0, 0.042, 0.050), 62, (1200, 1000)),
    "half": (60.0, (-0.22, -0.44, 0.25), (0.0, 0.042, 0.048), 58, (1200, 1000)),
    "open": (P.THETA_OPEN, (-0.21, -0.40, 0.36), (0.0, 0.045, 0.030), 54, (1300, 1000)),
    "detail": (60.0, (-0.075, -0.17, 0.085), (0.0, -0.002, 0.026), 85, (1200, 900)),
    "explosie": (P.THETA_DICHT, (-0.30, -0.52, 0.30), (0.0, 0.040, 0.085), 68, (1200, 1300)),
}


def bit():
    """1/4"-bit (25 mm): zeskant met de groef voor de snelwisselhouder, een hals en een kruiskop."""
    r = 6.35 / math.sqrt(3)
    hexa = Manifold.cylinder(18.0, r, r, 6).rotate([0, 0, 30])
    hexa = hexa - (Manifold.cylinder(1.6, 5, 5, 32) - Manifold.cylinder(1.6, 2.6, 2.6, 32)).translate([0, 0, 5.0])
    hals = Manifold.cylinder(3.0, 2.9, 2.0, 32).translate([0, 0, 18.0])
    kop = Manifold.cylinder(3.0, 2.0, 0.9, 32).translate([0, 0, 21.0])
    kruis = (Manifold.cube([0.7, 5, 2.2], center=True) + Manifold.cube([5, 0.7, 2.2], center=True)).translate([0, 0, 23.2])
    return hexa + hals + (kop - kruis)


def schroef(lengte=10.0, d=3.0, kop=5.5):
    return Manifold.cylinder(lengte, d / 2, d / 2, 16) + Manifold.cylinder(1.8, kop / 2, kop / 2, 24).translate([0, 0, -1.8])


def moer(sw=5.5, h=2.4, gat=3.0):
    r = sw / math.sqrt(3)
    return Manifold.cylinder(h, r, r, 6) - Manifold.cylinder(h + 2, gat / 2, gat / 2, 16).translate([0, 0, -1])


def lading(kant, rnd):
    """Wat schroefjes en moertjes op de bodem van een bak (rechts: schroefjes, links: moertjes)."""
    out = []
    z = P.BAK_Z0 + P.BODEM
    x0, x1 = P.BAK_X0 + P.ZIJWAND + 5, P.BAK_X0 + P.BAK_B - P.ZIJWAND - 9
    for k in range(14):
        x = rnd.uniform(x0, x1)
        y = rnd.uniform(P.WAND_VOOR + 6, P.DIEPTE / 2 - 6) if k < 9 else rnd.uniform(P.DIEPTE / 2 + 6, P.DIEPTE - 9)
        if kant == "rechts":
            s = schroef(rnd.choice((8.0, 10.0, 12.0)))
            s = s.rotate([90, 0, rnd.uniform(0, 360)]).translate([x, y, z + 2.8])
        else:
            s = moer().rotate([0, 0, rnd.uniform(0, 60)]).translate([x, y, z])
            x = -x
            s = s.mirror([1, 0, 0])
        out.append(s)
    return out


def scene(view, snel):
    theta, cam, doel, lens, res = VIEWS[view]
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN)
    staal = BL.metaal()
    explo = view == "explosie"
    dx, dz = P.verschuiving(theta)
    for naam, (kleur, m) in pod.stand(theta).items():
        if explo:
            m = m.translate(explosie(naam))
        BL.voeg_toe(naam, m, mats[kleur], afronding=0.25 if naam in ("voet", "dak") else 0.0)
    # rekwisieten: bits in het dak, schroefjes en moertjes in de bakken
    top = P.DAK_Z0 + P.DAK_T + P.KAM_H
    for i, (x, y) in enumerate(P.BITS):
        b = bit().translate([x, y, top - P.BIT_DIEP + 0.2])
        if explo:
            b = b.translate(explosie("dak"))
        BL.voeg_toe(f"bit {i}", b, staal)
    if not explo:
        rnd = random.Random(7)
        for kant, s in (("rechts", 1), ("links", -1)):
            for j, m in enumerate(lading(kant, rnd)):
                BL.voeg_toe(f"lading {kant} {j}", m.translate([s * dx, 0, dz]), staal)
    BL.studio(muur_y=0.40)
    BL.camera(cam, doel, lens=lens, diafragma=8.0 if explo else 5.6)
    pad = UIT / f"{view}.png"
    BL.render(pad, res, samples=96, snel=snel)
    return pad


def explosie(naam):
    """Verschuiving (mm) per onderdeel in de explosietekening."""
    if naam == "dak":
        return [0, 0, 55]
    if naam == "schuif":
        return [0, -28, 95]
    if naam.startswith("bak"):
        s = 1 if "rechts" in naam else -1
        return [s * 30, 0, 22]
    if "arm" in naam:
        s = 1 if "rechts" in naam else -1
        return [s * 10, -24 if "voor" in naam else 24, 8]
    if naam.startswith("pen"):
        s = 1 if "rechts" in naam else -1
        bak = " bak " in naam
        return [s * (40 if bak else 10), -40 if naam.endswith(" v") else 40, 30 if bak else 8]
    return [0, 0, 0]


def reeks():
    from PIL import Image
    ims = [Image.open(UIT / f"{v}.png").convert("RGB") for v in ("dicht", "half", "open")]
    h = min(im.height for im in ims)
    ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
    out = Image.new("RGB", (sum(im.width for im in ims), h), "white")
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width
    out.thumbnail((2400, 1000))
    out.save(UIT / "reeks.png")
    print("reeks klaar:", UIT / "reeks.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    namen = [a for a in argv if not a.startswith("--")] or ["dicht"]
    if namen == ["alle"]:
        namen = list(VIEWS) + ["reeks"]
    UIT.mkdir(exist_ok=True)
    for v in namen:
        if v == "reeks":
            reeks()
        else:
            scene(v, snel)
