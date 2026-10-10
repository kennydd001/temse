"""Productfoto's van B-34A Taakvreter (Blender/Cycles, zonder scherm).

Gebruik:  python render.py [voor|zij|doorsnede|explosie|ogen|alle] [--snel]
Schrijft naar renders/. 'ogen' maakt twee close-ups (kruk op 90° en 270°) en zet ze naast elkaar.
Het memoblaadje en de O-ringen zijn rekwisieten (niet om te printen).
"""
import sys
from pathlib import Path

from manifold3d import Manifold

import _pad  # noqa: F401
import params as P
import vreter as V
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"

VIEWS = {
    # naam: (krukhoek, camera, doel, lens, resolutie, briefje: hoe ver in de mond (mm voorbij de lip))
    "voor": (20.0, (0.26, -0.40, 0.24), (0.0, 0.040, 0.068), 58, (1200, 1100), 26.0),
    "zij": (65.0, (0.42, -0.10, 0.20), (0.02, 0.045, 0.062), 60, (1200, 1000), None),
    "doorsnede": (40.0, (0.40, 0.02, 0.10), (0.0, 0.045, 0.060), 62, (1200, 1100), 48.0),
    "explosie": (20.0, (0.40, -0.48, 0.30), (0.0, 0.04, 0.075), 44, (1400, 1100), None),
    "ogen op": (90.0, (0.05, -0.30, 0.13), (0.0, 0.0, 0.092), 95, (1000, 700), None),
    "ogen neer": (270.0, (0.05, -0.30, 0.13), (0.0, 0.0, 0.092), 95, (1000, 700), None),
}


def briefje(diep):
    """Geel memoblaadje van 76 mm, met de voorrand `diep` mm voorbij de voorkant van de lip, in het papiervlak."""
    y1 = -P.LIP_T + diep
    return Manifold.cube([P.PAPIER, P.PAPIER, 0.3]).translate([-P.PAPIER / 2, y1 - P.PAPIER, P.NIP_Z - 0.15])


def explosie(naam):
    if naam == "romp voor" or naam.startswith("paspen ") and "hoorn" not in naam:
        return [0, -75, 0]
    if naam in ("mondstuk",):
        return [0, -120, 0]
    if naam == "tanden":
        return [0, -135, 0]
    if naam == "lade":
        return [0, -75, -40]
    if naam == "romp achter":
        return [0, 70, 0]
    if naam in ("kruk", "knop"):
        return [35 if naam == "kruk" else 55, 0, 0]
    if naam.startswith(("hoorn", "paspen hoorn")):
        return [0, -75, 35 if naam.startswith("hoorn") else 20]
    if naam.startswith(("ogen", "pupil", "tuimelaar")):
        return [0, -30, 0]
    return [0, 0, 0]


def scene(view, snel):
    phi, cam, doel, lens, res, diep = VIEWS[view]
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN)
    mats[V.RUBBER] = BL.pla("rubber", "#3A2424", 0.8)
    st = V.stand(phi)
    snij = Manifold.cube([200, 300, 300]).translate([-200, -150, -50]) if view == "doorsnede" else None
    for naam, (kleur, m) in st.items():
        if view == "explosie":
            m = m.translate(explosie(naam))
        if snij is not None:
            m = m ^ snij
            if m.is_empty():
                continue
        BL.voeg_toe(naam, m, mats[kleur], afronding=0.3 if naam.startswith("romp") else 0.0)
    if diep is not None:
        b = briefje(diep)
        if snij is not None:
            b = b ^ snij
        BL.voeg_toe("briefje", b, BL.pla("briefje", "#FFE45C", 0.7))
    BL.studio(muur_y=0.45)
    BL.camera(cam, doel, lens=lens, diafragma=8.0 if view in ("explosie", "doorsnede") else 5.6)
    pad = UIT / f"{view.replace(' ', '_')}.png"
    BL.render(pad, res, samples=96, snel=snel)
    return pad


def ogen():
    from PIL import Image
    ims = [Image.open(UIT / f"{v}.png").convert("RGB") for v in ("ogen_op", "ogen_neer")]
    h = min(im.height for im in ims)
    out = Image.new("RGB", (sum(im.width for im in ims), h), "white")
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width
    out.save(UIT / "ogen.png")
    print("ogen klaar:", UIT / "ogen.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    namen = [a for a in argv if not a.startswith("--")] or ["voor"]
    if namen == ["alle"]:
        namen = ["voor", "zij", "doorsnede", "explosie", "ogen"]
    UIT.mkdir(exist_ok=True)
    for v in namen:
        if v == "ogen":
            scene("ogen op", snel)
            scene("ogen neer", snel)
            ogen()
        else:
            scene(v, snel)
