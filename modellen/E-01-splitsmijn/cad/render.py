"""Productfoto's van de pasproef van E-01 Splitsmijn (Blender/Cycles, zonder scherm).

Gebruik:  python render.py [voor|binnen|mes|explosie|alle] [--snel]
Schrijft naar renders/. 'voor' toont een ronde halfweg: 8 = 3 + 5, met nog een muntje onderweg; 'binnen' is
dezelfde ronde zonder voorplaat en beugel.
"""
import random
import sys
from pathlib import Path

import _pad  # noqa: F401
import params as P
import valbord as V
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"
OP_TAFEL = -P.VOET_Z[0]          # de voet staat op de tafel
VLOER = P.VOET_Z[0] + P.VOET_WAND

VIEWS = {
    # naam: (slingerhoek, camera, doel, lens, resolutie)
    "voor": (22.0, (0.34, -0.78, 0.36), (0.0, 0.0, 0.125), 62, (1100, 1250)),
    "binnen": (22.0, (0.20, -0.80, 0.30), (0.0, 0.0, 0.125), 62, (1100, 1250)),
    "mes": (-12.0, (0.12, -0.17, 0.34), (0.0, -0.014, 0.214), 85, (1200, 900)),
    "explosie": (0.0, (0.92, -0.58, 0.46), (0.0, -0.07, 0.14), 56, (1400, 1100)),
}


def explosie(naam):
    """Verschuiving in mm (vóór het omdraaien: +y is de kijkkant)."""
    return {"achterplaat": [0, -70, 0], "plank": [0, 0, 75], "voorplaat": [0, 60, 0], "beugel": [0, 115, 0],
            "slinger": [0, 175, 0], "mesblok": [0, 215, 0], "voet": [0, 0, -60]}.get(naam, [0, 0, 0])


def munten_in_bak(kant, n, rnd):
    """n muntjes plat in een bakje (kant +1 = links voor het kind, +x); wat niet naast elkaar past, ligt erop."""
    plek = [(kant * x, rnd.uniform(-1, 13), VLOER + 0.1) for x in (24.0, 51.5, 79.0)[:n]]
    for i in range(n - len(plek)):
        a, b, _ = plek[i % len(plek)]
        c, d, _ = plek[(i + 1) % len(plek)]
        plek.append(((a + c) / 2, (b + d) / 2, VLOER + 0.1 + P.MUNT_T))
    return [V.munt_plat(0, 0, 0).rotate([0, 0, rnd.uniform(0, 360)]).translate(p) for p in plek]


def scene(view, snel):
    hoek, cam, doel, lens, res = VIEWS[view]
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN)
    st = V.stand(hoek)
    delen = [(naam, kleur, m) for naam, (kleur, m) in st.items()]
    if view == "binnen":
        delen = [d for d in delen if d[0] not in ("voorplaat", "beugel")]
    if view in ("voor", "binnen"):
        # 8 = 3 + 5 halfweg: links 3, rechts 4, een muntje rolt van de plank (die rechts lager ligt), een ander
        # valt rechts tussen de pinnen
        rnd = random.Random(3)
        op_plank = V.draai_y(V.munt_op(-9.0, P.Z_PLANK + P.PLANK_H / 2 + P.MUNT_D / 2 + 0.1), hoek)
        delen += [("munt plank", P.BLAUW, op_plank), ("munt pinnen", P.BLAUW, V.munt_op(-44.0, P.rij_z(2) + 19.0))]
        delen += [(f"bak L {i}", P.BLAUW, m) for i, m in enumerate(munten_in_bak(1, 3, rnd))]
        delen += [(f"bak R {i}", P.BLAUW, m) for i, m in enumerate(munten_in_bak(-1, 4, rnd))]
    if view == "explosie":
        delen += [(f"munt {i}", P.BLAUW, V.munt_plat(-60 + 30 * i, 90, P.VOET_Z[0] - 60)) for i in range(5)]
    for naam, kleur, m in delen:
        if view == "explosie":
            m = m.translate(explosie(naam))
        m = m.rotate([0, 0, 180]).translate([0, 0, OP_TAFEL + (60 if view == "explosie" else 0)])
        BL.voeg_toe(naam, m, mats[kleur])
    BL.studio(muur_y=0.30)
    BL.camera(cam, doel, lens=lens, diafragma=8.0 if view == "explosie" else 5.6)
    pad = UIT / f"{view}.png"
    BL.render(pad, res, samples=96, snel=snel)
    return pad


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    namen = [a for a in argv if not a.startswith("--")] or ["voor"]
    if namen == ["alle"]:
        namen = list(VIEWS)
    UIT.mkdir(exist_ok=True)
    for v in namen:
        scene(v, snel)
