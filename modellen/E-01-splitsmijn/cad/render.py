"""Productfoto's van E-01 Splitsmijn (Blender/Cycles, zonder scherm): het eindmodel en de pasproef.

Gebruik:  python render.py [naam ...|eind|proef|alle] [--snel]
Schrijft naar renders/.
  eindmodel: splitsmijn (8 = 3 + 5: beide karretjes uit, de muntjes op het tienveld), splitsmijn_binnen (zonder
             voorplaat en mijnhout), splitsmijn_explosie, splitsmijn_kar (een karretje van dichtbij)
  pasproef:  voor (een ronde halfweg: 8 = 3 + 5, met nog een muntje onderweg), binnen (zonder voorplaat en
             beugel), mes, explosie, passtrip
"""
import random
import sys
from pathlib import Path

import _pad  # noqa: F401
import mijn as M
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
    "passtrip": (0.0, (0.06, -0.20, 0.20), (0.0, -0.005, 0.0), 70, (1200, 800)),
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
    if view == "passtrip":
        # de drie stukjes zoals ze van het bed komen
        delen = [("gatenstrip", P.ZWART, V.gatenstrip().translate([-55, -10, 0])),
                 ("pennenblok", P.ZWART, V.pennenblok().translate([8, 14, 0])),
                 ("beugelproef", P.ZWART, V.beugelproef().translate([12, -24, 0]))]
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
        if view != "passtrip":
            m = m.rotate([0, 0, 180]).translate([0, 0, OP_TAFEL + (60 if view == "explosie" else 0)])
        BL.voeg_toe(naam, m, mats[kleur])
    BL.studio(muur_y=0.30)
    BL.camera(cam, doel, lens=lens, diafragma=8.0 if view == "explosie" else 5.6)
    pad = UIT / f"{view}.png"
    BL.render(pad, res, samples=96, snel=snel)
    return pad


# --- eindmodel ------------------------------------------------------------------------------------

TAFEL = P.BERG_H                 # de mijnberg staat op de tafel
VIEWS_EIND = {
    # naam: (slingerhoek, karretjes uit (L, R), camera, doel, lens, resolutie)
    "splitsmijn": (18.0, (62.0, 40.0), (0.36, -0.86, 0.40), (0.005, -0.04, 0.125), 61, (1100, 1300)),
    "splitsmijn_binnen": (18.0, (0.0, 0.0), (0.26, -0.84, 0.36), (0.0, 0.0, 0.16), 55, (1100, 1300)),
    "splitsmijn_explosie": (0.0, (0.0, 0.0), (0.98, -0.78, 0.58), (0.0, -0.09, 0.205), 56, (1400, 1150)),
    "splitsmijn_kar": (-15.0, (70.0, 0.0), (0.03, -0.34, 0.21), (-0.05, -0.085, 0.012), 58, (1200, 900)),
}


def munt_eind(mats, kleur_boven, x, y, z, hoek, naam):
    """Tweekleurig muntje plat, met kleur_boven ('blauw' of 'geel') naar boven."""
    geel, blauw = M.munt_delen()
    delen = [(geel, P.GEEL), (blauw, P.BLAUW)]
    out = []
    for m, k in delen:
        if kleur_boven == "geel":
            m = m.translate([0, 0, -P.MUNT_T / 2]).rotate([180, 0, 0]).translate([0, 0, P.MUNT_T / 2])
        out.append((m.rotate([0, 0, hoek]).translate([x, y, z]), k))
    return [(f"{naam} {k}", k, m) for m, k in out]


def munten_in_kar(s, uit, n, rnd, mats):
    """n muntjes plat op de bodem van een karretje (kant s) dat `uit` mm uitgetrokken is; elk met zijn eigen kleur
    boven: zo vallen ze."""
    x0, y0, z0, x1, y1, z1 = M.kar_maat(s)
    vloer = z0 + P.KAR_WAND + 0.05
    out = []
    plekken = []
    while len(plekken) < n:
        x = rnd.uniform(min(x0, x1) + 22, max(x0, x1) - 22)
        y = rnd.uniform(y0 + 16, y0 + 70) + uit
        if all((x - a) ** 2 + (y - b) ** 2 > 27 ** 2 for a, b in plekken):
            plekken.append((x, y))
    for i, (x, y) in enumerate(plekken):
        out += munt_eind(mats, rnd.choice(("blauw", "geel")), x, y, vloer, rnd.uniform(0, 360), f"kar{s} {i}")
    return out


def tienveld_met(links, rechts, plek, mats, hoek=0.0):
    """Tienveld op plek (x, y, z), `hoek` graden gedraaid, met eerst `links` muntjes blauw boven, dan `rechts`
    geel boven."""
    out = [("tienveld", P.TERRA, M.tienveld())]
    st = P.TIEN_STAP
    vakjes = [(i, j) for j in range(2) for i in range(5)]
    for k, (i, j) in enumerate(vakjes[:links + rechts]):
        kleur = "blauw" if k < links else "geel"
        mx = -(i - 2) * st                       # het kind kijkt van +y: de eerste kolom links voor het kind (+x)
        my = (j - 0.5) * st                      # de eerste rij is de bovenste: het verst van het kind
        out += munt_eind(mats, kleur, mx, my, P.TIEN_T[0] + 0.05, 20 * k, f"tien {k}")
    return [(n, k, m.rotate([0, 0, hoek]).translate(list(plek))) for n, k, m in out]


def explosie_eind(naam):
    return {"achterplaat": [0, -80, 0], "plank": [0, 0, 70], "voorplaat": [0, 55, 0], "mijnhout": [0, 95, 0],
            "beugel": [0, 150, 0], "steel": [0, 200, 0], "kop": [0, 245, 0], "mesblok": [0, 250, 0],
            "portaal": [0, 0, 70], "letters": [0, 0, 70], "mijnberg": [0, 0, -70], "verdelervin": [0, 0, -25],
            "kokervoet L": [0, 0, -35], "kokervoet R": [0, 0, -35], "kokerkap L": [0, 0, 38], "kokerkap R": [0, 0, 38],
            "karretje L": [0, 110, -70], "karretje R": [0, 110, -70]}.get(naam, [0, 0, 0])


def scene_eind(view, snel):
    hoek, uit, cam, doel, lens, res = VIEWS_EIND[view]
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN_EIND)
    st = M.stand(hoek, uit)
    delen = [(naam, kleur, m) for naam, (kleur, m) in st.items()]
    rnd = random.Random(8)
    if view == "splitsmijn_binnen":
        delen = [d for d in delen if d[0] not in ("voorplaat", "mijnhout", "beugel")]
    if view in ("splitsmijn", "splitsmijn_binnen"):
        # een muntje rolt van de plank, een ander valt tussen de pinnen
        op_plank = V.draai_y(V.munt_op(-9.0, P.Z_PLANK + P.PLANK_H / 2 + P.MUNT_D / 2 + 0.1), hoek)
        delen += [("munt plank", P.BLAUW, op_plank), ("munt pinnen", P.BLAUW, V.munt_op(44.0, P.rij_z(1) + 21.0))]
    if view == "splitsmijn":
        delen += munten_in_kar(1, uit[0], 3, rnd, mats) + munten_in_kar(-1, uit[1], 5, rnd, mats)
        # het tienveld ligt los op tafel, vóór de uitgetrokken karretjes
        diepte = 2 * P.TIEN_STAP + 2 * P.TIEN_RAND
        delen += tienveld_met(3, 5, (-8.0, P.BERG_Y[1] + max(uit) + 14 + diepte / 2, -P.BERG_H), mats, hoek=-4.0)
    if view == "splitsmijn_kar":
        delen += munten_in_kar(1, uit[0], 3, rnd, mats)
    if view == "splitsmijn_explosie":
        delen = [(n, k, m.translate(explosie_eind(n))) for n, k, m in delen]
        delen += munt_eind(mats, "blauw", 60, 150, -P.BERG_H - 70, 0, "losse munt a")
        delen += munt_eind(mats, "geel", 90, 150, -P.BERG_H - 70, 30, "losse munt b")
    for naam, kleur, m in delen:
        BL.voeg_toe(naam, m.rotate([0, 0, 180]).translate([0, 0, TAFEL + (70 if view == "splitsmijn_explosie" else 0)]),
                    mats[kleur])
    BL.studio(muur_y=0.85)
    BL.camera(cam, doel, lens=lens, diafragma=8.0 if view == "splitsmijn_explosie" else 6.3)
    pad = UIT / f"{view}.png"
    BL.render(pad, res, samples=96, snel=snel)
    return pad


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    namen = [a for a in argv if not a.startswith("--")] or ["splitsmijn"]
    if namen == ["alle"]:
        namen = list(VIEWS_EIND) + list(VIEWS)
    elif namen == ["eind"]:
        namen = list(VIEWS_EIND)
    elif namen == ["proef"]:
        namen = list(VIEWS)
    UIT.mkdir(exist_ok=True)
    for v in namen:
        scene_eind(v, snel) if v in VIEWS_EIND else scene(v, snel)
