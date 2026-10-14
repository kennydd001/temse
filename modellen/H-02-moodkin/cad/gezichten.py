"""De kassen in de schil en de vier gezichten op de trommel, als gaten in een rechtopstaande wand.

Alles is getekend zoals je het van voren ziet: u = breedte (0 = midden, +u = rechts als je ervoor staat), v =
hoogte vanaf de onderkant van de band Z_GEZICHT. De gaten worden recht naar achteren (langs y) door de wand
gesneden, dus van voren zie je de tekening onvervormd.

De schil heeft één vast pompoengezicht: twee spitsboog-oogkassen, een driehoekige neuskas en een grijns van
vijf tandkolommen. Door die kassen zie je de gele trommel; elk gezicht is een set gaten in de trommel die
helemaal binnen de kassen valt (met MARGE). Overdag zie je het gezicht in het geel, 's nachts geeft alleen dat
gezicht licht.

Alleen vormen die rechtop zonder support printen: driehoeken en spitsbogen met de punt omhoog, en platte
plafonds tot BRUG mm (de printer overbrugt ze). Een tand die van boven in een mond hangt, kan niet: die zou in
de lucht beginnen. Daarom bestaat de grijns uit kolommen met tanden ertussen die op de onderlip staan.
"""
import math
from functools import lru_cache

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
from gereedschap.geom import sym
from gereedschap.zelfdragend import zelfdragend

B = P.GEZICHT_B / 2
H = P.Z_GEZICHT[1] - P.Z_GEZICHT[0]
MARGE = 0.6             # een gat in de trommel blijft zoveel binnen de rand van de kas

# --- kassen in de schil ----------------------------------------------------------------------------
OOG_U = (6.0, 28.0)     # oogkas: spitsboog tussen deze |u|
OOG_V = 12.5            # aanzet van de boog
NEUS = Polygon([(-5.0, 9.6), (5.0, 9.6), (0.0, 18.3)])
MOND_U = (-20.0, -10.0, 0.0, 10.0, 20.0)   # vijf tandkolommen; tussen twee kolommen staat een tand van 2 mm
MOND_B = 8.0
MOND_H = 7.5
MOND_R = 2.0


def mond_v(u):
    """Onderkant van de kolom op breedte u: een lichte glimlach."""
    return 0.5 + 0.006 * u * u


def spitsboog(u0, u1, v0, n=48):
    """Gelijkzijdige spitsboog (gotisch venster) tussen u0 en u1, aanzet op v0."""
    s = u1 - u0
    top = math.atan2(math.sqrt(3) / 2 * s, s / 2)
    links = [(u1 - s * math.cos(a), v0 + s * math.sin(a)) for a in [top * i / n for i in range(n + 1)]]
    rechts = [(u0 + s * math.cos(a), v0 + s * math.sin(a)) for a in [top * i / n for i in range(n, -1, -1)]]
    return Polygon(links + rechts[1:])


def _tand(uc, v0, v1, b=8.0, r=3.0):
    """Gat met een plat plafond (korte brug) en een ronde onderkant."""
    r = min(r, b / 2, v1 - v0)
    g = unary_union([box(uc - b / 2, v0 + r, uc + b / 2, v1), box(uc - b / 2 + r, v0, uc + b / 2 - r, v0 + r),
                     Point(uc - b / 2 + r, v0 + r).buffer(r, quad_segs=12),
                     Point(uc + b / 2 - r, v0 + r).buffer(r, quad_segs=12)])
    return g.intersection(box(uc - b / 2, v0, uc + b / 2, v1))     # een lage tand: de bochten niet boven het plafond


@lru_cache(maxsize=None)
def kassen():
    """De gaten vooraan in de schil. De top van een spitsboog is te vlak om zonder support te printen (de bogen
    raken elkaar onder 120°); printbaar() maakt er een rechte spits van."""
    oog = spitsboog(-OOG_U[1], -OOG_U[0], OOG_V)
    mond = [_tand(u, mond_v(u), mond_v(u) + MOND_H, MOND_B, MOND_R) for u in MOND_U]
    return printbaar(unary_union([sym(oog), NEUS] + mond))


# --- gezichten op de trommel -----------------------------------------------------------------------
OOG_M = -(OOG_U[0] + OOG_U[1]) / 2      # midden van de linker oogkas


def _kolom(k, onder, boven, b=6.4, r=2.4):
    """Gat in mondkolom k, van `onder` tot `boven` mm boven de bodem van de kolom."""
    u = MOND_U[k]
    return _tand(u, mond_v(u) + onder, mond_v(u) + boven, b, r)


def _driehoek(uc, v0, b, h):
    return Polygon([(uc - b / 2, v0), (uc + b / 2, v0), (uc, v0 + h)])


def blij():
    """Klassieke pompoen: driehoeksogen, driehoeksneus, een lach met vijf tandgaten."""
    oog = _driehoek(OOG_M, 13.5, 18.0, 16.0)
    neus = _driehoek(0.0, 10.4, 7.6, 6.2)
    lach = [_kolom(k, 0.8, 6.7) for k in range(5)]
    return unary_union([sym(oog), neus] + lach)


def boos():
    """Boos: schuine ogen die naar de neus toe aflopen, een kleine neus, een zaagtandmond."""
    oog = Polygon([(OOG_M - 9.0, 13.5), (OOG_M + 6.5, 13.5), (OOG_M - 2.5, 26.0)])
    neus = _driehoek(0.0, 10.4, 6.0, 4.4)
    mond = [_driehoek(u, mond_v(u) + 1.6, 6.2, 5.1) for u in MOND_U]
    return unary_union([sym(oog), neus] + mond)


def bang():
    """Bang: grote hoge ogen, een klein neusje, een open O-mond."""
    oog = _tand(OOG_M, 13.5, 24.0, b=11.0, r=5.5)
    neus = _driehoek(0.0, 10.6, 5.0, 3.8)
    mond = _kolom(2, 0.8, 6.7, b=6.0, r=3.0)
    return unary_union([sym(oog), neus, mond])


def knipoog():
    """Knipoog: één driehoeksoog, één dichtgeknepen oog (^), een scheve grijns die rechts omhoog trekt."""
    oog = _driehoek(OOG_M, 13.5, 18.0, 16.0)
    m = -OOG_M
    dicht = Polygon([(m - 7, 15), (m, 24), (m + 7, 15), (m + 3.3, 15), (m, 19.2), (m - 3.3, 15)])
    neus = _driehoek(0.0, 10.4, 7.0, 5.2)
    grijns = [_kolom(k, 0.8 + 0.95 * k, 3.4 + 0.83 * k, r=1.2) for k in range(5)]
    return unary_union([oog, dicht, neus] + grijns)


VORMEN = {"blij": blij, "boos": boos, "bang": bang, "knipoog": knipoog}


def _wand():
    return box(-B - 8, -6, B + 8, H + 8)


def printbaar(gaten):
    """Gaten zelfdragend maken: de wand eromheen zelfdragend, en wat daarvan overblijft is het gat."""
    wand = _wand()
    vol = zelfdragend(wand.difference(gaten), P.RES, P.OVERHANG, brug=P.BRUG)
    return wand.difference(vol)


@lru_cache(maxsize=None)
def gezicht(naam):
    """Gaten van één gezicht, zoals ze in de trommel komen (u, v)."""
    return printbaar(VORMEN[naam]())

