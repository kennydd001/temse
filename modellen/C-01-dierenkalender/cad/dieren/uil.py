"""Uil: Terracotta kop met oorpluimen op een Latte Brown bord, lichte gezichtsschijf, witte oogringen."""
from shapely.geometry import Polygon
from shapely.ops import unary_union

import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import arc_points, ellipse, smooth, stroke, sym
from params import LATTE, TERRA, WIT, ZWART

from .kit import Dier, Tekening, detail, driehoek, lach_ogen, poot, relief, vlak


def tekening():
    kop = ellipse(0, 139, 54, 43)
    pluim = smooth(Polygon([(-50, 160), (-20, 176), (-60, 193)]), open_=4)
    vorm = smooth(unary_union([kop, sym(pluim)]), close=7)

    schijf = smooth(unary_union([ellipse(-21, 138, 26, 25), ellipse(21, 138, 26, 25), ellipse(0, 122, 18, 14)]),
                    close=4).intersection(vorm.buffer(-2.5))
    ringen = sym(ellipse(-21, 139, 15.5, 15.5))
    ogen = lach_ogen(21, 141, r=7.0, w=3.2, wimper=False)
    snavel = driehoek((-6.5, 129), (6.5, 129), (0, 116), afronding=2.2)

    # veertjes op het voorhoofd: kleine v'tjes in de lichte kleur
    v = lambda x, y: stroke([(x - 3.5, y + 2.2), (x, y), (x + 3.5, y + 2.2)], 1.8)  # noqa: E731
    veren = unary_union([v(0, 170), v(-10, 166), v(10, 166)])
    # buikveertjes op de vleugelaanzet links en rechts
    boogjes = sym(unary_union([stroke(arc_points(-44, 112 - 7 * i, 5, 200, 340), 1.6) for i in range(2)]))

    voet = unary_union([ellipse(-30 + dx, 98, 4.6, 7.0) for dx in (-7.5, 0, 7.5)])
    voeten = smooth(sym(voet), close=1.0)
    return Tekening(
        silhouet=unary_union([vorm, voeten]),
        lagen=[
            relief(TERRA, vorm),
            vlak(LATTE, unary_union([schijf, veren, boogjes.intersection(vorm.buffer(-1))])),
            vlak(WIT, ringen),
            detail(ZWART, ogen),
            detail(TERRA, snavel),
            poot(TERRA, voeten),
        ])


DIER = Dier("uil", 3, {"nl": "uil", "en": "owl"}, bord=LATTE, kop=TERRA, decor=(TERRA, "blad"), tekening=tekening)
