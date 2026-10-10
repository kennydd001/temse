"""Koe: Ivory White kop op een Hot Pink bord, Charcoal vlekken, roze snuit, groene blaadjes."""
from shapely.ops import unary_union

import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import arc_points, ellipse, smooth, stroke, sym
from params import GROEN, ROZE, WIT, ZWART

from .kit import Dier, Tekening, blos, detail, hoefje, lach_ogen, op_poot, poot, relief, vlak

HOEF_X = 36.0


def tekening():
    kop = smooth(unary_union([ellipse(0, 142, 48, 39), ellipse(0, 113, 38, 22)]), close=10)
    kuif = unary_union([ellipse(-11, 178, 10, 9), ellipse(0, 183, 10.5, 10), ellipse(11, 178.5, 9.5, 8.5)])
    oor = ellipse(-60, 158, 21, 10.5, rot=-18)
    oren = sym(oor)
    vorm = smooth(unary_union([kop, kuif, oren]), close=5)

    binnen_oor = sym(ellipse(-62, 157.5, 13, 5.5, rot=-18))
    snuit = ellipse(0, 112, 31, 16.5)
    wangen = blos(35, 127, 6, 3.8)

    # eigen vlekkenpatroon: grote vlek over het linkeroor, kleine op het voorhoofd rechts
    vlek_l = unary_union([ellipse(-44, 160, 15, 12, rot=25), ellipse(-34, 168, 9, 7), ellipse(-52, 150, 8, 6)])
    vlek_r = unary_union([ellipse(27, 167, 9, 7, rot=-20), ellipse(33, 160, 5, 4.5)])
    vlekken = smooth(unary_union([vlek_l, vlek_r]), open_=1.0).intersection(vorm.buffer(-0.6))
    vlekken = vlekken.difference(binnen_oor.buffer(1.2))

    ogen = lach_ogen(22, 140)
    neusgaten = sym(ellipse(-10.5, 114, 3.6, 2.6, rot=15))
    mond = stroke(arc_points(0, 108, 7, 205, 335), 1.8)

    hoef, band, _ = hoefje(HOEF_X, cy=100.0, w=25, h=21, r=7.5)
    return Tekening(
        silhouet=unary_union([vorm, hoef]),
        lagen=[
            relief(WIT, vorm),
            vlak(ROZE, unary_union([binnen_oor, snuit, wangen])),
            vlak(ZWART, vlekken),
            detail(ZWART, unary_union([ogen, neusgaten, mond])),
            poot(WIT, hoef),
            op_poot(ZWART, band),
        ])


DIER = Dier("koe", 2, {"nl": "koe", "en": "cow"}, bord=ROZE, kop=WIT, decor=(GROEN, "blad"), tekening=tekening)
