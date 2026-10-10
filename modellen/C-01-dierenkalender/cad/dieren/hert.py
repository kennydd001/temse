"""Hert (reekalfje): Latte Brown kop op een Terracotta bord, witte stippen, snuit en binnenoren."""
from shapely.ops import unary_union

from geom import arc_points, ellipse, smooth, stroke, sym
from params import LATTE, TERRA, WIT, ZWART

from .kit import Dier, Tekening, blos, detail, hoefje, lach_ogen, op_poot, poot, relief, vlak


HOEF_X = 33.0


def tekening():
    kop = smooth(unary_union([ellipse(0, 146, 41, 35), ellipse(0, 117, 25, 19)]), close=14)
    oor = ellipse(-50, 165, 25, 11.5, rot=-32)
    oren = sym(oor)
    binnen = sym(ellipse(-51, 165.5, 16, 6.2, rot=-32))
    vorm = smooth(unary_union([kop, oren]), close=6)

    stippen = unary_union([ellipse(x, y, r, r) for x, y, r in
                           ((0, 171, 3.0), (-9, 166, 2.4), (9, 166, 2.4), (-5, 176, 2.0), (5, 176, 2.0))])
    snuit = ellipse(0, 112, 18, 12)
    neus = smooth(ellipse(0, 121, 7.5, 5.2), open_=0.5)
    mond = stroke(arc_points(0, 108, 5, 210, 330), 1.7)
    ogen = lach_ogen(19, 143, r=7, w=3.0)

    hoef, band, _ = hoefje(HOEF_X, cy=99.5, w=22, h=20, r=7.0)
    return Tekening(
        silhouet=unary_union([vorm, hoef]),
        lagen=[
            relief(LATTE, vorm),
            vlak(WIT, unary_union([binnen, stippen, snuit])),
            vlak(TERRA, blos(27, 128, 5.5, 3.6)),
            detail(ZWART, unary_union([ogen, neus, mond])),
            poot(LATTE, hoef),
            op_poot(ZWART, band),
        ])


DIER = Dier("hert", 7, {"nl": "hert", "en": "deer"}, bord=TERRA, kop=LATTE, decor=(WIT, "blad"), tekening=tekening)
