"""Leeuw: Terracotta manen, Sunflower Yellow gezicht, Latte Brown snuit en bord."""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import ellipse, smooth, sym
from params import GEEL, LATTE, TERRA, ZWART

from .kit import (Dier, Tekening, detail, driehoek, lach_mond, lach_ogen, op_poot, poot, poot_ovaal, relief,
                  snor_stipjes, teen_lijnen, vlak)

POOT_X = 36.0


def _manen(cx=0, cy=143, R=50, n=14, r=13):
    lokken = [ellipse(cx + R * np.cos(a), cy + R * 0.92 * np.sin(a), r, r)
              for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2]
    manen = smooth(unary_union(lokken + [ellipse(cx, cy, R, R * 0.92)]), close=2.5)
    # onderkant niet te ver over het bord: daar schuiven de kaartjes omhoog
    return smooth(manen.intersection(box(-120, 93, 120, 260)), open_=3)


def tekening():
    manen = _manen()
    gezicht = smooth(unary_union([ellipse(0, 140, 38, 34), ellipse(0, 119, 27, 17)]), close=6)
    oren = sym(ellipse(-31, 170, 10.5, 10.5))
    binnen_oor = sym(ellipse(-31, 170, 5.5, 5.5))
    vorm = unary_union([manen, oren])

    snuit = smooth(unary_union([ellipse(-8.5, 118, 11, 9), ellipse(8.5, 118, 11, 9), ellipse(0, 112, 9, 6)]),
                   close=2)
    neus = driehoek((-7, 128), (7, 128), (0, 121), afronding=2.5)
    gezichtje = unary_union([lach_ogen(17, 144, r=6.5, w=3.0), neus, lach_mond(cy=117, r=3.8, steel=4.5),
                             snor_stipjes(cx=12.0, cy=119.5, r=0.85)])

    poten = poot_ovaal(POOT_X, rx=15, ry=10.5)
    return Tekening(
        silhouet=unary_union([vorm, poten]),
        lagen=[
            relief(TERRA, vorm),
            relief(GEEL, gezicht),
            vlak(GEEL, oren),
            vlak(LATTE, unary_union([snuit, binnen_oor])),
            detail(ZWART, gezichtje),
            poot(GEEL, poten),
            op_poot(ZWART, teen_lijnen(POOT_X)),
        ])


DIER = Dier("leeuw", 6, {"nl": "leeuw", "en": "lion"}, bord=LATTE, kop=TERRA, decor=(TERRA, "blad"),
            tekening=tekening)
