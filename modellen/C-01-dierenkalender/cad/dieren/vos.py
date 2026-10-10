"""Vos: Pumpkin Orange, Ivory White, Charcoal, Mistletoe Green."""
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geom import ellipse, smooth, sym
from params import GROEN, ORANJE, WIT, ZWART

from .kit import (Dier, Tekening, detail, driehoek, lach_mond, lach_ogen, op_poot, poot, poot_ovaal, relief,
                  snor_stipjes, teen_lijnen, vlak)

PAW_X = 37.0


def _kop():
    dome = ellipse(0, 141, 55, 43)
    wang = Polygon([(-24, 146), (-76, 117), (-70, 111), (-48, 103), (-20, 99)])
    kin = ellipse(0, 108, 30, 13)
    return smooth(unary_union([dome, sym(wang), kin]), close=8, open_=3)


def _oren():
    return sym(smooth(Polygon([(-53, 156), (-13, 177), (-66, 206)]), open_=5))


def tekening():
    kop, oren = _kop(), _oren()
    omtrek = smooth(unary_union([kop, oren]), close=6)
    poten = poot_ovaal(PAW_X)

    binnen = sym(smooth(Polygon([(-49, 161), (-22, 175), (-61, 197)]), open_=3.5)).intersection(oren.buffer(-3.2))
    punt = sym(ellipse(-66, 206, 19.0, 19.0)).intersection(oren)

    # witte wangen en snuit; het oranje voorhoofd loopt in een V naar de neus
    wang = ellipse(-33, 118, 36, 19, rot=-18)
    onder = Polygon([(-90, 121), (-60, 119), (0, 104), (60, 119), (90, 121), (90, 80), (-90, 80)])
    masker = unary_union([sym(wang), onder, ellipse(0, 114, 21, 14)])
    masker = masker.difference(smooth(Polygon([(-15, 160), (15, 160), (4.5, 124), (-4.5, 124)]), open_=2.5))
    masker = smooth(masker.intersection(kop.buffer(-0.01)), open_=1.5)

    glim = ellipse(-2.6, 122.2, 1.4, 1.2)
    neus = driehoek((-7.5, 124.5), (7.5, 124.5), (0, 116.5), afronding=2.8).difference(glim.buffer(0.05))
    gezicht = unary_union([lach_ogen(25, 137), neus, lach_mond(cy=113.5, steel=4.5), snor_stipjes()])

    return Tekening(
        silhouet=unary_union([omtrek, poten]),
        lagen=[
            relief(ORANJE, omtrek),
            vlak(WIT, unary_union([masker, binnen])),
            vlak(ZWART, punt),
            detail(ZWART, gezicht),
            detail(WIT, glim),
            poot(ORANJE, poten),
            op_poot(ZWART, teen_lijnen(PAW_X)),
        ])


DIER = Dier("vos", 1, {"nl": "vos", "en": "fox"}, bord=ORANJE, kop=ORANJE, decor=(GROEN, "blad"),
            tekening=tekening)
