"""Eenhoorn: Ivory White kop op een Hot Pink bord, roze manen, gele hoorn met witte streepjes, gele sterren."""
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geom import ellipse, smooth, stroke, sym
from params import GEEL, ROZE, WIT, ZWART

from .kit import Z_KLEUR, Dier, Tekening, blos, detail, hoefje, lach_ogen, op_poot, poot, relief, vlak, vol

HOEF_X = 34.0


def tekening():
    kop = smooth(unary_union([ellipse(0, 141, 45, 38), ellipse(0, 113, 33, 21)]), close=10)
    oor = smooth(Polygon([(-40, 160), (-20, 172), (-42, 189)]), open_=3.5)
    oren = sym(oor)
    binnen_oor = sym(smooth(Polygon([(-36, 164), (-24, 171), (-39, 183)]), open_=2.0))

    hoorn = smooth(Polygon([(-13, 160), (13, 160), (0, 199)]), open_=1.6)
    strepen = unary_union([stroke([(-13, y - 3.6), (13, y + 3.6)], 1.8) for y in (168, 176.5, 185)]).intersection(
        hoorn.buffer(-1.4))

    # manen: lokken langs de linkerkant en een kuifje tussen de oren
    lokken = unary_union([ellipse(-33, 173, 10.5, 9.5), ellipse(-45, 163, 10.5, 10), ellipse(-53, 150, 10.5, 10),
                          ellipse(-57, 136, 10, 10), ellipse(-56, 122, 9, 9)])
    # kuif: een lok die vanaf de manen schuin over het voorhoofd valt, links van de hoorn
    kuif = unary_union([ellipse(-24, 168, 9.5, 8), ellipse(-14, 166, 8.5, 6.8), ellipse(-6, 162.5, 6, 5)])
    manen = smooth(unary_union([lokken, kuif]), close=2)
    vorm = smooth(unary_union([kop, oren, hoorn, manen]), close=4)

    ogen = lach_ogen(20, 141)
    wangen = blos(31, 127, 6, 3.8)
    neusgaten = sym(ellipse(-9, 113, 2.8, 2.0, rot=15))
    mond = stroke([(-5, 105.5), (0, 104), (5, 105.5)], 1.7)

    hoef, band, _ = hoefje(HOEF_X, cy=99.5, w=23, h=20, r=7.0)
    return Tekening(
        silhouet=unary_union([vorm, hoef]),
        lagen=[
            relief(WIT, vorm),
            vol(ROZE, manen.buffer(0.4), top=Z_KLEUR),   # manen in één stuk, door de hele dikte, 0,4 over de kop
            vol(GEEL, hoorn.difference(kop)),
            vlak(ROZE, unary_union([binnen_oor, wangen])),
            vlak(WIT, strepen),
            detail(ZWART, unary_union([ogen, neusgaten, mond])),
            poot(WIT, hoef),
            op_poot(GEEL, band),
        ])


DIER = Dier("eenhoorn", 5, {"nl": "eenhoorn", "en": "unicorn"}, bord=ROZE, kop=WIT, decor=(GEEL, "ster"),
            tekening=tekening)
