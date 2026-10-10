"""Draak: Mistletoe Green kop en bord, Sunflower Yellow stekels, witte snuit en hoorntjes."""
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geom import arc_points, ellipse, smooth, stroke, sym
from params import GEEL, GROEN, WIT, ZWART

from .kit import Dier, Tekening, detail, lach_ogen, op_poot, poot, poot_ovaal, relief, vlak, vol

POOT_X = 38.0


def _stekels():
    """Brede, afgeronde stekels langs de bovenkant van de kop."""
    out = []
    for a in (90, 62, 118, 36, 144):
        t = np.radians(a)
        cx, cy = 54 * 1.0 * np.cos(t), 138 + 44 * np.sin(t)
        nx, ny = np.cos(t), np.sin(t)
        h = 15 if a == 90 else 13 if a in (62, 118) else 11
        b = 10
        tip = (cx + nx * h, cy + ny * h)
        l = (cx - ny * b - nx * 4, cy + nx * b - ny * 4)
        r = (cx + ny * b - nx * 4, cy - nx * b - ny * 4)
        out.append(smooth(Polygon([l, tip, r]), open_=3.2))
    return unary_union(out)


def tekening():
    kop = smooth(unary_union([ellipse(0, 138, 54, 42), ellipse(0, 113, 40, 22)]), close=10)
    stekels = _stekels().difference(kop.buffer(-6))
    vorm = smooth(unary_union([kop, stekels]), close=3)

    snuit = ellipse(0, 112, 33, 17)
    neusgaten = sym(ellipse(-9.5, 115, 2.8, 2.2))
    mond = stroke(arc_points(0, 109, 9, 205, 335), 1.8)
    ogen = lach_ogen(23, 141)
    schubjes = unary_union([ellipse(0, 170, 3.2, 3.2), ellipse(-8, 165, 2.4, 2.4), ellipse(8, 165, 2.4, 2.4)])
    wangen = sym(ellipse(-38, 126, 4.5, 3.2))

    poten = poot_ovaal(POOT_X, rx=15.5, ry=10.5)
    # witte klauwtjes onderaan op de poot
    klauw = lambda x: Polygon([(x - 2.6, 95.5), (x + 2.6, 95.5), (x, 91.0)])  # noqa: E731
    klauwen = sym(unary_union([smooth(klauw(-POOT_X + d), open_=0.6) for d in (-6, 0, 6)]))
    klauwen = klauwen.intersection(poten.buffer(-0.6))
    return Tekening(
        silhouet=unary_union([vorm, poten]),
        lagen=[
            relief(GROEN, vorm),
            vol(GEEL, stekels.difference(kop)),
            vlak(WIT, snuit),
            vlak(GEEL, unary_union([schubjes, wangen])),
            detail(ZWART, unary_union([ogen, neusgaten, mond])),
            poot(GROEN, poten),
            op_poot(WIT, klauwen),
        ])


DIER = Dier("draak", 4, {"nl": "draak", "en": "dragon"}, bord=GROEN, kop=GROEN, decor=(GEEL, "blad"),
            tekening=tekening)
