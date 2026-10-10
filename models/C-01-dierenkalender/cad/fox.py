"""Eigen vossenkop voor het dierenfront, getekend uit eenvoudige vormen.

Alle maten in mm in frontcoördinaten: x = breedte (0 = midden), y = omhoog,
bovenrand bord op y = 100. Elke functie geeft shapely-geometrie terug.
"""
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geom import arc_points, ellipse, smooth, stroke, sym

EYE_Y = 137.0


def head():
    dome = ellipse(0, 141, 55, 43)
    cheek = Polygon([(-24, 146), (-76, 117), (-70, 111), (-48, 103), (-20, 99)])
    chin = ellipse(0, 108, 30, 13)
    h = unary_union([dome, sym(cheek), chin])
    return smooth(h, close=8, open_=3)


def ears():
    ear = Polygon([(-53, 156), (-13, 177), (-66, 206)])
    return sym(smooth(ear, open_=5))


def inner_ears():
    inner = Polygon([(-49, 161), (-22, 175), (-61, 197)])
    inner = smooth(inner, open_=3.5)
    return sym(inner).intersection(ears().buffer(-3.2))


def ear_tips():
    """Donkere oorpuntjes, met een gebogen rand (cirkel rond de oorspronkelijke scherpe punt)."""
    tip = ellipse(-66, 206, 19.0, 19.0)
    return sym(tip).intersection(ears())


def outline():
    return smooth(unary_union([head(), ears()]), close=6)


def mask():
    """Witte wangen en snuit; het oranje voorhoofd loopt in een V naar de neus."""
    cheek = ellipse(-33, 118, 36, 19, rot=-18)
    lower = Polygon([(-90, 121), (-60, 119), (0, 104), (60, 119), (90, 121), (90, 80), (-90, 80)])
    muzzle = ellipse(0, 114, 21, 14)
    white = unary_union([sym(cheek), lower, muzzle])
    v = Polygon([(-15, 160), (15, 160), (4.5, 124), (-4.5, 124)])
    white = white.difference(smooth(v, open_=2.5))
    return smooth(white.intersection(head().buffer(-0.01)), open_=1.5)


def eyes():
    """Gesloten, lachende oogjes (∩) met een klein wimpertje."""
    pts = arc_points(-25, EYE_Y - 4, 7.5, 160, 20)
    lash = stroke([(-31.5, EYE_Y - 1.0), (-35.5, EYE_Y + 1.5)], 2.2)
    return sym(unary_union([stroke(pts, 3.2), lash]))


def nose():
    tri = Polygon([(-7.5, 124.5), (7.5, 124.5), (0, 116.5)])
    return smooth(tri, open_=2.8).difference(highlight().buffer(0.05))


def highlight():
    """Wit glimlichtje op de neus (Ø 2,6 mm)."""
    return ellipse(-2.6, 122.2, 1.4, 1.2)


def whisker_dots():
    """Drie snorharenstipjes per kant (Ø 1,8 mm)."""
    return sym(unary_union([ellipse(x, y, 0.9, 0.9) for x, y in ((-11.5, 115.5), (-14.5, 112.0), (-11.0, 109.0))]))


def mouth():
    stem = stroke([(0, 118), (0, 113.5)], 1.8)
    smile = stroke(arc_points(-4.2, 113.5, 4.2, 180, 360), 1.8)
    return unary_union([stem, sym(smile)])


def details():
    return unary_union([eyes(), nose(), mouth(), whisker_dots()])


PAW_C = 37.0
PAW_Y = 100.0


def paws():
    p = ellipse(-PAW_C, PAW_Y, 15.5, 10.5)
    return sym(p)


def toes():
    lines = [stroke([(-PAW_C + dx, PAW_Y - 8.5), (-PAW_C + dx, PAW_Y - 3.5)], 1.6) for dx in (-4.5, 4.5)]
    return sym(unary_union(lines))


def fox_parts(top=194.0):
    """Alle vormen, geschaald rond het midden van de bovenrand zodat de oortoppen op `top` eindigen."""
    parts = {
        "outline": outline(),
        "inner_ears": inner_ears(),
        "mask": mask(),
        "details": details(),
        "highlight": highlight(),
        "ear_tips": ear_tips(),
        "paws": paws(),
        "toes": toes(),
    }
    k = (top - PAW_Y) / (parts["outline"].bounds[3] - PAW_Y)
    return {n: affinity.scale(g, k, k, origin=(0, PAW_Y)) for n, g in parts.items()}
