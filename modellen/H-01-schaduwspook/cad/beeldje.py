"""Een variant van H-01 bouwen: silhouetten op hoogte, op de voetbalk, zelfdragend, doorsneden."""
from functools import lru_cache

from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
import schaduw as S
import silhouetten
from gereedschap.zelfdragend import zelfdragend


def op_voet(g, hoogte, voet=P.VOET_H):
    """Silhouet g schalen tot `hoogte` (voetbalk inbegrepen), gecentreerd, op een voetbalk van `voet` mm."""
    x0, y0, x1, y1 = g.bounds
    h = hoogte - P.VOET_H + P.OVERLAP
    k = h / (y1 - y0)
    g = affinity.scale(g, k, k, origin=(x0, y0))
    x0, y0, x1, y1 = g.bounds
    g = affinity.translate(g, -(x0 + x1) / 2, P.VOET_H - P.OVERLAP - y0)
    x0, _, x1, _ = g.bounds
    return unary_union([g, box(x0 - P.VOET_MARGE, 0, x1 + P.VOET_MARGE, voet)])


def printklaar(vormen, hoogte):
    """Elk silhouet op hoogte, op zijn balk (elk een laag hoger) en zelfdragend (elk een eigen rijenraster)."""
    n = len(vormen)
    return tuple(zelfdragend(op_voet(silhouetten.vorm(v), hoogte, P.VOET_H + k * P.VOET_STAP),
                               P.RES, P.OVERHANG, fase=k / n) for k, v in enumerate(vormen))


@lru_cache(maxsize=None)
def silhouetten_van(sleutel):
    """De silhouetten zoals ze geprint worden."""
    _, _, vormen, _, hoogte, _ = P.VARIANTEN[sleutel]
    return printklaar(vormen, hoogte)


def ontwerp(sleutel):
    """De silhouetten zoals getekend, op hoogte en op de balk: de schaduw die je wilt zien."""
    _, _, vormen, _, hoogte, _ = P.VARIANTEN[sleutel]
    return tuple(op_voet(silhouetten.vorm(v), hoogte, P.VOET_H + k * P.VOET_STAP) for k, v in enumerate(vormen))


@lru_cache(maxsize=None)
def beeldje(sleutel):
    return S.object_(silhouetten_van(sleutel), P.VARIANTEN[sleutel][3])
