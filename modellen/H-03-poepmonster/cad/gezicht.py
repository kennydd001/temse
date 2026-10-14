"""Het gezicht van H-03: de gaten vooraan, getekend zoals je ze van voren ziet (u = breedte, v = hoogte boven
het bed). Alles wordt recht naar achteren (langs y) door de voorwand gesneden.

- Twee ogen van verschillende grootte, met een spitse bovenkant. Onderin elk oog staat een pupil: een bolletje
  dat op de onderrand van het oog rust (dus zonder support print) en scheel naar de neus kijkt.
- De buik: een spleet in de vorm van een lens, dichtgenaaid met verticale hechtingen. De hechtingen houden de
  poep binnen en dragen de plafonds van de spleetjes ertussen (bruggen van 6,6 mm).
"""
from functools import lru_cache

from shapely.geometry import Point, box
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
from gereedschap.geom import druppel
from gereedschap.zelfdragend import zelfdragend


def oog(u, v, r, scheel):
    """Oog met een spitse bovenkant en een pupil die op de onderrand rust."""
    pupil = Point(u + scheel * 0.3 * r, v - 0.62 * r).buffer(0.5 * r, quad_segs=16)
    return druppel((u, v), r, omhoog=True).difference(pupil)


def lens(b, v0, v1):
    """Lens (twee cirkelbogen) van breedte b tussen hoogte v0 en v1, midden op u = 0."""
    h = (v1 - v0) / 2
    r = (b * b / 4 + h * h) / (2 * h)
    vm = (v0 + v1) / 2
    return Point(0, vm - (r - h)).buffer(r, quad_segs=64).intersection(Point(0, vm + (r - h)).buffer(r, quad_segs=64))


def buik(v0, v1):
    """Spleetjes tussen de hechtingen. Elk spleetje heeft een plat plafond (een korte brug van hechting tot
    hechting) op de laagste hoogte van de lens erboven: een schuin plafond zou overhangen."""
    rand = lens(P.BUIK_B, v0, v1)
    n = int(P.BUIK_B / 2 // P.HECHTING_STAP)
    spleten = []
    for k in range(-n - 1, n + 1):
        a = k * P.HECHTING_STAP + P.HECHTING_B / 2
        b = a + P.HECHTING_STAP - P.HECHTING_B
        strook = rand.intersection(box(a, v0 - 1, b, v1 + 1))
        if strook.is_empty:
            continue
        top = min(_boven(rand, u) for u in (a, b))
        spleten.append(strook.intersection(box(a, v0 - 1, b, top)))
    return unary_union([s for s in spleten if s.area > 4.0])


def _boven(g, u):
    """Bovenkant van g op breedte u (de onderkant als u buiten g valt)."""
    s = g.intersection(box(u - 1e-3, -1e3, u + 1e-3, 1e3))
    return s.bounds[3] if not s.is_empty else g.bounds[1]


def _wand(g):
    x0, y0, x1, y1 = g.bounds
    return box(x0 - 8, y0 - 6, x1 + 8, y1 + 8)


def printbaar(gaten):
    """Gaten zelfdragend maken: de wand eromheen zelfdragend, en wat daarvan overblijft is het gat."""
    wand = _wand(gaten)
    vol = zelfdragend(wand.difference(gaten), P.RES, P.OVERHANG, brug=P.BRUG)
    return wand.difference(vol)


def ontwerp(sleutel):
    _, _, achter, voor, (b0, b1) = P.VARIANTEN[sleutel]
    v_oog = voor - P.OOG_ONDER_RAND
    ogen = [oog(P.OOG_GROOT[0], v_oog, P.OOG_GROOT[1], +1), oog(P.OOG_KLEIN[0], v_oog + 1.5, P.OOG_KLEIN[1], -1)]
    return unary_union(ogen + [buik(b0, b1)])


@lru_cache(maxsize=None)
def gaten(sleutel):
    """Alle gaten vooraan, zelfdragend (u, v)."""
    return printbaar(ontwerp(sleutel))
