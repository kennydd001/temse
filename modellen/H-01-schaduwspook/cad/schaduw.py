"""Schaduwkunst: één object, twee totaal verschillende schaduwen.

Silhouet A ligt in het x-z-vlak (vooraanzicht), silhouet B in het y-z-vlak (zijaanzicht). Het object is de
doorsnede van A, uitgetrokken langs y, en B, uitgetrokken langs x. Op hoogte z is de doorsnede dus
A_z × B_z (de stukken van A op die hoogte maal de stukken van B). Van voren gezien is dat precies A_z, van
opzij precies B_z, zolang beide op elke hoogte iets hebben. Daarom staan beide silhouetten op dezelfde
voetbalk en reiken ze even hoog.

Printen zonder support: elk silhouet wordt eerst 45°-zelfdragend gemaakt (zelfdragend()). Dan steekt elke
laag van A en van B hooguit één laagdikte uit over de laag eronder, en de doorsnede A_z × B_z dus ook (op
een buitenhoek hooguit √2 laagdikte, ~55°). Ook hangt dan alles via een dalende weg aan de voetbalk: één
stuk, geen losse eilandjes.
"""
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

import _pad  # noqa: F401
from gereedschap.geom import cs_to_shapely, polys, uittrekken
from gereedschap.zelfdragend import lengte as _lengte, rij as _rij, samen as _samen, zonder as _min


# --- het object ---------------------------------------------------------------------------------

def prisma(g, hoek, lengte):
    """Silhouet g (x = breedte, y = hoogte) rechtop in het x-z-vlak, uitgetrokken langs y over `lengte` rond 0,
    en dan `hoek` graden gedraaid rond de verticale as. Van voren gezien na een draai over -hoek: g."""
    m = uittrekken(g, lengte).transform(np.array([[1, 0, 0, 0], [0, 0, -1, lengte / 2], [0, 1, 0, 0]], dtype=float))
    return m.rotate([0, 0, hoek]) if hoek else m


def object_(silhouetten, hoeken):
    """De doorsnede van alle silhouetten, elk uitgetrokken in zijn eigen kijkrichting -> Manifold."""
    lengte = 2 * max(g.bounds[2] - g.bounds[0] for g in silhouetten) + 20
    m = None
    for g, h in zip(silhouetten, hoeken):
        p = prisma(g, h, lengte)
        m = p if m is None else m ^ p
    return m


def schaduw(m, as_, zs):
    """Projectie van m langs as_ ('y' geeft x-z, 'x' geeft y-z) als stukken per hoogte z.

    De doorsnede op hoogte z bestaat uit rechthoekige blokjes; elk los stuk projecteert op één interval.
    """
    uit = []
    for z in zs:
        s = cs_to_shapely(m.slice(float(z)))
        k = 0 if as_ == "y" else 1
        uit.append(_samen([(p.bounds[k], p.bounds[k + 2]) for p in polys(s)]))
    return uit


def verschil(m, silhouetten, hoeken, res=0.2):
    """Per kijkrichting het oppervlak (mm²) dat in de schaduw van m ontbreekt of te veel is."""
    out = []
    for g, h in zip(silhouetten, hoeken):
        mr = m.rotate([0, 0, -h]) if h else m
        _, z0, _, z1 = g.bounds
        zs = np.arange(z0 + res / 2, z1, res)
        x0, _, x1, _ = g.bounds
        mis = teveel = 0.0
        for z, echt in zip(zs, schaduw(mr, "y", zs)):
            doel = _rij(g, z, x0 - 1, x1 + 1)
            mis += _lengte(_min(doel, echt)) * res
            teveel += _lengte(_min(echt, doel)) * res
        out.append({"hoek": h, "doel_mm2": round(g.area, 1), "ontbreekt_mm2": round(mis, 1),
                    "te_veel_mm2": round(teveel, 1)})
    return out


def als_polygonen(g):
    """Altijd een MultiPolygon (voor tekenen en tellen)."""
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    return g
