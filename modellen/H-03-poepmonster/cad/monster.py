"""H-03 Poepmonster in 3D: één stuk, zo op het bed (z = 0 is de bodem, het gezicht kijkt naar -y).

De romp is een pompoen met ribben en een wand van overal WAND dik (de ribben lopen binnen mee). De bovenrand
is een schuin vlak (laag achteraan, hoog vooraan) met daarbovenop een zaagtand: TANDEN tanden rondom, van
dal tot punt TAND_H. Vooraan gaan de gaten van het gezicht (gezicht.py) recht naar achteren door de wand.
"""
import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold, Mesh

import _pad  # noqa: F401
import gezicht as G
import params as P
from gereedschap.geom import uittrekken

SEG = 192


def r_romp(z):
    """Straal zonder ribben: buik onderaan (zoals H-02), daarboven langzaam weer naar binnen."""
    if z <= P.Z_BOL:
        return P.R_MAX - (P.R_MAX - P.R_BODEM) * (1 - z / P.Z_BOL) ** 1.5
    t = min(1.0, (z - P.Z_BOL) / (P.Z_TOP - P.Z_BOL))
    return P.R_MAX - (P.R_MAX - P.R_TOP) * t * t


def _draai(profiel):
    """Rond de z-as, een half segment verdraaid: de meridianen vallen dan niet samen met de punten en dalen van de
    tanden (die liggen om de 7,5°, een veelvoud van 360/SEG)."""
    return Manifold.revolve(CrossSection([profiel]), SEG).rotate([0, 0, 180.0 / SEG])


def _ribben(m):
    def f(v):
        v = np.array(v, dtype=float)
        s = 1 + P.RIB * np.cos(P.N_RIBBEN * np.arctan2(v[:, 1], v[:, 0]))
        v[:, 0] *= s
        v[:, 1] *= s
        return v
    return m.warp_batch(f)


def rand_hoogte(sleutel, x, y):
    """Hoogte van de bovenrand boven (x, y): schuin vlak door de tanddalen plus de zaagtand."""
    _, _, achter, voor, _ = P.VARIANTEN[sleutel]
    vlak = achter + (voor - achter) * (P.R_MAX - y) / (2 * P.R_MAX)
    fase = ((math.atan2(y, x) + math.pi / 2) * P.TANDEN / (2 * math.pi)) % 1.0     # 0 = punt recht vooraan
    return vlak + P.TAND_H * abs(1 - 2 * fase)


def _boven_rand(sleutel, ringen=(20.0, 46.0, 63.0, 90.0), top=400.0):
    """Alles boven de getande rand, als één veelvlak: ringen rond het midden, twee punten per tand (punt en dal),
    zodat de zaagtand op elke ring precies klopt. Geen ring in de wand zelf (straal ~48-58 rond de rand): een
    rand van het veelvlak die op de wand ligt, geeft splinterdriehoekjes die in een STL (float32) samenvallen."""
    n = 2 * P.TANDEN
    th = -math.pi / 2 + np.arange(n) * 2 * math.pi / n
    v = [(0.0, 0.0, rand_hoogte(sleutel, 0.0, 0.0))]
    for r in ringen:
        v += [(r * math.cos(t), r * math.sin(t), rand_hoogte(sleutel, r * math.cos(t), r * math.sin(t))) for t in th]
    R = ringen[-1]
    v += [(R * math.cos(t), R * math.sin(t), top) for t in th] + [(0.0, 0.0, top)]
    ring = lambda i, k: 1 + i * n + k % n  # noqa: E731
    boven = lambda k: 1 + len(ringen) * n + k % n  # noqa: E731
    tc = len(v) - 1
    f = []
    for k in range(n):                                       # onderkant: normaal naar beneden
        f.append((0, ring(0, k + 1), ring(0, k)))
        for i in range(len(ringen) - 1):
            a, b, c, d = ring(i, k), ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k)
            f += [(a, b, c), (a, c, d)]
        o, o1 = ring(len(ringen) - 1, k), ring(len(ringen) - 1, k + 1)
        f += [(o, o1, boven(k + 1)), (o, boven(k + 1), boven(k))]   # zijkant
        f.append((tc, boven(k), boven(k + 1)))                    # bovenkant
    return Manifold(Mesh(vert_properties=np.asarray(v, dtype=np.float32), tri_verts=np.asarray(f, dtype=np.uint32)))


def prisma_y(g, y0, y1):
    """2D-vorm (x, z) uittrekken langs y, van y0 tot y1 (y0 < y1)."""
    m = uittrekken(g, y1 - y0).translate([0, 0, -y1])
    return m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))


@lru_cache(maxsize=None)
def monster(sleutel):
    zs = np.linspace(0, P.Z_TOP, 121)
    romp = _ribben(_draai([(0, 0)] + [(r_romp(z), z) for z in zs] + [(0, P.Z_TOP)]))
    zs = np.linspace(P.VLOER, P.Z_TOP + 1, 120)
    holte = _ribben(_draai([(0, P.VLOER)] + [(r_romp(min(z, P.Z_TOP)) - P.WAND, z) for z in zs] + [(0, P.Z_TOP + 1)]))
    gaten = prisma_y(G.gaten(sleutel), -P.R_MAX * 1.3, -P.R_MAX * 0.3)
    return romp - holte - _boven_rand(sleutel) - gaten


def inhoud_cm3(sleutel):
    """Wat erin past tot het laagste tanddal (achteraan)."""
    holte = _ribben(_draai([(0, P.VLOER)] + [(r_romp(z) - P.WAND, z)
                                             for z in np.linspace(P.VLOER, P.VARIANTEN[sleutel][2], 60)]
                           + [(0, P.VARIANTEN[sleutel][2])]))
    return round(holte.volume() / 1000, 0)
