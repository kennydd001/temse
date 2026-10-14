"""De vier onderdelen van H-02 Moodkin, in de stand zoals ze samen op tafel staan (z = 0 is de onderkant van
de schil, het gezicht kijkt naar -y). printstand() legt elk onderdeel zoals het op het bed moet.

Een pompoen met ribben: de straal zonder ribben is r_buiten(z), de ribben zijn ±RIB daarbovenop, met een rib
recht vooraan (het gezicht zit op een bol stuk, zoals bij een echte pompoen). Het deksel heeft dezelfde ribben;
één klik is 90°, twee ribben, dus de ribben van deksel en schil liggen in elke stand op elkaar.
"""
import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold

import _pad  # noqa: F401
import gezichten as G
import params as P
from gereedschap.geom import uittrekken, union, vlak_yz

SEG = 192               # stappen rond: 24 per rib


def r_buiten(z):
    """Straal van de schil zonder ribben: een buik die onderaan naar binnen loopt, daarboven recht."""
    if z >= P.Z_BOL:
        return P.R_MAX
    return P.R_MAX - (P.R_MAX - P.R_BODEM) * (1 - z / P.Z_BOL) ** 1.5


def r_binnen(z):
    """Straal van de holte: overal minstens WAND in een ribdal, en nooit wijder dan R_BINNEN_SCHIL (de trommel
    moet er recht in zakken, met SPELING rondom)."""
    return min(P.R_BINNEN_SCHIL, r_buiten(z) * (1 - P.RIB) - P.WAND)


def _draai(profiel):
    """(r, z)-profiel rond de z-as."""
    return Manifold.revolve(CrossSection([profiel]), SEG)


def _ribben(m):
    """Straal ×(1 + RIB·cos 8θ): ribben, met een top recht vooraan (θ = -90°)."""
    def f(v):
        v = np.array(v, dtype=float)
        th = np.arctan2(v[:, 1], v[:, 0])
        s = 1 + P.RIB * np.cos(P.N_RIBBEN * th)
        v[:, 0] *= s
        v[:, 1] *= s
        return v
    return m.warp_batch(f)


def prisma_y(g, y0, y1):
    """2D-vorm (x, z) uittrekken langs y, van y0 tot y1 (y0 < y1)."""
    m = uittrekken(g, y1 - y0).translate([0, 0, -y1])
    return m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))


def _naar_band(g):
    """Gezichtstekening (u, v) naar (x, z) op de band."""
    from shapely import affinity
    return affinity.translate(g, 0, P.Z_GEZICHT[0])


# --- schil -----------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def schil():
    zs = np.linspace(0, P.Z_BOL, 69)
    buiten = _ribben(_draai([(0, 0)] + [(r_buiten(z), z) for z in zs] + [(P.R_MAX, P.H_SCHIL), (0, P.H_SCHIL)]))
    zs = np.concatenate([np.arange(P.VLOER, P.H_SCHIL, 0.25), [P.H_SCHIL + 1]])
    holte = _draai([(0, P.VLOER)] + [(r_binnen(z), z) for z in zs] + [(0, P.H_SCHIL + 1)])
    kassen = prisma_y(_naar_band(G.kassen()), -P.R_MAX * 1.2, 0)
    ring = (Manifold.cylinder(P.LED_RING_H, P.LED_RING_D / 2 + P.LED_RING_W, circular_segments=96)
            - Manifold.cylinder(P.LED_RING_H + 1, P.LED_RING_D / 2, circular_segments=96).translate([0, 0, -0.5]))
    nok = vlak_yz(_driehoek(P.NOK_B, P.NOK_H, P.H_SCHIL - 0.01), P.NOK_R - 1.0, P.NOK_R + 1.0)
    nokjes = [nok.rotate([0, 0, h]) for h in P.NOK_HOEKEN]
    return union([buiten - holte - kassen, ring.translate([0, 0, P.VLOER - 0.01])] + nokjes)


def _driehoek(b, h, z0):
    """Driehoek (in het vlak van één coördinaat en z) met basis b op z0 en de top h hoger: nokje en gleuf."""
    from shapely.geometry import Polygon
    return Polygon([(-b / 2, z0), (b / 2, z0), (0, z0 + h)])


# --- trommel ---------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def gaten_trommel():
    """De vier gezichten als prisma's: gezicht k staat op -90° + 90°·k (k = 0 vooraan)."""
    delen = []
    for k, naam in enumerate(P.GEZICHTEN):
        g = prisma_y(_naar_band(G.gezicht(naam)), -P.R_TROMMEL - 5, -(P.R_TROMMEL - P.TROMMEL_WAND) * 0.5)
        delen.append(g.rotate([0, 0, 90.0 * k]))
    return union(delen)


@lru_cache(maxsize=None)
def trommel():
    r_in = P.R_TROMMEL - P.TROMMEL_WAND
    z0, z1 = P.Z_TROMMEL, P.H_SCHIL
    buis = (Manifold.cylinder(z1 - z0, P.R_TROMMEL, circular_segments=SEG)
            - Manifold.cylinder(z1 - z0 + 2, r_in, circular_segments=SEG).translate([0, 0, -1])).translate([0, 0, z0])
    buis = buis - gaten_trommel()
    # richel binnen bovenaan, met daaronder een schuine kant van hoogstens 40° (vanaf verticaal)
    r_r = r_in - P.RICHEL_B
    schuin = (P.RICHEL_B + 0.2) / math.tan(math.radians(P.OVERHANG))
    zr = z1 - P.RICHEL_DIK
    richel = _draai([(r_r, zr), (r_in + 0.2, zr - schuin), (r_in + 0.2, z1), (r_r, z1)])
    pen = (Manifold.cylinder(P.PEN_H - 0.5, P.PEN_D / 2, circular_segments=32)
           + Manifold.cylinder(0.5, P.PEN_D / 2, P.PEN_D / 2 - 0.4, circular_segments=32).translate([0, 0, P.PEN_H - 0.5]))
    pennen = [pen.translate([P.PEN_R, 0, z1 - 0.01]).rotate([0, 0, h]) for h in P.PEN_HOEKEN]
    return union([buis, richel] + pennen)


# --- deksel en steeltje ----------------------------------------------------------------------------

DEKSEL_TOP = P.DEKSEL_RAND + P.DEKSEL_H


@lru_cache(maxsize=None)
def deksel():
    """Deksel in zijn eigen stand: onderkant op z = 0 (zo ligt het ook op het bed)."""
    hd = P.DEKSEL_H * 1.08                       # de koepel wordt bovenaan afgesneden: een vlak voor het steeltje
    prof = [(0, 0), (P.R_DEKSEL, 0), (P.R_DEKSEL, P.DEKSEL_RAND)]
    prof += [(P.R_DEKSEL * math.sqrt(1 - (h / hd) ** 2), P.DEKSEL_RAND + h) for h in np.linspace(0, P.DEKSEL_H, 31)[1:]]
    prof += [(0, DEKSEL_TOP)]
    koepel = _ribben(_draai(prof))
    gat = Manifold.cylinder(P.PEN_H + 0.6, P.PEN_GAT / 2, circular_segments=32).translate([P.PEN_R, 0, -0.5])
    gaten = [gat.rotate([0, 0, h]) for h in P.PEN_HOEKEN]
    gleuf = vlak_yz(_driehoek(P.GLEUF_B, P.GLEUF_DIEP, -0.01), P.NOK_R - 1.5, P.NOK_R + 1.5)
    gleuven = [gleuf.rotate([0, 0, h]) for h in P.NOK_HOEKEN]
    s = P.STEEL_GAT
    steelgat = Manifold.cube([s, s, P.STEEL_PEN_H + 1.5]).translate([-s / 2, -s / 2, DEKSEL_TOP - P.STEEL_PEN_H - 0.5])
    return koepel - union(gaten + gleuven + [steelgat])


@lru_cache(maxsize=None)
def steel():
    """Steeltje in zijn eigen stand: onderkant op z = 0 (op het deksel), vierkant pennetje eronder."""
    n = 96
    phi = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r = P.STEEL_R[0] * (1 + 0.07 * np.cos(P.STEEL_GROEVEN * phi))
    doorsnede = CrossSection([list(zip(r * np.cos(phi), r * np.sin(phi)))])
    m = Manifold.extrude(doorsnede, P.STEEL_H, n_divisions=40, twist_degrees=P.STEEL_DRAAI,
                         scale_top=(P.STEEL_R[1] / P.STEEL_R[0],) * 2)

    def buig(v):
        v = np.array(v, dtype=float)
        v[:, 0] += P.STEEL_BOCHT * (np.clip(v[:, 2], 0, None) / P.STEEL_H) ** 2
        return v
    m = m.warp_batch(buig)
    s = P.STEEL_PEN
    pen = Manifold.cube([s, s, P.STEEL_PEN_H + 0.01]).translate([-s / 2, -s / 2, -P.STEEL_PEN_H])
    return m + pen


# --- samen en printstand ---------------------------------------------------------------------------

def samen(stand=0, open_=0.0):
    """Alle onderdelen op hun plek. stand k: het deksel k klikken verder (gezicht k vooraan). open_: deksel met
    trommel zoveel mm opgetild."""
    draai = -90.0 * stand
    return {
        "schil": schil(),
        "trommel": trommel().rotate([0, 0, draai]).translate([0, 0, open_]),
        "deksel": deksel().rotate([0, 0, draai]).translate([0, 0, P.H_SCHIL + open_]),
        "steel": steel().rotate([0, 0, draai]).translate([0, 0, P.H_SCHIL + DEKSEL_TOP + open_]),
    }


def printstand():
    """Elk onderdeel zoals het op het bed ligt (onderkant op z = 0, midden op x = y = 0)."""
    st = steel().rotate([180, 0, 0])
    return {
        "schil": schil(),
        "trommel": trommel().translate([0, 0, -P.Z_TROMMEL]),
        "deksel": deksel(),
        "steel": st.translate([0, 0, -st.bounding_box()[2]]),   # op zijn kop: het vlakke topje op het bed
    }
