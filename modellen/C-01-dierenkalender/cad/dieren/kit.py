"""Gedeelde bouwstenen voor de dierenfronten.

Een dier is een `Dier` met een `tekening()` die een `Tekening` teruggeeft:
  - silhouet: wat boven/over het bord uitsteekt (kop, oren, pootjes); hoort bij de 3 mm plaat
  - lagen:    gekleurde reliëflagen in schildervolgorde (latere lagen winnen waar ze overlappen)

Assen: x = breedte (0 = midden), y = omhoog, bovenrand bord op y = 100, hoogste punt <= 194.
z is vanaf de achterkant van de plaat: plaat 0-3, kopreliëf tot 4, kleurvlakken tot 4,6,
details (ogen, neus) tot 5, pootjes tot 5, teenlijntjes tot 5,2.
"""
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from geom import arc_points, ellipse, rounded_rect, smooth, stroke, sym

E = 0.01
Z_PLAAT = 3.0
Z_KOP = 4.0
Z_KLEUR = 4.6
Z_DETAIL = 5.0
Z_POOT = 5.0
Z_TEEN = 5.2


@dataclass
class Laag:
    filament: str
    vorm: object
    z0: float
    z1: float


@dataclass
class Tekening:
    silhouet: object
    lagen: list


@dataclass
class Dier:
    sleutel: str                 # mapnaam, b.v. "vos"
    nr: int                      # volgorde in de set
    naam: dict                   # {"nl": "vos", "en": "fox"}
    bord: str                    # filament van bord en voet
    kop: str                     # filament van de kop (volle plaatdikte buiten het bord)
    decor: tuple                 # (filament, "blad" | "ster" | None)
    tekening: object             # functie -> Tekening
    label: str = "11101"         # kleur van Dag/Datum/Maand (standaard Charcoal)
    fanart: bool = False         # bestaand personage van een ander: alleen voor eigen gebruik, niet publiceren


# --- vaste lagen ----------------------------------------------------------------

def relief(filament, vorm, top=Z_KOP):
    """Kopreliëf: ligt op de plaat."""
    return Laag(filament, vorm, Z_PLAAT - E, top)


def vol(filament, vorm, top=Z_KOP):
    """Door de hele plaatdikte gekleurd (manen, stekels buiten de kop), zodat ook de zijkant klopt."""
    return Laag(filament, vorm, 0.0, top)


def vlak(filament, vorm):
    """Kleurvlak óp het kopreliëf (wangen, binnenoren, vlekken). Een andere kleur hoeft niet te overlappen."""
    return Laag(filament, vorm, Z_KOP, Z_KLEUR)


def detail(filament, vorm):
    """Ogen, neus, mond: iets hoger dan de kleurvlakken."""
    return Laag(filament, vorm, Z_KOP, Z_DETAIL)


def poot(filament, vorm, z0=Z_PLAAT - E):
    """Pootjes/hoefjes over de bovenrand van het bord, hoogste reliëf.

    Zit de poot al in het kopreliëf (zelfde kleur), begin dan op z0 = Z_KOP - E: dan vallen de
    zijwanden van poot en reliëf niet net naast elkaar.
    """
    return Laag(filament, vorm, z0, Z_POOT)


def op_poot(filament, vorm, top=Z_TEEN):
    """Teenlijntjes en hoefranden, 0,4 mm in de poot en 0,2 mm erboven."""
    return Laag(filament, vorm, Z_POOT - 0.4, top)


# --- vormen -----------------------------------------------------------------------

def lach_ogen(cx, cy, r=7.5, w=3.2, wimper=True, spreiding=1.0):
    """Gesloten, lachende oogjes (∩), gespiegeld rond x = 0. cx > 0 is het rechteroog."""
    pts = arc_points(-cx, cy - 4, r, 160, 20)
    oog = stroke(pts, w)
    if wimper:
        x0 = -cx - r * 0.87 * spreiding
        oog = unary_union([oog, stroke([(x0, cy - 1.0), (x0 - 4.0, cy + 1.5)], w * 0.68)])
    return sym(oog)


def snor_stipjes(cx=11.5, cy=115.5, r=0.9):
    pts = [(-cx, cy), (-cx - 3.0, cy - 3.5), (-cx + 0.5, cy - 6.5)]
    return sym(unary_union([ellipse(x, y, r, r) for x, y in pts]))


def poot_ovaal(cx, cy=100.0, rx=15.5, ry=10.5):
    return sym(ellipse(-cx, cy, rx, ry))


def teen_lijnen(cx, cy=100.0, dx=4.5, y0=-8.5, y1=-3.5, w=1.6):
    lines = [stroke([(-cx + d, cy + y0), (-cx + d, cy + y1)], w) for d in (-dx, dx)]
    return sym(unary_union(lines))


def hoefje(cx, cy=100.0, w=24.0, h=21.0, r=7.0):
    """Afgerond hoefje over de bordrand; geeft (hoef, onderband, spleet)."""
    hoef = rounded_rect(-cx - w / 2, cy - h / 2, -cx + w / 2, cy + h / 2, r)
    band = hoef.intersection(rounded_rect(-cx - w / 2 - 1, cy - h / 2 - 1, -cx + w / 2 + 1, cy - h / 2 + 7, 0.1))
    spleet = stroke([(-cx, cy - h / 2 + 0.5), (-cx, cy - h / 2 + 6.0)], 1.6)
    return sym(hoef), sym(band), sym(spleet)


def blad(cx, cy, lengte=24.0, breedte=11.0, rot=0.0):
    """Blaadje (lens) met nerf; geeft (blad, nerf)."""
    r = (lengte ** 2 / 4 + breedte ** 2 / 4) / breedte
    d = r - breedte / 2
    a = affinity.rotate(ellipse(0, d, r, r).intersection(ellipse(0, -d, r, r)), rot, origin=(0, 0))
    nerf = affinity.rotate(stroke([(-lengte * 0.36, 0), (lengte * 0.36, 0)], 1.2), rot, origin=(0, 0))
    return affinity.translate(a, cx, cy), affinity.translate(nerf, cx, cy)


def ster(cx, cy, r=11.0, binnen=0.48, rot=0.0, afronding=1.4):
    pts = []
    for i in range(10):
        a = np.radians(90 + rot + i * 36)
        rr = r if i % 2 == 0 else r * binnen
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    return smooth(Polygon(pts), open_=afronding)


def blos(cx, cy, rx=5.5, ry=3.5):
    return sym(ellipse(-cx, cy, rx, ry))


def driehoek(p1, p2, p3, afronding=0.0):
    t = Polygon([p1, p2, p3])
    return smooth(t, open_=afronding) if afronding else t


def omlijning(vorm, w=1.5):
    """Cartoonrand: een band van w mm langs de binnenkant van de vorm."""
    return vorm.difference(vorm.buffer(-w))


def lijn(punten, w=1.5):
    return stroke(punten, w)


def handschoenen(xs, kleur, y=100.5, rx=13.5, ry=10.0, rand_kleur="11101"):
    """Gehandschoende handjes over de bordrand met cartoonrand en vingerlijntjes. Geeft (vorm, lagen)."""
    handen = unary_union([ellipse(x, y, rx, ry) for x in xs])
    vingers = unary_union([lijn([(x + d, y - ry + 2.0), (x + d, y - ry + 7.0)], 1.4) for x in xs for d in (-4.5, 0.0, 4.5)])
    return handen, [poot(kleur, handen), op_poot(rand_kleur, unary_union([omlijning(handen, 1.3), vingers]))]


def pixels(raster, x0, y_top, maat, overlap=0.0):
    """Pixelkunst: raster is een lijst strings (bovenste rij eerst). Geeft {teken: vorm}."""
    out = {}
    for r, rij in enumerate(raster):
        for c, t in enumerate(rij):
            if t in " .":
                continue
            x, y = x0 + c * maat, y_top - (r + 1) * maat
            out.setdefault(t, []).append(rounded_rect(x - overlap, y - overlap, x + maat + overlap, y + maat + overlap, 0.01))
    return {t: unary_union(v) for t, v in out.items()}


def lach_mond(cx=0.0, cy=113.5, r=4.2, w=1.8, steel=4.5):
    """Mondje als 'ω' met een steeltje naar de neus."""
    st = stroke([(cx, cy + steel), (cx, cy)], w)
    boog = stroke(arc_points(cx - r, cy, r, 180, 360), w)
    return unary_union([st, boog, affinity.scale(boog, -1, 1, origin=(cx, cy))])
