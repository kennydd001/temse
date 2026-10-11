"""Geometrie van het valbord van E-01 Splitsmijn, en de pasproef eromheen.

Onderdelen (alles in de assen van het valbord, zie params.py; de slinger in de ruststand):
  achterplaat   pinnen, zijranden met halve pinnen, bovenrand met inworpgleuf, nokjes voor de voorplaat
  voorplaat     tralies (kijkspleten), V-trechter bovenaan, gat voor de naaf, beugel met V-groef voor het mes
  plank         kantelplank in de spleet met naaf en zeskant
  slinger       arm met mesblok bovenaan en een massief gewicht onderaan; schuift op de zeskant
  voet          (pasproef) twee bakjes onder het bord, verdeler met vin, kokers voor de zijranden
  munt          diamantmuntje Ø 25 x 4,5

`stand(hoek)` zet de slinger (en de plank) op hoek graden (positief = gewicht naar rechts).
"""
import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold, OpType
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
from gereedschap.geom import box3, druppel, extrude, text, vlak_xz

E = 0.01


# --- hulpjes ------------------------------------------------------------------------------------

def zeskant(c, sw):
    """Zeskant met sleutelwijdte sw, platte kanten boven en onder."""
    r = sw / math.sqrt(3)
    return Polygon([(c[0] + r * math.cos(math.radians(60 * k)), c[1] + r * math.sin(math.radians(60 * k))) for k in range(6)])


def stadion(x0, x1, zc, h):
    """Langwerpig gat of staafje van x0 tot x1 met ronde uiteinden, hoogte h rond zc."""
    return LineString([(x0 + h / 2, zc), (x1 - h / 2, zc)]).buffer(h / 2, quad_segs=12)


def draai_y(m, graden, om=(0.0, P.Z_PLANK)):
    """Draai om een as evenwijdig aan y door (x, z) = om; positief = van +z naar +x (gewicht naar rechts...)."""
    a = math.radians(graden)
    c, s = math.cos(a), math.sin(a)
    ox, oz = om
    # (x, z) -> rotatie in het x-z-vlak; positief: onderkant (gewicht) beweegt naar +x
    T = np.array([[c, 0, -s, ox - (c * ox - s * oz)], [0, 1, 0, 0], [s, 0, c, oz - (s * ox + c * oz)]], dtype=float)
    return m.transform(T)


def rev_z(prof, n=64):
    pts = list(orient(Polygon(prof), 1.0).exterior.coords)[:-1]
    return Manifold.revolve(CrossSection([pts]), n)


def nokken():
    """[(x, z)] van alle nokjes die de voorplaat vasthouden: op de pinnen en op de randen."""
    W, R = P.BINNEN_HALF, P.RAND
    rand = [(s * (W + R / 2), z) for z in (15.0, 75.0, 135.0, 195.0) for s in (-1, 1)]
    boven = [(s * x, (P.PLAFOND_Z + P.BORD_H) / 2) for x in (P.INWORP_HALF + 22.0, 80.0) for s in (-1, 1)]
    return list(P.pinnen()) + rand + boven


# --- achterplaat --------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def achterplaat():
    W, R, H = P.BINNEN_HALF, P.RAND, P.BORD_H
    m = vlak_xz(box(-W - R, 0, W + R, H), 0, P.ACHTER_T)
    rand = unary_union([box(-W - R, 0, -W, H), box(W, 0, W + R, H), box(-W - R, P.PLAFOND_Z, W + R, H)])
    gleuf = Polygon([(-P.INWORP_HALF, P.PLAFOND_Z - 1), (P.INWORP_HALF, P.PLAFOND_Z - 1), (P.INWORP_HALF, H - 3),
                     (P.INWORP_HALF + 4, H + 1), (-P.INWORP_HALF - 4, H + 1), (-P.INWORP_HALF, H - 3)])
    rand = rand.difference(gleuf)
    bult = unary_union([Point(xz).buffer(P.BULT_R, quad_segs=16) for xz in P.bulten()])
    rand = unary_union([rand, bult.intersection(box(-W - R, 0, W + R, H))])
    pin = unary_union([Point(xz).buffer(P.PIN_D / 2, quad_segs=16) for xz in P.pinnen()])
    m = m + vlak_xz(unary_union([rand, pin]), P.ACHTER_T - E, P.Y_VOOR)
    nok = unary_union([Point(xz).buffer(P.NOK_D / 2, quad_segs=8) for xz in nokken()])
    top = unary_union([Point(xz).buffer(P.NOK_D / 2 - 0.4, quad_segs=8) for xz in nokken()])
    m = m + vlak_xz(nok, P.Y_VOOR - E, P.Y_VOOR + P.NOK_H - 0.4) + vlak_xz(top, P.Y_VOOR + P.NOK_H - 0.4 - E, P.Y_VOOR + P.NOK_H)
    # versielabel achterop: van achteren gezien loopt x naar rechts, dus niet gespiegeld
    return m - vlak_xz(text(f"E-01 Splitsmijn {P.VERSIE}", 5.0, 0, 100), -1, 0.4)


# --- voorplaat ----------------------------------------------------------------------------------

def tralies():
    """Kijkspleten tussen de pinkolommen, onderbroken bij elke pinrij (dan blijft de plaat stijf)."""
    stukken = []
    kolommen = [s * (P.STEEK / 4 + k * P.STEEK / 2) for k in range(4) for s in (-1, 1)]
    breuken = sorted([P.rij_z(k) for k in range(P.RIJEN)] + [P.Z_PLANK])
    z_lo, z_hi = 6.0, P.PLAFOND_Z - 8.0
    grenzen = [z_lo] + [v for z in breuken for v in (z - 5.0, z + 5.0)] + [z_hi]
    for x in kolommen:
        for a, b in zip(grenzen[::2], grenzen[1::2]):
            if b - a > 8:
                stukken.append(stadion(a, b, 0, P.TRALIE_B))
                stukken[-1] = affinity.rotate(stukken[-1], 90, origin=(0, 0))
                stukken[-1] = affinity.translate(stukken[-1], x, 0)
    return stukken


def beugel():
    """Beugel (los onderdeel) vóór de voorplaat: een voorwand met bovenop de V-groef voor het mes, en twee poten
    naar de voorplaat. De arm zwaait achter de voorwand tussen de poten door.

    Geprint staand (z omhoog): alles begint op het bed, er hangt niets in de lucht. De groef loopt niet tot de
    voorkant door: dat stukje houdt het mes tegen als het naar voren wil schuiven.
    """
    zp = P.Z_PLANK
    z0, zb1 = zp - P.POOT[1], zp + P.GROEF_DIEP                          # bodem van de groef precies op de as
    y0, y1 = P.BORD_T + 0.2, P.MES[1] + 2.0
    delen = [box3(-P.BEUGEL_HALF, P.BEUGEL_Y, z0, P.BEUGEL_HALF, y1, zb1)]
    for a, b in ((-P.BEUGEL_HALF, -P.POOT[0]), (P.POOT[0], P.BEUGEL_HALF)):
        delen.append(box3(a, y0, z0, b, P.BEUGEL_Y + E, zb1))
    m = Manifold.batch_boolean(delen, OpType.Add)
    g = P.GROEF_DIEP / math.tan(math.radians((180 - P.GROEF_HOEK) / 2))
    groef = Polygon([(-g, zb1 + E), (g, zb1 + E), (0, zb1 - P.GROEF_DIEP)])
    m = m - vlak_xz(groef, P.BEUGEL_Y - 1, P.MES[1] + 0.5)
    # gaten voor de pennen van de voorplaat: liggend in de print, dus druppelvormig (punt naar +z)
    px, pz, pd = P.BEUGEL_PEN
    for k in (-1, 1):
        m = m - vlak_xz(druppel((k * px, zp - pz), pd / 2 + 0.05, omhoog=True), y0 - 1, y1 - 2.0)
    return m


def mesblok():
    """Mesblok (los onderdeel): V-mes onderaan op de as, twee gaten voor de pennetjes van de arm.

    Geprint met de achterkant op het bed: het mes staat dan rechtop, de gaten zijn verticaal.
    """
    zp = P.Z_PLANK
    h, (bh, bt) = P.MES_HALF, P.MES_BLOK
    zf = zp + h / math.tan(math.radians(P.MES_HOEK / 2))
    vorm = Polygon([(0, zp), (h, zf), (bh, zf), (bh, zp + bt), (-bh, zp + bt), (-bh, zf), (-h, zf)])
    m = vlak_xz(vorm, *P.MES)
    px, pz, pd = P.MES_PEN
    gaten = unary_union([Point(s * px, zp + pz).buffer(pd / 2 + 0.05, quad_segs=12) for s in (-1, 1)])
    return m - vlak_xz(gaten, P.MES[0] - 1, P.MES[1] + 1)


@lru_cache(maxsize=None)
def voorplaat(met_tralies=True):
    W, R, H = P.BINNEN_HALF, P.RAND, P.BORD_H
    y0, y1 = P.Y_VOOR, P.Y_VOOR + P.VOOR_T
    m = vlak_xz(box(-W - R, 0, W + R, H), y0, y1)
    trechter = Polygon([(-P.TRECHTER_HALF, H + 1), (P.TRECHTER_HALF, H + 1), (P.INWORP_HALF, P.PLAFOND_Z),
                        (-P.INWORP_HALF, P.PLAFOND_Z)])
    gaten = [trechter, Point(0, P.Z_PLANK).buffer(P.NAAF_R + 1.0, quad_segs=24)]
    gaten += [Point(xz).buffer(P.NOK_GAT / 2, quad_segs=8) for xz in nokken()]
    m = m - vlak_xz(unary_union(gaten), y0 - 1, y1 + 1)
    if met_tralies:
        # kijkspleten, aan de binnenkant afgeschuind: een muntje dat tegen de plaat leunt, haakt nergens achter
        spleten = [Manifold.batch_hull([vlak_xz(t.buffer(0.8), y0 - 1, y0), vlak_xz(t, y0 + 0.8, y0 + 0.81)])
                   + vlak_xz(t, y0, y1 + 1) for t in tralies()]
        m = m - Manifold.batch_boolean(spleten, OpType.Add)
    px, pz, pd = P.BEUGEL_PEN
    pennen = unary_union([Point(s * px, P.Z_PLANK - pz).buffer(pd / 2, quad_segs=12) for s in (-1, 1)])
    return m + vlak_xz(pennen, y1 - E, P.MES[1] - 1.0)


# --- kantelplank en slinger ---------------------------------------------------------------------

def plank():
    zp = P.Z_PLANK
    m = vlak_xz(stadion(-P.PLANK_HALF, P.PLANK_HALF, zp, P.PLANK_H), *P.PLANK_Y)
    m = m + vlak_xz(Point(0, zp).buffer(P.NAAF_R, quad_segs=24), P.PLANK_Y[0], P.ARM_Y[0] - 0.3)
    return m + vlak_xz(zeskant((0, zp), P.ZESKANT), P.ARM_Y[0] - 0.3 - E, P.ARM_Y[1] - 0.3)


def slinger():
    """Arm met zeskantgat (door: geen plafond in de print), twee pennetjes voor het mesblok, massief gewicht."""
    zp, L = P.Z_PLANK, P.SLINGER_L
    bw, bh, bd = P.GEWICHT
    bh_x, bt = P.MES_BLOK
    arm = unary_union([box(-P.ARM_B / 2, zp - L, P.ARM_B / 2, zp), Point(0, zp).buffer(8.0, quad_segs=24),
                       box(-bh_x, zp, bh_x, zp + bt)])
    m = vlak_xz(arm, *P.ARM_Y)
    m = m - vlak_xz(zeskant((0, zp), P.ZESKANT + P.ZESKANT_SPEL), P.ARM_Y[0] - 1, P.ARM_Y[1] + 1)
    px, pz, pd = P.MES_PEN
    pennen = unary_union([Point(s * px, zp + pz).buffer(pd / 2, quad_segs=12) for s in (-1, 1)])
    m = m + vlak_xz(pennen, P.ARM_Y[1] - E, P.MES[1] - 1.0)
    gewicht = box(-bw / 2, zp - L - bh / 2, bw / 2, zp - L + bh / 2)
    m = m + vlak_xz(gewicht, P.ARM_Y[0], P.ARM_Y[0] + bd)
    # duwvlak: een ondiepe kuil in de voorkant van het gewicht, hier duwt het kind
    kuil = Point(0, zp - L).buffer(9.0, quad_segs=24)
    return m - vlak_xz(kuil, P.ARM_Y[0] + bd - 1.0, P.ARM_Y[0] + bd + 1)


# --- voet (pasproef) ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def voet():
    """Lichte voet: bodem, dunne wanden, twee bakjes, verdeler met vin, U-vormige kokers voor de zijranden."""
    x0, (y0, y1), (z0, z1), w = P.VOET_HALF, P.VOET_Y, P.VOET_Z, P.VOET_WAND
    W, R, s = P.BINNEN_HALF, P.RAND, P.SPEL
    delen = [box3(-x0, y0, z0, x0, y1, z0 + w),                             # bodem
             box3(-x0, y0, z0, x0, y0 + w, 0),                              # achterwand
             box3(-x0, y1 - w, z0, x0, y1, z0 + P.VOORRAND_H)]              # lage voorrand
    for k in (-1, 1):
        lo, hi = sorted((k * (x0 - w), k * x0))
        delen.append(box3(lo, y0, z0, hi, y1, 0))                          # zijwand
        a, b = sorted((k * (W - s), k * x0))
        kb = box3(a, -s - w, z0, b, P.BORD_T + s + w, z1)                   # koker om de zijrand, open naar binnen
        a, b = sorted((k * (W - s - 5.0), k * (W + R + s)))
        delen.append(kb - box3(a, -s, 0, b, P.BORD_T + s, z1 + 1))
    # verdeler: wand tussen de bakken, met een vin die in de spleet steekt
    h = P.VERDELER_T / 2
    delen.append(box3(-h, y0 + w - E, z0 + w - E, h, y1 - w + E, 0))
    vin = Polygon([(-h, -E), (h, -E), (h, P.VERDELER_TOP - h), (0.4, P.VERDELER_TOP - 0.4), (-0.4, P.VERDELER_TOP - 0.4),
                   (-h, P.VERDELER_TOP - h)])
    delen.append(vlak_xz(vin, P.ACHTER_T + 0.6, P.Y_VOOR - 0.6))
    m = Manifold.batch_boolean(delen, OpType.Add)
    # L en R in de bodem van de bakjes (bovenop: print zonder support). Het kind kijkt van +y: links is +x,
    # en de letters staan 180° gedraaid zodat ze vanaf de voorkant leesbaar zijn
    for x, t in ((55.0, "L"), (-55.0, "R")):
        letter = affinity.rotate(text(t, 14.0, x, (y0 + y1) / 2), 180, origin=(x, (y0 + y1) / 2))
        m = m - extrude(letter, z0 + w - 0.6, z0 + w + 1)
    return m


# --- munt ---------------------------------------------------------------------------------------

def diamant2d(c=(0.0, 0.0), b=15.0):
    """Lijntekening van een geslepen diamant (kroon met facetten, punt onder), breedte b."""
    x, y = c
    h_k, h_p = 0.28 * b, 0.55 * b
    rand = Polygon([(x - b / 2, y), (x - 0.3 * b, y + h_k), (x + 0.3 * b, y + h_k), (x + b / 2, y), (x, y - h_p)])
    lijnen = [LineString([(x - b / 2, y), (x + b / 2, y)]), LineString([(x - 0.12 * b, y + h_k), (x - 0.22 * b, y), (x, y - h_p)]),
              LineString([(x + 0.12 * b, y + h_k), (x + 0.22 * b, y), (x, y - h_p)])]
    omtrek = rand.exterior.buffer(0.45)
    return unary_union([omtrek] + [l.buffer(0.35) for l in lijnen]).intersection(rand.buffer(0.45))


def munt():
    r, t, f = P.MUNT_D / 2, P.MUNT_T, P.MUNT_FAS
    m = rev_z([(0, 0), (r - f, 0), (r, f), (r, t - f), (r - f, t), (0, t)], n=72)
    return m - extrude(diamant2d((0, 1.2), 15.0), t - 0.5, t + 1)


# --- montage ------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def delen():
    """{naam: (kleur, Manifold, beweegt)} in de ruststand."""
    return {
        "achterplaat": (P.ZWART, achterplaat(), False),
        "voorplaat": (P.ZWART, voorplaat(), False),
        "voet": (P.BRUIN, voet(), False),
        "beugel": (P.BRUIN, beugel(), False),
        "plank": (P.ROOD, plank(), True),
        "slinger": (P.ROOD, slinger(), True),
        "mesblok": (P.ROOD, mesblok(), True),
    }


def munt_op(x, z, y=None):
    """Muntje rechtop in de spleet (as langs y), midden op (x, z)."""
    m = munt().translate([0, 0, -P.MUNT_T / 2]).rotate([90, 0, 0])
    return m.translate([x, (P.ACHTER_T + P.Y_VOOR) / 2 if y is None else y, z])


def munt_plat(x, y, z):
    return munt().translate([x, y, z])


def stand(hoek=0.0, munten=()):
    """Alle onderdelen bij slingerhoek `hoek`; munten: [(x, z)] rechtop in de spleet."""
    out = {}
    for naam, (kleur, m, beweegt) in delen().items():
        out[naam] = (kleur, draai_y(m, hoek) if beweegt else m)
    for i, (x, z) in enumerate(munten):
        out[f"munt {i + 1}"] = (P.BLAUW, munt_op(x, z))
    return out
