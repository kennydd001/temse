"""Geometrie van B-34B Repair Pod: alle onderdelen, de stand bij elke armhoek, en de printligging.

Elk onderdeel wordt gebouwd in de montage-assen in de DICHTE stand (zie params.py). `stand(theta)`
plaatst ze bij armhoek theta: de bakken schuiven (parallellogram: ze draaien niet), de armen draaien
om hun voetpunt, de heugel schuift omhoog.

Onderdelen (kleur):
  voet       Charcoal   voetplaat, middenrug met zwaluwstaartgroef, flanken, draaipunten, aanslagen
  dak        Charcoal   dekt de dichte bakken af; bitkam met 10 zeskantgaten voor 1/4"-bits
  bak_r/l    Ivory      twee schroevenbakjes met elk 2 vakken
  arm_*      Orange     8 platte armen; de binnenste voorarmen dragen de tandsector
  schuif     Orange     heugel met tanden aan beide kanten, zwaluwstaart, knop boven het dak
  pen        Orange     16 klikpennen (+2 reserve)
"""
import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

import _pad  # noqa: F401  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import afgeronde_plaat, extrude, rounded_rect, text
from gereedschap.tandwiel import heugel, sector, tandwiel
from params import *  # noqa: F401,F403
import params as P

# --- hulpjes ------------------------------------------------------------------------------------


def xz(g, y0, y1):
    """2D-vorm in het x-z-vlak (vooraanzicht) extruderen van y0 tot y1."""
    m = extrude(g, -y1, -y0)
    return m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))


def yz(g, x0, x1):
    """2D-vorm in het y-z-vlak (zijaanzicht, eerste coördinaat y) extruderen van x0 tot x1."""
    m = extrude(g, x0, x1)
    return m.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))


def cil_y(cx, cz, r, y0, y1, n=48):
    return xz(Point(cx, cz).buffer(r, quad_segs=n // 4), y0, y1)


def draai_xz(m, graden, om):
    """Draai een Manifold in het x-z-vlak (linksom gezien van voren) om punt om=(x, z)."""
    a = math.radians(graden)
    c, s = math.cos(a), math.sin(a)
    # (x, z) -> (c x - s z, s x + c z), om het punt
    T = np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0]], dtype=float)
    ox, oz = om
    T[0, 3] = ox - (c * ox - s * oz)
    T[2, 3] = oz - (s * ox + c * oz)
    return m.transform(T)


def spiegel(m):
    return m.mirror([1, 0, 0])


def stadion(a, b, r):
    return LineString([a, b]).buffer(r, quad_segs=24)


Y_ARM_V = (-ARM_SPLEET - ARM_T, -ARM_SPLEET)                 # voorarmen
Y_ARM_A = (DIEPTE + ARM_SPLEET, DIEPTE + ARM_SPLEET + ARM_T)  # achterarmen
P_STEEK = math.pi * M


# --- bak ----------------------------------------------------------------------------------------

def bak_doorsnede():
    """Vooraanzicht van de rechterbak (dicht): 45°-afschuining onderaan buiten (printbaar), ronde bovenrand buiten."""
    x0, x1, z0, z1 = BAK_X0, BAK_X0 + BAK_B, BAK_Z0, BAK_TOP
    c, r = 9.0, BAK_R
    boog = [(x1 - r + r * math.cos(a), z1 - r + r * math.sin(a)) for a in np.radians(np.linspace(0, 90, 16))]
    return Polygon([(x0, z0), (x1 - c, z0), (x1, z0 + c)] + boog + [(x0, z1)])


def bak():
    x0, x1, z0, z1 = BAK_X0, BAK_X0 + BAK_B, BAK_Z0, BAK_TOP
    buiten = xz(bak_doorsnede(), 0, DIEPTE)
    # vorm omhoog doortrekken (open bovenkant) en ZIJWAND naar binnen; binnen in de ronde bovenrand een
    # 45°-schuinte die de ronding op ZIJWAND raakt: buiten rond, binnen printbaar zonder overhang
    ext = unary_union([bak_doorsnede(), box(x0, z1 - BAK_R - 1, x1, z1 + 12)])
    binnen = ext.buffer(-ZIJWAND, join_style="mitre").intersection(box(x0, z0 + BODEM, x1, z1 + 20))
    k = (x1 - BAK_R) + (z1 - BAK_R) + (BAK_R - ZIJWAND) * math.sqrt(2)
    binnen = binnen.intersection(Polygon([(x0 - 1, z0 - 1), (k - z0 + 1, z0 - 1), (k - z1 - 21, z1 + 20), (x0 - 1, z1 + 20)]))
    binnen = binnen.buffer(-4.0).buffer(4.0)                               # ronde binnenhoeken onderin
    binnen = unary_union([binnen, box(x0 + ZIJWAND, z1 - 1, k - z1 - 1, z1 + 20)])
    m = buiten - xz(binnen, WAND_VOOR, DIEPTE - WAND_VOOR)
    # vakken: scheidingswand dwars op y, iets lager dan de rand
    tussen = DIEPTE / VAKKEN
    for k in range(1, VAKKEN):
        y = k * tussen
        wand = xz(box(x0 + 1, z0 + 1, x1 - 1, z1 - 4.0), y - 0.8, y + 0.8) ^ buiten
        m = m + wand
    # oren: verdikking rond de draaipunten, 1 mm boven de rand
    for tx, tz in (P.T1_DICHT, P.T2_DICHT):
        for y0, y1 in ((0, WAND_VOOR), (DIEPTE - WAND_VOOR, DIEPTE)):
            m = m + cil_y(tx, tz, OOG_R + 0.3, y0, y1)
            m = m - xz(druppel((tx, tz), GAT_WAND / 2, omhoog=True), y0 - 1, y1 + 1)
        # ruimte voor de klikhaakjes achter de wand (ook in de binnenste zijwand)
        m = m - xz(druppel((tx, tz), 3.4, omhoog=True), WAND_VOOR - 0.01, WAND_VOOR + 2.8)
        m = m - xz(druppel((tx, tz), 3.4, omhoog=True), DIEPTE - WAND_VOOR - 2.8, DIEPTE - WAND_VOOR + 0.01)
    return m


# --- armen --------------------------------------------------------------------------------------

def arm2d(p, met_tandwiel=False):
    """Arm van voetpunt p naar de bak, in de dichte stand; optioneel met tandsector rond p."""
    t = P.arm_punt(p, THETA_DICHT)
    g = stadion(p, t, ARM_B / 2)
    g = unary_union([g, Point(p).buffer(OOG_R, quad_segs=24), Point(t).buffer(OOG_R, quad_segs=24)])
    if met_tandwiel:
        wiel = tandwiel(M, Z_WIEL, speling=SPELING_TAND)
        # tand recht naar links (180°) in de dichte stand; de sector dekt de ingrijping over de hele
        # slag plus een tand extra aan beide kanten
        st = sector(wiel, M, Z_WIEL, 180 - 25, 180 + (THETA_DICHT - THETA_OPEN) + 25, naaf=OOG_R)
        g = unary_union([g, affinity.translate(st, p[0], p[1])])
    gaten = unary_union([Point(p).buffer(GAT_ARM / 2, quad_segs=24), Point(t).buffer(GAT_ARM / 2, quad_segs=24)])
    return g.difference(gaten)


def arm(p, voor=True, met_tandwiel=False):
    y0, y1 = Y_ARM_V if voor else Y_ARM_A
    return xz(arm2d(p, met_tandwiel), y0, y1)


# --- schuif (heugel) ----------------------------------------------------------------------------

HEUGEL_TANDEN = range(-4, 2)       # tandnummers; tand k op z = P1z + p/2 + k p (dicht)
SCHUIF_ONDER = 1.5
STANG_HALF = 4.0                   # boven de tanden: rechte stang
SCHUIF_HALF = HEUGEL_HALF + 1.25 * M   # tandtoppen van de heugel: de schuif gaat bij de montage door het dak


def schuif2d():
    z0 = P1[1] + P_STEEK / 2
    rechts = heugel(M, len(HEUGEL_TANDEN), speling=SPELING_TAND, dikte=0.0, eerste=HEUGEL_TANDEN[0])
    # heugel(): tanden langs y naar +x, steeklijn x = 0  ->  hier: z = y, steeklijn op x = HEUGEL_HALF
    rechts = affinity.translate(rechts, HEUGEL_HALF, z0)
    links = affinity.scale(rechts, -1, 1, origin=(0, 0))
    voet_lijn = HEUGEL_HALF - 1.25 * M
    z_tand_top = z0 + HEUGEL_TANDEN[-1] * P_STEEK + P_STEEK / 2
    rug = box(-voet_lijn - 0.01, SCHUIF_ONDER, voet_lijn + 0.01, z_tand_top + 2)
    tanden = unary_union([rechts, links, rug]).intersection(box(-10, SCHUIF_ONDER, 10, z_tand_top))
    knop_z0 = DAK_Z0 + DAK_T
    stang = box(-STANG_HALF, z_tand_top - 0.5, STANG_HALF, knop_z0 + 1)
    return unary_union([tanden, stang]).buffer(0), knop_z0


def zwaluw2d(speling=0.0):
    """Zwaluwstaart in het x-y-vlak: smal aan de voorkant (y = -ARM_SPLEET), breed achteraan."""
    b0, b1, d = ZWALUW
    y0 = -ARM_SPLEET
    return Polygon([(-b0 / 2 - speling, y0 - 0.01), (b0 / 2 + speling, y0 - 0.01),
                    (b1 / 2 + speling, y0 + d + speling), (-b1 / 2 - speling, y0 + d + speling)])


def schuif():
    g, knop_z0 = schuif2d()
    m = xz(g, *Y_ARM_V)
    z_tong1 = knop_z0 + 1
    tong = extrude(zwaluw2d(), SCHUIF_ONDER, z_tong1)
    m = m + tong
    kx, ky, kz = KNOP
    knop = extrude(box(-kx / 2, Y_ARM_V[0], kx / 2, Y_ARM_V[0] + ky), knop_z0, knop_z0 + kz)
    # bovenkant afronden en twee grip-groefjes
    rond = xz(rounded_rect(-kx / 2, knop_z0 - 5, kx / 2, knop_z0 + kz, 3.5), Y_ARM_V[0] - 1, Y_ARM_V[0] + ky + 1)
    knop = knop ^ rond
    for gx in (-6, 0, 6):
        knop = knop - xz(Point(gx, knop_z0 + kz + 0.6).buffer(1.3), Y_ARM_V[0] - 1, Y_ARM_V[0] + ky + 1)
    return m + knop


# --- dak ----------------------------------------------------------------------------------------

def zeskant(cx, cy, sw, draai=30.0):
    """Zeskant met sleutelwijdte sw; standaard (30°) met de platte kanten naar +-x."""
    r = sw / math.sqrt(3)
    return Polygon([(cx + r * math.cos(math.radians(draai + 60 * k)), cy + r * math.sin(math.radians(draai + 60 * k)))
                    for k in range(6)])


PEN_DAK = [(0.0, 23.0), (0.0, 65.0)]   # pennen op de rug die in het dak steken
Y_DAK = (-8.0, DIEPTE + 8.0)


def dak():
    z0, z1 = DAK_Z0, DAK_Z0 + DAK_T
    m = afgeronde_plaat(-DAK_HALF, Y_DAK[0], DAK_HALF, Y_DAK[1], z0, z1, 10.0, 4.5)
    kam = afgeronde_plaat(-KAM_HALF, 4.0, KAM_HALF, DIEPTE - 4.0, z1 - 0.5, z1 + KAM_H, 5.0, 4.0)
    m = m + kam
    top = z1 + KAM_H
    gaten = unary_union([zeskant(x, y, BIT_SW) for x, y in BITS])
    m = m - extrude(gaten, top - BIT_DIEP, top + 1)
    # inkeping voor de schuif aan de voorkant, zo breed als de heugel: de schuif gaat er bij de montage
    # van boven doorheen (de knop is breder dan de inkeping, dus het dak kan er niet over)
    inkeping = unary_union([box(-SCHUIF_HALF - 0.35, Y_DAK[0] - 1, SCHUIF_HALF + 0.35, Y_ARM_V[1] + 0.35),
                            zwaluw2d().buffer(0.35, join_style="mitre")])
    m = m - extrude(inkeping, z0 - 1, top + 1)
    for x, y in PEN_DAK:
        m = m - extrude(Point(x, y).buffer(3.15, quad_segs=24), z0 - 1, z0 + 4.5)
    return m


# --- voet ---------------------------------------------------------------------------------------

def zwaai_rechts(stappen=40):
    """Alle plekken (vooraanzicht) waar de rechterbak tijdens de slag komt."""
    d = bak_doorsnede()
    ts = np.linspace(THETA_OPEN, THETA_DICHT, stappen)
    vormen = [affinity.translate(d, *P.verschuiving(t)) for t in ts]
    # tussen twee standen de bolle omhullende, zodat er geen gaatjes tussen de stappen vallen
    return unary_union([unary_union([a, b]).convex_hull for a, b in zip(vormen[:-1], vormen[1:])])


def aanslag_open():
    """Middelpunt van de nok die de binnenarm in de open stand tegenhoudt (zit helemaal op de voet)."""
    t = math.radians(THETA_OPEN)
    rho, r = 16.0, 3.0
    d = ARM_B / 2 + r
    return (P1[0] + rho * math.cos(t) + d * math.sin(t), P1[1] + rho * math.sin(t) - d * math.cos(t)), r


def druppel(c, r, omhoog=False):
    """Cirkel met een 45°-punt: horizontaal printbaar zonder support.

    Een uitsteeksel (pen, nok) krijgt de punt onder; een holte de punt boven (dan is het plafond schuin).
    """
    cx, cz = c
    s = 1 if omhoog else -1
    return unary_union([Point(c).buffer(r, quad_segs=24),
                        Polygon([(cx - r / math.sqrt(2), cz + s * r / math.sqrt(2)), (cx, cz + s * r * math.sqrt(2)),
                                 (cx + r / math.sqrt(2), cz + s * r / math.sqrt(2)), (cx, cz)])])


def voet():
    # voetplaat (olifantenvoet: de slicer compenseert de eerste laag zelf)
    plaat = extrude(rounded_rect(-VOET_HALF, 0, VOET_HALF, DIEPTE, 12.0), 0, VOET_H)
    # flanken: alles onder de bakken wat de zwaaiende bak nooit raakt
    vrij = zwaai_rechts().buffer(SPELING_ZWAAI)
    ruim = box(RUG_HALF - 0.01, VOET_H - 0.01, VOET_HALF - 2.0, BAK_Z0).difference(vrij)
    flank = ruim.buffer(-1.0).buffer(1.0)                                   # geen splinters
    flank = unary_union([flank, ruim.intersection(box(0, VOET_H - 0.01, VOET_HALF, VOET_H + 3))])  # scherpe voet
    flank = unary_union([flank, affinity.scale(flank, -1, 1, origin=(0, 0))])
    rug = box(-RUG_HALF, 0, RUG_HALF, DAK_Z0)
    lichaam = xz(unary_union([flank, rug]).buffer(0), 0, DIEPTE)
    # lichaam binnen de (afgeronde) voetplaat houden: anders hangen de hoeken over
    lichaam = lichaam ^ extrude(rounded_rect(-VOET_HALF + 2.0, 0, VOET_HALF - 2.0, DIEPTE, 10.0), 0, DAK_Z0 + 1)
    m = plaat + lichaam
    # zwaluwstaartgroef voor de schuif, voorin de rug en de voet
    m = m - extrude(zwaluw2d().buffer(ZWALUW_SPEL, join_style="mitre"), 0.8, DAK_Z0 + 1)
    # versielabel in de onderkant, gespiegeld zodat het van onderen leesbaar is
    label = affinity.scale(text(f"B-34B Repair Pod {P.VERSIE}", 5.0, 0, DIEPTE / 2), -1, 1, origin=(0, 0))
    m = m - extrude(label, -0.01, 0.4)
    # pennen bovenop voor het dak
    for x, y in PEN_DAK:
        m = m + extrude(Point(x, y).buffer(3.0, quad_segs=24), DAK_Z0 - 0.01, DAK_Z0 + 4.0)
    # draaipunten voor en achter (beide kanten), met een holte erachter voor de klikhaakjes
    for s in (1, -1):
        for px, pz in (P1, P2):
            x = s * px
            m = m - xz(druppel((x, pz), GAT_WAND / 2, omhoog=True), -1, WAND_VOOR + 0.01)
            m = m - xz(druppel((x, pz), GAT_WAND / 2, omhoog=True), DIEPTE - WAND_VOOR - 0.01, DIEPTE + 1)
            m = m - xz(druppel((x, pz), 4.2, omhoog=True), WAND_VOOR, WAND_VOOR + 4.5)
            m = m - xz(druppel((x, pz), 4.2, omhoog=True), DIEPTE - WAND_VOOR - 4.5, DIEPTE - WAND_VOOR)
        # nok voor de binnenarm in de open stand (voor en achter), met een vinnetje eronder tot de tafel:
        # anders hangt de onderste laag van de nok 5,5 mm vrij uit de wand
        (ax, az), r = aanslag_open()
        nok = unary_union([druppel((s * ax, az), r), box(s * ax - 1.0, 0, s * ax + 1.0, az)])
        m = m + xz(nok, Y_ARM_V[0] - 0.3, 0.01)
        m = m + xz(nok, DIEPTE - 0.01, Y_ARM_A[1] + 0.3)
    return m


# --- pen ----------------------------------------------------------------------------------------

PEN_LENGTE = KOP_T + ARM_T + ARM_SPLEET + WAND_VOOR + 2.0


PEN_WAND = PEN_SPEL + ARM_T + ARM_SPLEET + WAND_VOOR    # van de binnenkant van de kop tot achter de wand


def pen():
    """Klikpen langs +y: kop y = [-KOP_T, 0], schacht door arm en wand, haakjes erachter.

    De gleuf (x = +-0,65) splitst de pen in twee veren die naar elkaar toe buigen als het haakje
    (0,3 mm groter dan het gat) door het wandgat gaat: rek ~1,4 %, ook dwars op de lagen veilig.
    Geprint staand op de kop: dan is de schacht precies rond.
    """
    w = PEN_WAND
    r, rk = PEN_D / 2, PEN_KOP_D / 2
    # één omwentelingsprofiel (straal, as): kop, schacht, haakje met 45°-vangkant, afgeschuinde punt
    prof = [(0.0, -KOP_T), (rk, -KOP_T), (rk, 0.0), (r, 0.0), (r, w + 0.05), (2.8, w + 0.35), (2.8, w + 0.65),
            (2.2, w + 2.25), (0.0, w + 2.25)]
    m = Manifold.revolve(CrossSection([prof]), 48)
    m = m.transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float))   # as z -> y
    return m - Manifold.cube([1.3, w + 3, 20]).translate([-0.65, 1.0, -10])            # gleuf


def pen_op(x, z, voor=True):
    """Pen in de montage: kop tegen de buitenkant van de arm, steekt de wand in."""
    m = pen()
    if voor:
        return m.translate([x, Y_ARM_V[0] - PEN_SPEL, z])
    return m.mirror([0, 1, 0]).translate([x, Y_ARM_A[1] + PEN_SPEL, z])


# --- montage ------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def onderdelen():
    """{naam: (kleur, Manifold)} in de dichte stand."""
    br = bak()
    d = {
        "voet": (ZWART, voet()),
        "dak": (ZWART, dak()),
        "bak rechts": (WIT, br),
        "bak links": (WIT, spiegel(br)),
        "schuif": (ORANJE, schuif()),
    }
    for kant, s in (("rechts", 1), ("links", -1)):
        for naam, p, tw in (("binnenarm", P1, True), ("buitenarm", P2, False)):
            for voor in (True, False):
                a = arm(p, voor, met_tandwiel=tw and voor)
                d[f"{naam} {'voor' if voor else 'achter'} {kant}"] = (ORANJE, a if s > 0 else spiegel(a))
    return d


@lru_cache(maxsize=None)
def pennen_dicht():
    """Alle pennen in de dichte stand: (naam, Manifold, beweegt mee met)."""
    out = []
    for s, kant in ((1, "rechts"), (-1, "links")):
        for (px, pz), (tx, tz), naam in ((P1, P.T1_DICHT, "binnenarm"), (P2, P.T2_DICHT, "buitenarm")):
            for voor in (True, False):
                out.append((f"pen voet {naam} {kant} {'v' if voor else 'a'}", pen_op(s * px, pz, voor), "voet"))
                out.append((f"pen bak {naam} {kant} {'v' if voor else 'a'}", pen_op(s * tx, tz, voor), f"bak {kant}"))
    return out


def stand(theta):
    """Alle onderdelen (incl. pennen) bij armhoek theta: {naam: (kleur, Manifold)}."""
    dx, dz = P.verschuiving(theta)
    out = {}
    for naam, (kleur, m) in onderdelen().items():
        if naam.startswith("bak"):
            sx = 1 if "rechts" in naam else -1
            m = m.translate([sx * dx, 0, dz])
        elif "arm" in naam:
            sx = 1 if "rechts" in naam else -1
            p = P1 if "binnen" in naam else P2
            m = draai_xz(m, sx * (theta - THETA_DICHT), (sx * p[0], p[1]))
        elif naam == "schuif":
            m = m.translate([0, 0, P.heugel_slag(theta)])
        out[naam] = (kleur, m)
    for naam, m, mee in pennen_dicht():
        if mee.startswith("bak"):
            sx = 1 if "rechts" in mee else -1
            m = m.translate([sx * dx, 0, dz])
        out[naam] = (ORANJE, m)
    return out
