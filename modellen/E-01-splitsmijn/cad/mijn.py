"""Het eindmodel van E-01: De Splitsmijn. Hetzelfde valbord als de pasproef, in een mijn.

Onderdelen (assen van het valbord, zie params.py; de slinger in de ruststand):
  achterplaat    die van de pasproef, ongewijzigd
  voorplaat      die van de pasproef; de pennen voor de beugel zijn VOOR_DY langer
  mijnhout       bruin raamwerk van balken vóór de voorplaat (los: een tweede kleur op de voorplaat zou een
                 spoeltoren vragen, en daar is naast een plaat van 218 x 230 geen plaats voor)
  portaal        de mijningang met de naam: een plaat met twee tongen die in de kokerkappen steken
  kokerkappen    klemmen over de bovenhoeken van het bord en het mijnhout; bovenin staat de tong van het portaal
  houweel        de slinger: een bruine steel (met het mesblok bovenaan) en een gele kop, die op de steel klemt
  mijnberg       de voet, ondersteboven geprint: een dek met twee trechters naar twee tunnels, uitsparingen voor
                 de kokervoeten, een sleuf voor de verdelervin
  kokervoeten    klemmen in het dek; in hun U-koker staan de zijranden van het bord en de staanders van het mijnhout
  karretjes      twee laatjes in de tunnels: de muntjes vallen erin en blijven verborgen tot je ze uittrekt
  verdelervin    steekt uit het dek omhoog in de spleet, recht onder de middelste pin
  tienveld       plankje met 2 x 5 kuiltjes om de muntjes op te leggen
  munten         half blauw (met de diamant), half geel: zo leg je de splitsing in twee kleuren

Alles vóór de voorplaat (slinger, mesblok, beugel) staat VOOR_DY naar voren, zodat de arm over de balken zwaait.
"""
import math
import random
from functools import lru_cache

import numpy as np
from manifold3d import Manifold, OpType
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
import valbord as V
from gereedschap.geom import box3, extrude, rounded_rect, text, vlak_xz, vlak_yz

E = 0.01
DY = P.VOOR_DY
ARM_Y, MES, BEUGEL_Y = V._vooraan(DY)
ZH = P.Z_PLANK - P.SLINGER_L          # midden van de houweelkop


def _gespiegeld(g, cx):
    """Tekst in het x-z-vlak zo zetten dat ze van voren (+y) leesbaar is, met het midden op x = cx."""
    return affinity.translate(affinity.scale(g, -1, 1, origin=(0, 0)), cx, 0)


def _relief(g, y0, h, schuin=None):
    """2D-vorm (x, z) h hoog op een vlak y = y0 dat naar +y kijkt. schuin: rand onder 45° (alleen bolle vormen)."""
    if not schuin:
        return vlak_xz(g, y0 - E, y0 + h)
    return Manifold.batch_hull([vlak_xz(g, y0 - E, y0), vlak_xz(g.buffer(-schuin), y0 + h - E, y0 + h)])


# --- voorplaat: balken en naam --------------------------------------------------------------------

def _trechter():
    H = P.BORD_H
    return Polygon([(-P.TRECHTER_HALF, H + 1), (P.TRECHTER_HALF, H + 1), (P.INWORP_HALF, P.PLAFOND_Z),
                    (-P.INWORP_HALF, P.PLAFOND_Z)])


def balken2d():
    """Staanders over de zijranden, dwarsbalken op de pinrijen (daar zijn de kijkspleten onderbroken), een balk
    op de hoogte van de plank (niet onder de beugel) en een bovenbalk naast de trechter."""
    s0, s1 = P.STIJL
    h = P.BALK_H / 2
    zp = P.Z_PLANK
    delen = [box(-s1, 0, -s0, P.BORD_H), box(s0, 0, s1, P.BORD_H)]
    delen += [box(-s0 - E, z - h, s0 + E, z + h) for z in (P.rij_z(k) for k in range(P.RIJEN))]
    delen += [box(-s0 - E, zp - h, -P.BEUGEL_HALF - 4, zp + h), box(P.BEUGEL_HALF + 4, zp - h, s0 + E, zp + h)]
    delen.append(box(-s1, P.PLAFOND_Z + 1, s1, P.BORD_H).difference(_trechter().buffer(2.0)))
    return unary_union(delen)


def nerf():
    """Houtnerf: korte groefjes in de lengte van elke balk, en spijkerkoppen waar de balken de staanders raken."""
    rnd = random.Random(5)
    s0, s1 = P.STIJL
    lijnen = []
    for z in [P.rij_z(k) for k in range(P.RIJEN)] + [P.Z_PLANK]:
        for dz in (-1.8, 1.6):
            x = -s0 + 3
            while x < s0 - 3:
                lengte = rnd.uniform(12, 34)
                lijnen.append(LineString([(x, z + dz), (min(x + lengte, s0 - 3), z + dz + rnd.uniform(-0.4, 0.4))]))
                x += lengte + rnd.uniform(4, 10)
    for s in (-1, 1):
        xc = s * (s0 + s1) / 2
        for dx in (-3.5, 3.2):
            z = 3.0
            while z < P.BORD_H - 3:
                lengte = rnd.uniform(18, 40)
                lijnen.append(LineString([(xc + dx, z), (xc + dx + rnd.uniform(-0.4, 0.4), min(z + lengte, P.BORD_H - 3))]))
                z += lengte + rnd.uniform(5, 12)
    groef = unary_union([l.buffer(0.3, cap_style="round") for l in lijnen])
    spijkers = [Point(s * (s0 - 3.5), z).buffer(1.1, quad_segs=8)
                for s in (-1, 1) for z in [P.rij_z(k) for k in range(P.RIJEN)]]
    return unary_union([groef] + spijkers).intersection(balken2d().buffer(-0.8))


@lru_cache(maxsize=None)
def mijnhout():
    """Het houten raamwerk vóór de voorplaat (los onderdeel, plat geprint). Het klemt zonder lijm: onderaan staan
    de staanders mee met de zijranden van het bord in de kokervoeten, bovenaan klemt het portaal erover, en nokjes
    achterop houden het op zijn plaats."""
    y = P.BORD_T
    m = vlak_xz(balken2d(), y + 0.05, y + P.RELIEF_T)
    m = m - vlak_xz(nerf(), y + P.RELIEF_T - 0.4, y + P.RELIEF_T + 1)
    # nokjes achterop: ze zakken 0,8 mm in de gaatjes van de voorplaat (de nokjes van de achterplaat komen er
    # van achteren 1,6 in): dan schuift het raamwerk nergens. Geprint met de voorkant op het bed: nokjes omhoog
    nok = unary_union([Point(xz).buffer(P.NOK_D / 2 - 0.1, quad_segs=8) for xz in V.nokken()])
    diepte = P.VOOR_T - P.NOK_H - 0.1
    return m + vlak_xz(nok, y - diepte, y + 0.05 + E)


# --- mijningang --------------------------------------------------------------------------------------

def portaal2d():
    """Dwarsbalk met de naam en twee tongen die in de kokerkappen steken (van voren gezien)."""
    z0, z1 = P.PORTAAL_Z
    x0, x1 = _kap_x()
    balk = rounded_rect(-P.PORTAAL_X, z0, P.PORTAAL_X, z1, 2.0)
    zt = P.BORD_H + P.KAP_LAT
    tongen = [box(x0 + P.SPEL, zt, P.BINNEN_HALF + P.RAND, z0 + 6), box(-P.BINNEN_HALF - P.RAND, zt, -x0 - P.SPEL, z0 + 6)]
    return balk, unary_union(tongen)


def letters2d():
    z0, z1 = P.PORTAAL_Z
    return _gespiegeld(text("SPLITSMIJN", P.LETTER_H, 0, (z0 + z1) / 2), 0.0)


@lru_cache(maxsize=None)
def portaal_delen():
    """{naam: (kleur, Manifold)}: het bruine portaal en de letters (zwart, als ingebrand). Eén vlakke plaat, geprint
    met de achterkant op het bed: de letters liggen dan in de bovenste lagen."""
    balk2d, tong2d = portaal2d()
    y0, y1 = P.PORTAAL_Y
    m = vlak_xz(balk2d.union(tong2d), y0, y1)
    letters = letters2d()
    m = m - vlak_xz(letters, y1 - P.LETTER_IN, y1 + 1)
    nerf_ = unary_union([LineString([(-P.PORTAAL_X + 4, z), (P.PORTAAL_X - 4, z)]).buffer(0.3)
                         for z in (P.PORTAAL_Z[0] + 2.5, P.PORTAAL_Z[1] - 2.5)])
    m = m - vlak_xz(nerf_, y1 - 0.4, y1 + 1)
    return {"portaal": (P.BRUIN, m), "letters": (P.ZWART, vlak_xz(letters, y1 - P.LETTER_IN - E, y1))}


# --- houweel ---------------------------------------------------------------------------------------

def _boog(x_punt, z_punt, z_midden):
    """Cirkel door (-x_punt, z_punt), (0, z_midden) en (x_punt, z_punt): (middelpunt z, straal)."""
    s = (x_punt ** 2 + z_punt ** 2 - z_midden ** 2) / (2 * (z_punt - z_midden))
    return s, abs(s - z_midden)


def kop2d():
    """Houweelkop van voren: een sikkel met twee stompe punten die naar de steel toe buigen, en het oog
    (het blok waar de steel in zit) in het midden."""
    half, dik = P.KOP_B / 2, P.KOP_DIK
    zo, ro = _boog(half, P.KOP_PUNT, -dik * 0.6)
    zb, rb = _boog(half, P.KOP_PUNT, dik * 0.4)
    sikkel = Point(0, zo).buffer(ro, quad_segs=96).difference(Point(0, zb).buffer(rb, quad_segs=160))
    sikkel = sikkel.intersection(box(-half, -40, half, 40))
    ob, oh = P.KOP_OOG
    oog = rounded_rect(-ob, -oh, ob, oh, 3.0)
    g = unary_union([sikkel, oog]).buffer(-2.5, quad_segs=12).buffer(2.5, quad_segs=12)   # stompe punten
    return affinity.translate(g, 0, ZH)


def steel():
    """Slingerarm als steel: de arm van de pasproef, tot onder het midden van de kop, met houtnerf en twee
    pennetjes voor de kop."""
    m = V.arm(DY, onder=ZH - P.STEEL_ONDER)
    nerf_ = unary_union([LineString([(x, ZH + P.KOP_OOG[1] + 4), (x + 0.3, P.Z_PLANK - 14)]).buffer(0.3)
                         for x in (-1.9, 1.7)])
    m = m - vlak_xz(nerf_, ARM_Y[1] - 0.4, ARM_Y[1] + 1)
    pennen = unary_union([Point(0, ZH + dz).buffer(P.MES_PEN[2] / 2, quad_segs=12) for dz in P.KOP_PEN])
    return m + vlak_xz(pennen, ARM_Y[1] - E, ARM_Y[1] + 3.0)


def kop():
    """Houweelkop: van achteren een gleuf waar de steel in ligt, twee gaten voor de pennetjes (klemmen, MES_GAT),
    vooraan een kuiltje: hier duwt het kind. Geprint met de achterkant op het bed.

    Het plafond van de gleuf is een brug, en de gaten beginnen in die brug. Daarom eerst een laag met een strook
    over de hele breedte open, dan een laag met een vierkant, dan pas het ronde gat: zo hangt er nergens een rand
    van een gat in de lucht."""
    y0, y1 = ARM_Y[0], ARM_Y[0] + P.KOP_D
    m = vlak_xz(kop2d(), y0, y1)
    b, r = P.ARM_B / 2 + 0.15, P.MES_GAT / 2
    plafond = ARM_Y[1] + 0.1
    m = m - box3(-b, y0 - 1, ZH - P.STEEL_ONDER - 0.3, b, plafond, ZH + P.KOP_OOG[1] + 1)
    for dz in P.KOP_PEN:
        z = ZH + dz
        m = m - box3(-b, plafond - E, z - r, b, plafond + 0.25, z + r) - box3(-r, plafond - E, z - r, r, plafond + 0.5, z + r)
    gaten = unary_union([Point(0, ZH + dz).buffer(r, quad_segs=12) for dz in P.KOP_PEN])
    m = m - vlak_xz(gaten, plafond + 0.5 - E, ARM_Y[1] + 3.6)
    return m - vlak_xz(Point(0, ZH).buffer(7.0, quad_segs=24), y1 - 1.0, y1 + 1)


# --- mijnberg --------------------------------------------------------------------------------------

def _stenen(x0, x1, z0, z1, seed):
    """Voegen van een stenen muur in een vlak van x0..x1 bij z0..z1: [(lijn, staand)], al binnen de rand."""
    rnd = random.Random(seed)
    lijnen, z = [], z1
    while z > z0 + 4:
        h = rnd.uniform(8, 12)
        lijnen.append((LineString([(x0, z), (x1, z)]), False))
        x = x0 + rnd.uniform(0, 14)
        while x < x1:
            lijnen.append((LineString([(x, z), (x + rnd.uniform(-1, 1), max(z - h, z0))]), True))
            x += rnd.uniform(14, 26)
        z -= h
    binnen = box(x0 + 1.3, z0 + 1.3, x1 - 1.3, z1 - 1.3)
    return [(g, staand) for lijn, staand in lijnen if (g := lijn.intersection(binnen)).geom_type == "LineString"
            and g.length > 0.1]


def _voegen(stenen, plaat, r=0.5):
    """Voegen met flanken onder 45° (aan het oppervlak 2r breed): de mijnberg print ondersteboven, en een rechte voeg
    zou daar een plat plafondje krijgen. Ook de uiteinden lopen rond onder 45° af. plaat(g, a, b) maakt van vorm g
    een plaat van a tot b diep in de wand (negatief: erbuiten). Elke voeg is de hull van een brede strook aan het
    oppervlak en een smalle onderin. De staande voegen zijn minder diep: zo vallen nergens twee vlakken samen (dat
    geeft flinters die in een STL kapotgaan)."""
    stukken = []
    for lijn, staand in stenen:
        bodem = 0.2 if staand else 0.1                         # halve breedte onderin; diepte r - bodem
        stukken.append(Manifold.batch_hull([plaat(lijn.buffer(r, quad_segs=6), -1.0, E),
                                            plaat(lijn.buffer(bodem, quad_segs=2), r - bodem - E, r - bodem)]))
    return Manifold.batch_boolean(stukken, OpType.Add)


def trechter2d(s):
    """Muntsleuf door het dek aan kant s, in het x-z-vlak: bovenaan van de middenwig tot de binnenkant van het
    bord, onderaan smaller dan de binnenkant van het karretje."""
    kar_binnen = P.TUNNEL_X[1] - P.KAR_SPEL - P.KAR_WAND - 1.0
    g = Polygon([(P.WIG[0], E), (P.TRECHTER_BOVEN, E), (kar_binnen, -P.DEK - E), (P.WIG[1], -P.DEK - E)])
    return affinity.scale(g, s, 1, origin=(0, 0))


@lru_cache(maxsize=None)
def mijnberg():
    xb, (y0, y1), hb, dek, w = P.BERG_X, P.BERG_Y, P.BERG_H, P.DEK, P.BERG_WAND
    m = box3(-xb, y0, -hb, xb, y1, 0)
    gaten = []
    for s in (-1, 1):
        a, b = sorted((s * P.TUNNEL_X[0], s * P.TUNNEL_X[1]))
        gaten.append(box3(a, y0 + w, -hb - 1, b, y1 + 1, -dek))                         # tunnel
        gaten.append(vlak_xz(trechter2d(s), *P.SLEUF_Y))                                  # trechter
        xk0, yk0, xk1, yk1 = _kokervoet_maat()
        a, b = sorted((s * xk0, s * (xk1 + P.KOKER_SPEL)))
        gaten.append(box3(a, yk0 - P.KOKER_SPEL, -P.KOKER_D, b, yk1 + P.KOKER_SPEL, 1))     # uitsparing kokervoet
    v = P.VIN_SLEUF[0]
    gaten.append(box3(-v, P.ACHTER_T + 0.5, -P.VIN_SLEUF[1], v, P.Y_VOOR - 0.5, 1))        # sleuf verdelervin
    m = m - Manifold.batch_boolean(gaten, OpType.Add)
    # voorkant: houten stijlen naast de tunnels en een latei erboven, stenen muur op de rest
    stijlen = [box(-P.TUNNEL_X[0], -hb, P.TUNNEL_X[0], -dek), box(P.TUNNEL_X[1], -hb, P.TUNNEL_X[1] + 6, -dek),
               box(-P.TUNNEL_X[1] - 6, -hb, -P.TUNNEL_X[1], -dek)]
    latei = box(-(P.TUNNEL_X[1] + 6), -dek, P.TUNNEL_X[1] + 6, 0.0)          # tot het dek: ondersteboven geen richel
    m = m + _relief(unary_union(stijlen + [latei]), y1, 1.5)
    voeg = [_voegen(_stenen(*sorted((s * (P.TUNNEL_X[1] + 6), s * xb)), -hb, 0, 3 + s),
                    lambda g, a, b: vlak_xz(g, y1 - b, y1 - a)) for s in (-1, 1)]
    return m - Manifold.batch_boolean(voeg, OpType.Add) - _zijvoegen()


def _zijvoegen():
    """Stenen muur op de zijkanten en de achterkant."""
    xb, (y0, y1), hb = P.BERG_X, P.BERG_Y, P.BERG_H
    delen = [_voegen(_stenen(y0, y1, -hb, 0, 11), lambda g, a, b: vlak_yz(g, xb - b, xb - a)),
             _voegen(_stenen(y0, y1, -hb, 0, 9), lambda g, a, b: vlak_yz(g, -xb + a, -xb + b)),
             _voegen(_stenen(-xb, xb, -hb, 0, 20), lambda g, a, b: vlak_xz(g, y0 + a, y0 + b))]
    return Manifold.batch_boolean(delen, OpType.Add)


def _kokervoet_maat():
    """Buitenmaat (x0, y0, x1, y1) van een kokervoet aan de +x-kant; x0 is ook de binnenkant van de koker."""
    sp, w = P.SPEL, P.KOKER_WAND
    return P.BINNEN_HALF - sp, -sp - w, P.BINNEN_HALF + P.RAND + sp + w, P.BORD_T + P.RELIEF_T + sp + w


def _achthoek(x0, y0, x1, y1, c):
    return Polygon([(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c), (x1 - c, y1), (x0 + c, y1), (x0, y1 - c),
                    (x0, y0 + c)])


@lru_cache(maxsize=None)
def kokervoet(s):
    """Kokervoet aan kant s: een prop die in de uitsparing in het dek klemt, met daarboven een U-koker (open naar
    binnen) waarin de zijrand van het bord en de staander van het mijnhout staan, zoals in de voet van de
    pasproef. Rechtop geprint, de prop op het bed. De staande hoeken zijn afgeschuind (een geprint gat heeft ronde
    hoeken, een vierkante prop zou daarin vastlopen) en onderaan loopt de prop schuin in."""
    W, R, sp, k = P.BINNEN_HALF, P.RAND, P.SPEL, P.KOKER_SPEL
    bx0, by0, bx1, by1 = _kokervoet_maat()
    c, a = P.KOKER_AFSCHUINING, P.KOKER_INLOOP
    z0 = -P.KOKER_D + 0.5
    onder = extrude(_achthoek(bx0 + k + a, by0 + a, bx1 - a, by1 - a, c), z0, z0 + E)
    boven = extrude(_achthoek(bx0 + k, by0, bx1, by1, c), z0 + a, P.KOKER_H)
    m = Manifold.batch_hull([onder, boven]) - box3(bx0 - 1, -sp, 0.0, W + R + sp, P.BORD_T + P.RELIEF_T + sp, P.KOKER_H + 1)
    return m if s > 0 else m.mirror([1, 0, 0])


def _kap_x():
    """x0, x1 van een kokerkap (en een kokervoet) aan de +x-kant."""
    bx0, _, bx1, _ = _kokervoet_maat()
    return bx0 + P.KOKER_SPEL, bx1


@lru_cache(maxsize=None)
def kokerkap(s):
    """Kokerkap aan kant s: een blok met onderaan een gleuf over de bovenhoek van het bord en het mijnhout, en
    bovenaan een gleuf voor de tong van het portaal, allebei open naar binnen. Op de eindwand aan de buitenkant na is
    hij één doorsnede in het yz-vlak: liggend op die eindwand geprint heeft hij geen overhang."""
    sp, w = P.SPEL, P.KOKER_WAND
    x0, x1 = _kap_x()
    xe = P.BINNEN_HALF + P.RAND + sp                       # eindwand tegen de zijkant van het bord
    y0, y1 = P.PORTAAL_Y[0] - sp - w, P.BORD_T + P.RELIEF_T + sp + w
    z0, zl, z1 = P.BORD_H - P.KAP_ONDER, P.BORD_H + P.KAP_LAT, P.PORTAAL_Z[0]
    m = box3(x0, y0, z0, x1, y1, z1)
    m = m - box3(x0 - 1, -sp, z0 - 1, xe, P.BORD_T + P.RELIEF_T + sp, P.BORD_H)          # bord en mijnhout
    m = m - box3(x0 - 1, P.PORTAAL_Y[0] - sp, zl, xe, P.PORTAAL_Y[1] + sp, z1 + 1)      # tong van het portaal
    return m if s > 0 else m.mirror([1, 0, 0])


# --- karretjes, verdelervin ------------------------------------------------------------------------

def kar_maat(s):
    x0, x1 = sorted((s * (P.TUNNEL_X[0] + P.KAR_SPEL), s * (P.TUNNEL_X[1] - P.KAR_SPEL)))
    y0, y1 = P.BERG_Y[0] + P.BERG_WAND + P.KAR_SPEL, P.BERG_Y[1]
    z0, z1 = -P.BERG_H, -P.DEK - P.KAR_RAND_ONDER_DEK
    return x0, y0, z0, x1, y1, z1


@lru_cache(maxsize=None)
def karretje(s):
    """Mijnkarretje (laatje) aan kant s (+1 = links voor het kind): een bak die onderaan smaller is, vooraan
    wielen, een band en klinknagels, en bovenin een uitsparing voor de duim. Geprint rechtop, open boven: de
    schuine zijwanden hangen KAR_SCHUIN graden over."""
    x0, y0, z0, x1, y1, z1 = kar_maat(s)
    w, f = P.KAR_WAND, P.KAR_FRONT
    t = math.tan(math.radians(P.KAR_SCHUIN))
    c = math.cos(math.radians(P.KAR_SCHUIN))
    inset = (z1 - z0) * t

    def rand(z):                       # x van de linker buitenwand op hoogte z
        return x0 + inset * (z1 - z) / (z1 - z0)
    buiten = Polygon([(x0 + inset, z0), (x1 - inset, z0), (x1, z1), (x0, z1)])
    binnen = Polygon([(rand(z0 + w) + w / c, z0 + w), (x1 + x0 - rand(z0 + w) - w / c, z0 + w),
                      (x1 - w / c + t, z1 + 1), (x0 + w / c - t, z1 + 1)])
    m = vlak_xz(buiten, y0, y1) - vlak_xz(binnen, y0 + w, y1 - f)
    xc = (x0 + x1) / 2
    m = m - vlak_xz(Point(xc, z1).buffer(P.KAR_GREEP, quad_segs=24), y1 - f - 1, y1 + 3)
    # versiering vooraan: twee wielen met een naaf, klinknagels langs de bovenrand; randen onder 45° (rechtop geprint)
    deco = []
    zw, rw = z0 + 8.0, 7.5
    for xw in (rand(zw) + rw + 1.5, x1 + x0 - rand(zw) - rw - 1.5):
        deco.append(_relief(Point(xw, zw).buffer(rw, quad_segs=24), y1, 1.4, schuin=1.4))
        deco.append(_relief(Point(xw, zw).buffer(2.6, quad_segs=16), y1 + 1.4 - E, 0.8, schuin=0.8))
    for i in range(12):
        xr = x0 + 7 + i * (x1 - x0 - 14) / 11
        if abs(xr - xc) > P.KAR_GREEP + 3:
            deco.append(_relief(Point(xr, z1 - 4.0).buffer(1.6, quad_segs=10), y1, 0.8, schuin=0.8))
    return m + Manifold.batch_boolean(deco, OpType.Add)


def verdelervin():
    """Vin die uit het dek omhoog in de spleet steekt, recht onder de middelste pin; een tong klemt in de wig.
    Geprint liggend."""
    h, t = P.VERDELER_T / 2, P.VERDELER_TOP
    v = P.VIN_TONG
    vorm = Polygon([(-v, -P.VIN_SLEUF[1] + 0.3), (v, -P.VIN_SLEUF[1] + 0.3), (v, 0), (h, 0), (h, t - h),
                    (0.4, t - 0.4), (-0.4, t - 0.4), (-h, t - h), (-h, 0), (-v, 0)])
    return vlak_xz(vorm, P.ACHTER_T + 0.6, P.Y_VOOR - 0.6)


# --- tienveld en munten ----------------------------------------------------------------------------

def tienveld():
    """Plankje met twee rijen van vijf kuiltjes. Geprint rechtop; ligt in de mijn vóór het bord."""
    st, r = P.TIEN_STAP, P.TIEN_RAND
    b, d = 5 * st + 2 * r, 2 * st + 2 * r
    bodem, diep = P.TIEN_T
    m = extrude(rounded_rect(-b / 2, -d / 2, b / 2, d / 2, 6.0), 0, bodem + diep)
    gaten = unary_union([Point((i - 2) * st, (j - 0.5) * st).buffer(P.TIEN_GAT / 2, quad_segs=32)
                         for i in range(5) for j in range(2)])
    return m - extrude(gaten, bodem, bodem + diep + 1)


@lru_cache(maxsize=None)
def munt_delen():
    """(geel onderste helft, blauw bovenste helft met de diamant) van een muntje, plat zoals geprint."""
    m = V.munt()
    snee = box3(-20, -20, -1, 20, 20, P.MUNT_T / 2)
    return m ^ snee, m - snee


# --- montage -----------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def delen():
    """{naam: (kleur, Manifold, beweegt)} in de ruststand."""
    out = {"achterplaat": (P.ZWART, V.achterplaat(), False), "voorplaat": (P.ZWART, V.voorplaat(dy=DY), False),
           "mijnhout": (P.BRUIN, mijnhout(), False)}
    for naam, (kleur, m) in portaal_delen().items():
        out[naam] = (kleur, m, False)
    out.update({
        "mijnberg": (P.TERRA, mijnberg(), False),
        "kokervoet L": (P.BRUIN, kokervoet(1), False),
        "kokervoet R": (P.BRUIN, kokervoet(-1), False),
        "kokerkap L": (P.BRUIN, kokerkap(1), False),
        "kokerkap R": (P.BRUIN, kokerkap(-1), False),
        "verdelervin": (P.GEEL, verdelervin(), False),
        "beugel": (P.BRUIN, V.beugel(DY), False),
        "plank": (P.GEEL, V.plank(DY), True),
        "steel": (P.BRUIN, steel(), True),
        "kop": (P.GEEL, kop(), True),
        "mesblok": (P.BRUIN, V.mesblok(DY), True),
        "karretje L": (P.ROOD, karretje(1), False),
        "karretje R": (P.ROOD, karretje(-1), False),
    })
    return out


def stand(hoek=0.0, uit=(0.0, 0.0)):
    """Alle onderdelen bij slingerhoek `hoek`; uit: hoe ver het linker en rechter karretje uitgetrokken zijn."""
    out = {}
    for naam, (kleur, m, beweegt) in delen().items():
        if naam == "karretje L":
            m = m.translate([0, uit[0], 0])
        if naam == "karretje R":
            m = m.translate([0, uit[1], 0])
        out[naam] = (kleur, V.draai_y(m, hoek) if beweegt else m)
    return out


# --- slinger: massa uit de geometrie ----------------------------------------------------------------

def massa_eigenschappen(m, dichtheid=1.24e-3, as_=(0.0, P.Z_PLANK)):
    """(massa g, zwaartepunt (x, z), traagheid om de as evenwijdig aan y door as_ in g mm²), exact uit de mesh
    (tetraëders vanuit de as)."""
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties[:, :3], dtype=float) - np.array([as_[0], 0.0, as_[1]])
    f = np.asarray(mesh.tri_verts)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    zw = (a + b + c) / 4.0

    def tweede(k):
        return vol / 10.0 * (a[:, k] ** 2 + b[:, k] ** 2 + c[:, k] ** 2 + a[:, k] * b[:, k] + a[:, k] * c[:, k]
                             + b[:, k] * c[:, k])
    V_ = vol.sum()
    massa = V_ * dichtheid
    cx, cz = (zw[:, 0] * vol).sum() / V_, (zw[:, 2] * vol).sum() / V_
    I = (tweede(0).sum() + tweede(2).sum()) * dichtheid
    return massa, (cx + as_[0], cz + as_[1]), I


def slingermassa():
    """(m, d, I) van alles wat draait (steel, kop, mesblok, plank), allemaal massief geprint: zoals simulatie.py
    ze verwacht (d = afstand van het zwaartepunt onder de as)."""
    eig = [massa_eigenschappen(m) for _, m, beweegt in delen().values() if beweegt]
    massa = sum(e[0] for e in eig)
    cz = sum(e[0] * e[1][1] for e in eig) / massa
    return massa, P.Z_PLANK - cz, sum(e[2] for e in eig)
