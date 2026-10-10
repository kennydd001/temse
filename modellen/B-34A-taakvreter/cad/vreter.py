"""Geometrie van B-34A Taakvreter: alle onderdelen, de stand bij elke krukhoek, en de printligging.

Onderdelen (kleur):
  romp voor / romp achter   Orange   tweedelige romp, gedeeld in het vlak van de rolassen
  kruk                      Orange   krukarm met de aandrijfas (zeskant in de onderste rol) in één stuk
  paspen                    Orange   6 tussen de rompdelen, 2 onder de hoorntjes
  lade                      Ivory    uitneembare buikbak
  ogen                      Ivory    oogas met twee oogtrommels
  tanden, knop, hoorn       Ivory
  rol onder / rol boven     Charcoal invoerrollen met tandwiel (1:1) en geprinte asstompen; onder 3 O-ringen
  nokas                     Charcoal krukschijf voor de ogen, lager in de beugel, zeskant in de bovenste rol
  drijfstang, tuimelaar     Charcoal vierstangenmechanisme naar de oogas
  mondstuk                  Charcoal lip, mond en invoertunnel (4 mm hoog)
  pupil (2x)                Charcoal

Elk bewegend onderdeel wordt gebouwd rond zijn eigen as (door y = z = 0); `stand(phi)` zet alles op zijn
plek bij krukhoek phi (graden; positief = invoeren: de bovenkant van de kruk gaat van je af).
"""
import math
from functools import lru_cache

import numpy as np
from manifold3d import CrossSection, Manifold
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import _pad  # noqa: F401
import params as P
from gereedschap.geom import druppel, extrude, rounded_rect, text, vlak_xz, vlak_yz
from gereedschap.tandwiel import tandwiel

E = P.E
A_ONDER = (P.Y_DEEL, P.Z_ONDER)
A_BOVEN = (P.Y_DEEL, P.Z_BOVEN)
B_OOG = (P.OOG_Y, P.OOG_Z)


# --- hulpjes ------------------------------------------------------------------------------------

def cil_x(cy, cz, r, x0, x1, n=48):
    return vlak_yz(Point(cy, cz).buffer(r, quad_segs=n // 4), x0, x1)


def cil_y(cx, cz, r, y0, y1, n=48):
    return vlak_xz(Point(cx, cz).buffer(r, quad_segs=n // 4), y0, y1)


def cil_z(cx, cy, r, z0, z1, n=48):
    return extrude(Point(cx, cy).buffer(r, quad_segs=n // 4), z0, z1)


def zeskant(c, sw, draai=0.0):
    r = sw / math.sqrt(3)
    return Polygon([(c[0] + r * math.cos(math.radians(draai + 30 + 60 * k)),
                     c[1] + r * math.sin(math.radians(draai + 30 + 60 * k))) for k in range(6)])


def draai_x(m, graden, om=(0.0, 0.0)):
    """Draai om een as evenwijdig aan x door (y, z) = om; positief = van +y naar +z."""
    a = math.radians(graden)
    c, s = math.cos(a), math.sin(a)
    oy, oz = om
    T = np.array([[1, 0, 0, 0], [0, c, -s, oy - (c * oy - s * oz)], [0, s, c, oz - (s * oy + c * oz)]], dtype=float)
    return m.transform(T)


def rev_x(prof, n=64):
    """Omwentelingslichaam om de x-as; prof = [(straal, x), ...] als gesloten omtrek."""
    pts = list(orient(Polygon(prof), 1.0).exterior.coords)[:-1]
    m = Manifold.revolve(CrossSection([pts]), n)                                     # om z, straal in x
    return m.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))   # z -> x


def rev_z(prof, n=64):
    pts = list(orient(Polygon(prof), 1.0).exterior.coords)[:-1]
    return Manifold.revolve(CrossSection([pts]), n)


def op(m, as_):
    """Een onderdeel dat rond zijn eigen as is gebouwd (door y = z = 0) naar de as (y, z) schuiven."""
    return m.translate([0, as_[0], as_[1]])


# --- 2D-vormen ----------------------------------------------------------------------------------

def profiel(krimp=0.0):
    """Vooraanzicht van de romp (x, z), krimp mm naar binnen."""
    onder = rounded_rect(-P.B2, 0, P.B2, P.H, P.R_VOET).intersection(box(-P.B2, 0, P.B2, P.H / 2))
    boven = rounded_rect(-P.B2, 0, P.B2, P.H, P.R_TOP).intersection(box(-P.B2, P.H / 2 - 1, P.B2, P.H))
    g = unary_union([onder, boven])
    return g.buffer(-krimp, join_style="round") if krimp else g


def mond2d(groei=0.0):
    """Grijns: superellips met een vlakke bovenkant op MOND_TOP; groei > 0 maakt hem groter."""
    a, b = P.MOND
    n = P.MOND_EXP
    us = np.linspace(-1, 1, 161)
    g = Polygon([(a * u, P.MOND_TOP - b * (1 - abs(u) ** n) ** (1 / n)) for u in us])
    if groei > 0:
        return g.buffer(groei, quad_segs=12)
    return g.buffer(groei, join_style="mitre") if groei else g


def lade2d(groei=0.0):
    """Doorsnede (x, z) van de lade: 45° onderaan de lange kanten."""
    L, f = P.LADE_HALF, P.LADE_FAS
    z0, z1 = P.LADE_Z
    g = Polygon([(-L + f, z0), (L - f, z0), (L, z0 + f), (L, z1), (-L, z1), (-L, z0 + f)])
    return g.buffer(groei, join_style="mitre") if groei else g


def u_lager(c, r, opening, lengte=12.0):
    """Klik-lager in het y-z-vlak: rond gat, open naar +y (de kant van het deelvlak)."""
    return unary_union([Point(c).buffer(r, quad_segs=24),
                        box(c[0], c[1] - opening / 2, c[0] + lengte, c[1] + opening / 2)])


def tanden2d():
    """Bovengebit: band met negen tanden, hangt binnen in de bovenrand van de mond."""
    a, t = 40.0, P.MOND_TOP
    band = box(-a, t - 3.4, a, t - 0.6)
    n = 9
    w = 2 * a / n
    tand = [Polygon([(-a + k * w + 0.5, t - 3.3), (-a + (k + 1) * w - 0.5, t - 3.3), (-a + (k + 0.5) * w, t - 6.0)])
            for k in range(n)]
    return unary_union([band] + tand).intersection(mond2d(-0.8))


TAND_PENNEN = [(-24.0, P.MOND_TOP - 2.0), (0.0, P.MOND_TOP - 2.0), (24.0, P.MOND_TOP - 2.0)]


# --- romp ---------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def buiten():
    """Buitenvorm van de romp, met 45° rond voor- en achterkant."""
    s = P.SCHUIN
    return Manifold.batch_hull([vlak_xz(profiel(s), 0, E), vlak_xz(profiel(), s, P.D - s),
                                vlak_xz(profiel(s), P.D - E, P.D)])


@lru_cache(maxsize=None)
def romp(met_goot=True):
    """De hele romp met alles wat over het deelvlak heen loopt (nog niet gedeeld)."""
    s, w, xi = P.SCHUIN, P.WAND, P.X_BINNEN
    m = buiten() - vlak_xz(profiel(w), w, P.D - w)
    binnen = vlak_xz(profiel(w - 0.5), 0, P.D)               # alles wat binnen zit, raakt de wand net
    y0, y1 = w - E, P.D - w + E
    inbouw = []
    # lagerribben in de zijwanden, van voor tot achter (printbaar in beide helften)
    for z in (P.Z_ONDER, P.Z_BOVEN):
        inbouw.append(vlak_yz(box(y0, z - 7.0, y1, z + 7.0), P.RIB[0], xi + 0.5))
    inbouw.append(vlak_yz(box(y0, P.Z_ONDER - 7.0, y1, P.Z_ONDER + 7.0), -xi - 0.5, P.RIB_LINKS[1]))
    # lagerbeugel links: draagt de nokas en scheidt het oogmechaniek van het papier; vooraan onder 45°
    # weggesneden, zodat hij in de print niet boven het mondgat begint
    beugel = Polygon([(y0, P.MOND_TOP + 1.0), (y0, P.H), (y1, P.H), (y1, 57.0), (y0 + P.MOND_TOP + 1.0 - 57.0, 57.0)])
    inbouw.append(vlak_yz(beugel, *P.BEUGEL_X))
    # schap onder de rollen (voorhelft; het stuk voorbij het deelvlak komt in romp_voor)
    inbouw.append(vlak_xz(box(-xi - 0.5, P.SCHAP[0], xi + 0.5, P.SCHAP[1]), y0, P.Y_DEEL))
    # kolommen voor de paspennen
    for x, z in P.PASPENNEN:
        inbouw.append(cil_y(x, z, P.KOLOM_R, y0, y1))
    # oogaslagers: links naast de beugel en tussen de trommels, aan het gezicht
    for x0, x1 in (P.OOGKLEM_LINKS, P.OOGKLEM_X):
        inbouw.append(vlak_yz(box(y0, P.OOG_Z - 5.5, P.OOG_Y + 4.5, P.OOG_Z + 5.5), x0, x1))
    # naven voor de hoorntjes, onder het dak (aan het gezicht vast)
    for x in (-P.HOORN_X, P.HOORN_X):
        inbouw.append(extrude(box(x - 5.5, y0, x + 5.5, P.HOORN_Y + 5.5), P.H - P.WAND - 8.0, P.H - 1.0))
    m = m + (Manifold.batch_boolean(inbouw, manifold_add()) ^ binnen)
    if met_goot:
        m = m + goot()
    # gaten: halve gaten in het deelvlak voor stompen en assen
    r = P.GAT_DRAAI / 2
    m = m - cil_x(*A_ONDER, r, -P.GAT_EIND, P.RIB_LINKS[1] + 1)            # stomp onderste rol links
    m = m - cil_x(*A_ONDER, r, P.RIB[0] - 1, P.B2 + 1)                      # kruk-as, door de wand
    m = m - cil_x(*A_BOVEN, r, P.RIB[0] - 1, P.GAT_EIND)                    # stomp bovenste rol rechts
    m = m - cil_x(*A_BOVEN, r, P.BEUGEL_X[0] - 1, P.BEUGEL_X[1] + 1)        # nokas in de beugel
    # oogas: klik-lagers en een vrije sleuf door de beugel
    u = u_lager(B_OOG, P.OOGAS_R + 0.15, P.U_OPENING)
    for x0, x1 in (P.OOGKLEM_LINKS, P.OOGKLEM_X):
        m = m - vlak_yz(u, x0 - 1, x1 + 1)
    sleuf = unary_union([Point(B_OOG).buffer(P.OOGAS_R + 0.4, quad_segs=24),
                         box(P.OOG_Y, P.OOG_Z - P.OOGAS_R - 0.4, P.Y_DEEL + 1, P.OOG_Z + P.OOGAS_R + 0.4)])
    m = m - vlak_yz(sleuf, P.BEUGEL_X[0] - 1, P.BEUGEL_X[1] + 1)
    # openingen in het gezicht: ogen, mond, ladebaai (die loopt door tot de achterwand)
    gezicht = unary_union([Point(-P.OOG_X, P.OOG_Z).buffer(P.VENSTER_R, quad_segs=32),
                           Point(P.OOG_X, P.OOG_Z).buffer(P.VENSTER_R, quad_segs=32), mond2d()])
    m = m - vlak_xz(gezicht, -1, w + E)
    m = m - vlak_xz(lade2d(P.BAAI_SPEL), -1, P.D - w)
    # hoorngaten: liggend in de print, dus druppelvormig (punt naar +y)
    for x in (-P.HOORN_X, P.HOORN_X):
        m = m - extrude(druppel((x, P.HOORN_Y), P.HOORN_GAT / 2, omhoog=True), P.H - 8.5, P.H + 1)
    return m


def manifold_add():
    from manifold3d import OpType
    return OpType.Add


def goot():
    """Glijgoot achter de rollen: een plafond dat het briefje omlaag de lade in buigt (massief erachter)."""
    (gy, gz), (ry, rz) = P.GOOT_MIDDEN, P.GOOT_R
    ell = affinity.translate(affinity.scale(Point(0, 0).buffer(1.0, quad_segs=64), ry, rz), gy, gz)
    g = box(gy, gz, P.D - P.WAND + E, P.GOOT_TOP).difference(ell)
    return vlak_yz(g, P.BEUGEL_X[1] - E, P.X_BINNEN + 0.5)


def paspengaten(y0, y1, d):
    return Manifold.batch_boolean([cil_y(x, z, d / 2, y0, y1, n=32) for x, z in P.PASPENNEN], manifold_add())


def helften(lijf, paspennen, label, achterkant=P.D):
    """Deel een romp(stuk) in het deelvlak; paspengaten: klemmend in de voorhelft, schuivend in de achterhelft."""
    voor = lijf ^ Manifold.cube([300, P.Y_DEEL + 10, 300]).translate([-150, -10, -10])
    # schap loopt 10 mm door voorbij het deelvlak: schermt de onderkant van de rollen af
    h = P.SCHAP_HALF_ACHTER
    voor = voor + vlak_xz(box(-h, P.SCHAP[0], h, P.SCHAP[1]), P.Y_DEEL - E, P.SCHAP[2])
    voor = voor - paspengaten(paspennen, P.Y_DEEL - P.PASPEN_L / 2 - 0.5, P.Y_DEEL + 1, P.GAT_PASPEN_VAST)
    achter = lijf ^ Manifold.cube([300, P.D - P.Y_DEEL + 10, 300]).translate([-150, P.Y_DEEL, -10])
    achter = achter - paspengaten(paspennen, P.Y_DEEL - 1, P.Y_DEEL + P.PASPEN_L / 2 + 0.5, P.GAT_PASPEN_LOS)
    # versielabel in de achterkant, gespiegeld zodat het van achteren leesbaar is
    lab = affinity.scale(text(label, 4.5, 0, 14 if lijf.bounding_box()[2] < 1 else P.SCHAP[0] + 8), -1, 1, origin=(0, 0))
    return voor, achter - vlak_xz(lab, achterkant - 0.4, achterkant + 1)


def paspengaten(paspennen, y0, y1, d):
    return Manifold.batch_boolean([cil_y(x, z, d / 2, y0, y1, n=32) for x, z in paspennen], manifold_add())


@lru_cache(maxsize=None)
def _helften():
    return helften(romp(), P.PASPENNEN, f"B-34A Taakvreter {P.VERSIE}")


def romp_voor():
    return _helften()[0]


def romp_achter():
    return _helften()[1]


# --- pasproef: afgeschermd invoerrol-proefstuk ----------------------------------------------------

PROEF_TOP = 86.0                       # onder de oogaslagers en de oogvensters
PROEF_ACHTER = 50.0                    # ondiepe achterhelft: de glijgoot hoort niet bij de proef
PROEF_PASPENNEN = [(-57.5, 38.6), (57.5, 38.6), (-54.0, 80.0), (54.0, 80.0)]


@lru_cache(maxsize=None)
def proefframe():
    """De romp tussen het schap en PROEF_TOP en tot PROEF_ACHTER, met een dicht deksel en een achterwand:
    mond, rollen, kruk, beugel en ribben zoals in het echte model, zonder ogen, lade en glijgoot. Het papier
    komt er achter de rollen door een sleuf in de achterwand weer uit."""
    z0, z1, ya = P.SCHAP[0], PROEF_TOP, PROEF_ACHTER
    blok = Manifold.cube([300, ya + 10, z1 - z0]).translate([-150, -10, z0])
    lijf = romp(False) ^ blok
    lijf = lijf + (buiten() ^ Manifold.cube([300, ya + 10, P.WAND]).translate([-150, -10, z1 - E]))
    achterwand = buiten() ^ Manifold.cube([300, P.WAND, z1 + P.WAND - z0]).translate([-150, ya - P.WAND, z0])
    uit = vlak_xz(box(-P.TUNNEL_HALF, P.NIP_Z - 4.0, P.TUNNEL_HALF, P.NIP_Z + 4.0), ya - P.WAND - 1, ya + 1)
    lijf = lijf + (achterwand - uit)
    ramen = unary_union([Point(x, P.OOG_Z).buffer(P.VENSTER_R + 0.5) for x in (-P.OOG_X, P.OOG_X)])
    lijf = lijf + (buiten() ^ vlak_xz(ramen.intersection(box(-100, z0, 100, z1 + 1)), 0, P.WAND + E))
    binnen = vlak_xz(profiel(P.WAND - 0.5), 0, P.D)
    for x, z in PROEF_PASPENNEN[2:]:
        lijf = lijf + (cil_y(x, z, P.KOLOM_R, P.WAND - E, ya - P.WAND + E) ^ binnen)
    return helften(lijf, PROEF_PASPENNEN, f"B-34A proef {P.VERSIE}", achterkant=ya)


def paspenstrip():
    """Strip met paspengaten in zes maten (staand, zoals in de rompdelen), maten bovenop."""
    m = extrude(rounded_rect(0, 0, 62, 16, 3), 0, 10)
    for i, d in enumerate((4.85, 4.9, 4.95, 5.0, 5.05, 5.1)):
        x = 6 + i * 10
        m = m - cil_z(x, 10.5, d / 2, 1.5, 11, n=32)
        m = m - extrude(text(f"{d:.2f}".replace(".", ",")[1:], 2.6, x, 3.6), 9.5, 10.01)
    return m


# --- rollen -------------------------------------------------------------------------------------

def _fase(tand_op):
    """Draaiing (graden) van het tandwiel zodat er een tand (of een tussenruimte) op hoek tand_op staat."""
    st = 360.0 / P.Z_TAND
    return tand_op - st * round(tand_op / st)


FASE_ONDER = _fase(90.0)                       # tand naar boven
FASE_BOVEN = _fase(270.0 - 180.0 / P.Z_TAND)   # tussenruimte naar beneden


def wiel(fase):
    w = tandwiel(P.M_ROL, P.Z_TAND, speling=0.08)
    return vlak_yz(affinity.rotate(w, fase, origin=(0, 0)), *P.TAND_X)


def koker(x_open, x_bodem):
    """Zeskantkoker vanaf het roluiteinde x_open tot x_bodem, met een 45°-punt (printbaar plafond)."""
    sw = P.ZESKANT_GAT
    lo, hi = sorted((x_open, x_bodem))
    pr = vlak_yz(zeskant((0, 0), sw), lo - (1 if x_open < x_bodem else 0), hi + (1 if x_open > x_bodem else 0))
    rk = sw / math.sqrt(3)
    s = 1 if x_bodem > x_open else -1
    punt = rev_x([(0.0, x_bodem - s * E), (rk, x_bodem - s * E), (0.0, x_bodem + s * rk)])
    return pr + punt


def rol_onder():
    """Aangedreven rol: stomp links, O-ringgroeven, tandwiel rechts met de zeskantkoker voor de kruk.

    Geprint staand op het tandwiel; de hals onder de rol (45°) houdt de tanden vrij van overhang.
    """
    r, (t0, t1) = P.ROL_R, P.TAND_X
    rf = P.TAND_VOET
    xs = P.RIB_LINKS[1] + 0.3                  # schouder stopt 0,3 voor de linkerrib
    prof = [(0.0, -P.STOMP_EIND), (P.STOMP_R - 0.5, -P.STOMP_EIND), (P.STOMP_R, -P.STOMP_EIND + 0.5),
            (P.STOMP_R, xs), (P.SCHOUDER_R, xs), (P.SCHOUDER_R, -P.ROL_X), (r, -P.ROL_X)]
    d, b = P.GROEF_DIEP, P.GROEF_B
    for gx in P.GROEF:
        prof += [(r, gx - b / 2 - d), (r - d, gx - b / 2), (r - d, gx + b / 2), (r, gx + b / 2 + d)]
    prof += [(r, t0 - (r - rf)), (rf, t0 + E), (0.0, t0 + E)]
    m = rev_x(prof) + wiel(FASE_ONDER)
    return m - koker(t1, t1 - P.KOKER_DIEP)


def rol_boven():
    """Meelopende rol: naaf links (tegen de beugel) met de koker voor de nokas, tandwiel en stomp rechts.

    Geprint staand op het linkereind; onder het tandwiel een 45°-kraag tot de kopcirkel.
    """
    r, (t0, t1) = P.ROL_R, P.TAND_X
    x0 = P.BOVEN_X0
    xs = P.RIB[0] - 0.3
    prof = [(0.0, x0), (5.5, x0), (5.5, x0 + 0.2), (r, x0 + 0.2 + (r - 5.5)),
            (r, t0 - (P.TAND_KOP - r)), (P.TAND_KOP, t0), (P.SCHOUDER_R, t0),
            (P.SCHOUDER_R, xs), (P.STOMP_R, xs), (P.STOMP_R, P.STOMP_EIND - 0.5),
            (P.STOMP_R - 0.5, P.STOMP_EIND), (0.0, P.STOMP_EIND)]
    m = rev_x(prof) + wiel(FASE_BOVEN)
    return m - koker(x0, x0 + P.KOKER_DIEP)


def oring(gx):
    """O-ring 14 x 2 in een groef (rekwisiet: niet om te printen)."""
    rm = P.ROL_R - P.GROEF_DIEP + P.ORING_SNOER / 2
    c = Point(rm, gx).buffer(P.ORING_SNOER / 2, quad_segs=8)
    return rev_x(list(c.exterior.coords)[:-1], n=48)


# --- kruk en knop -------------------------------------------------------------------------------

def kruk():
    """Krukarm (buiten de rechterwand) met de aandrijfas in één stuk; arm naar +z, as door (0, 0).

    Geprint met de buitenkant van de arm op het bed: de as staat rechtop, de kraag zit op een 45°-kegel.
    """
    x0, x1 = P.KRUK_X
    g = unary_union([LineString([(0, 0), (0, P.KRUK_R)]).buffer(P.KRUK_B, quad_segs=16),
                     Point(0, 0).buffer(7.5, quad_segs=24)])
    g = g.difference(Point(0, P.KRUK_R).buffer(P.KNOP_GAT, quad_segs=16))
    m = vlak_yz(g, x0, x1)
    k0, k1 = P.KRAAG
    m = m + rev_x([(0.0, k0), (P.SCHOUDER_R, k0), (P.SCHOUDER_R, k1), (P.STOMP_R, k1 + 1.5),
                   (P.STOMP_R, x0 + E), (0.0, x0 + E)])
    t1 = P.TAND_X[1]
    m = m + vlak_yz(zeskant((0, 0), P.ZESKANT), t1 - P.KOKER_DIEP + 0.3, k0 + E)
    return m


def knop():
    """Draaiknop met klikpen: de pen draait in het gat van de krukarm, het haakje vangt achter de arm."""
    x0, x1 = P.KRUK_X
    k0 = x1 + 0.3
    r, rk = P.KNOPPEN_R, P.KNOP_R
    prof = [(0.0, x0 - 1.6), (1.9, x0 - 1.6), (2.55, x0 - 0.65), (2.55, x0 - 0.35), (r, x0 - 0.05),
            (r, k0), (rk, k0), (rk, k0 + P.KNOP_L - 1.0), (rk - 1.0, k0 + P.KNOP_L), (0.0, k0 + P.KNOP_L)]
    m = rev_x(prof)
    m = m - Manifold.cube([x1 - x0 + 1.5, 1.0, 10.0]).translate([x0 - 2.0, -0.5, -5.0])   # gleuf: de pen veert in
    return m.translate([0, 0, P.KRUK_R])


# --- oogmechaniek -------------------------------------------------------------------------------

def nokas():
    """Krukschijf met pengat op straal NOK_E (bij hoek 0 op +y), lager door de beugel, zeskant in de rol."""
    x0, x1 = P.NOKSCHIJF_X
    m = cil_x(0, 0, P.NOKSCHIJF_R, x0, x1)
    m = m - cil_x(P.NOK_E, 0, P.PEN_GAT, x0 - 1, x1 + 1, n=24)
    m = m + cil_x(0, 0, P.STOMP_R, x1 - E, P.BOVEN_X0 - 0.3)
    m = m + vlak_yz(zeskant((0, 0), P.ZESKANT), P.BOVEN_X0 - 0.3 - E, P.BOVEN_X0 + P.KOKER_DIEP - 0.3)
    return m


def ogen():
    """Oogas met twee trommels, as door (0, 0); links een D-eind voor de tuimelaar (vlak naar +y).

    Geprint staand op de rechtertrommel; onder de linkertrommel een 45°-kegel.
    """
    rs, r, h = P.OOGAS_R, P.TROMMEL_R, P.TROMMEL_HALF
    xr0, xr1 = P.OOG_X - h, P.OOG_X + h
    xl0, xl1 = -P.OOG_X - h, -P.OOG_X + h
    xd = P.TUIMEL_X[0] - 0.2
    prof = [(0.0, xr1), (r, xr1), (r, xr0), (rs, xr0), (rs, xl1 + (r - rs)), (r, xl1), (r, xl0), (rs, xl0),
            (rs, xd), (0.0, xd)]
    m = rev_x(prof)
    m = m - vlak_yz(box(rs - 1.0, -5, 5, 5), xd - 1, P.TUIMEL_X[1] + 0.2)
    for x in (-P.OOG_X, P.OOG_X):                    # gaatjes voor de pupilpennen: druppel, punt naar -x (boven in de print)
        gat = affinity.rotate(druppel((x, 0.0), 1.05, omhoog=True, n=8), 90, origin=(x, 0.0))
        m = m - vlak_xz(gat, -r - 1, -r + 2.0)
    return m


def pupil(x):
    """Pupil: schijfje vóór op een trommel (in het ogen-frame), met een pennetje in de trommel."""
    r = P.TROMMEL_R
    m = cil_y(x, 0, P.PUPIL_R, -(r + P.PUPIL_T), -r)
    return m + cil_y(x, 0, 1.0, -r - E, -r + 1.7, n=16)


@lru_cache(maxsize=None)
def vierstang_bereik():
    hoeken = [vierstang(f)[2] for f in np.linspace(0, 360, 145)]
    return min(hoeken), max(hoeken)


def hoek_midden():
    lo, hi = vierstang_bereik()
    return (lo + hi) / 2


def tuimelaar():
    """Tuimelaar op het D-eind van de oogas (as door (0, 0)), arm bij de middenstand, pen naar +x."""
    x0, x1 = P.TUIMEL_X
    arm = LineString([(0, 0), (P.TUIMEL, 0)]).buffer(4.0, quad_segs=12)
    m = vlak_yz(arm, x0, x1)
    m = m + cil_x(P.TUIMEL, 0, P.PEN_R, x1 - E, P.STANG_X[1] - 0.3, n=24)
    m = draai_x(m, hoek_midden())
    d = Point(0, 0).buffer(P.OOGAS_R + 0.15, quad_segs=24).intersection(box(-5, -5, P.OOGAS_R - 1.0 + 0.15, 5))
    return m - vlak_yz(d, x0 - 1, x1 + 1)


def drijfstang():
    """Drijfstang langs +y vanaf (0, 0): gat boven (tuimelaarpen), pen onder naar +x (in de krukschijf)."""
    x0, x1 = P.STANG_X
    g = LineString([(0, 0), (P.STANG, 0)]).buffer(3.6, quad_segs=12).difference(Point(0, 0).buffer(P.PEN_GAT, quad_segs=16))
    m = vlak_yz(g, x0, x1)
    return m + cil_x(P.STANG, 0, P.PEN_R, x1 - E, P.NOKSCHIJF_X[1] - 0.3, n=24)


def vierstang(phi_boven):
    """Oogmechaniek bij draaihoek van de bovenste rol (graden).

    Geeft (krukpen (y, z), tuimelaarpen (y, z), tuimelaarhoek in graden). De tuimelaar staat achter de
    oogas (tak naar +y): daar is de meeste ruimte.
    """
    A, B = np.array(A_BOVEN), np.array(B_OOG)
    a = math.radians(phi_boven)
    Pk = A + P.NOK_E * np.array([math.cos(a), math.sin(a)])
    D = float(np.linalg.norm(B - Pk))
    c, r4 = P.STANG, P.TUIMEL
    x = (r4 ** 2 - c ** 2 + D ** 2) / (2 * D)
    h = math.sqrt(max(r4 ** 2 - x ** 2, 0.0))
    u = (Pk - B) / D
    Q = B + x * u + h * np.array([-u[1], u[0]])
    hoek = math.degrees(math.atan2(Q[1] - B[1], Q[0] - B[0]))
    return tuple(Pk), tuple(Q), hoek


# --- mond, lade, kleine delen -------------------------------------------------------------------

def mondstuk():
    """Lip vóór het gezicht, prop in de mond, tunnel tot vlak voor de rollen; de invoer is 4 mm hoog.

    Geprint met de lip op het bed: alles daarboven is even groot of kleiner.
    """
    y0 = -P.LIP_T
    m = vlak_xz(mond2d(P.LIP), y0, 0.0)
    m = m + vlak_xz(mond2d(-P.MOND_SPEL), -E, P.WAND + E)
    tunnel = mond2d(-P.MOND_SPEL).intersection(box(-P.TUNNEL_HALF, 0, P.TUNNEL_HALF, 200))
    m = m + vlak_xz(tunnel, P.WAND, P.TUNNEL_Y)
    b, h, t = P.SLEUF_B / 2, P.SLEUF_H / 2, P.TRECHTER
    sleuf = box(-b, P.NIP_Z - h, b, P.NIP_Z + h)
    m = m - vlak_xz(sleuf, y0 - 1, P.TUNNEL_Y + 1)
    m = m - Manifold.batch_hull([vlak_xz(sleuf.buffer(t, join_style="mitre"), y0 - 1, y0),
                                 vlak_xz(sleuf, y0 + t, y0 + t + 0.1)])
    # tanden: drie gaatjes voor de pennetjes (de tanden liggen op de voorkant, binnen de lip), met een
    # 45°-punt: in de print is dat het plafond van het gaatje
    for x, z in TAND_PENNEN:
        m = m - cil_y(x, z, 1.25, y0 - 1, 2.6, n=16)
        punt = Manifold.cylinder(1.25, 1.25, 0.0, 16).transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))
        m = m - punt.rotate([180, 0, 0]).translate([x, 2.6 - E, z])
    return m


def tanden():
    y0 = -P.LIP_T
    m = vlak_xz(tanden2d(), y0 - P.TANDEN_T, y0)
    for x, z in TAND_PENNEN:
        m = m + cil_y(x, z, 1.2, y0 - E, 2.3, n=16)
    return m


def lade():
    x, (z0, z1), (y0, y1), w = P.LADE_HALF, P.LADE_Z, P.LADE_Y, P.LADE_WAND
    m = vlak_xz(lade2d(), y0, y1) ^ extrude(rounded_rect(-x, y0, x, y1, 3.0), z0 - 1, z1 + 1)
    binnen = unary_union([lade2d(-w), box(-x + w, z1 - w - 1, x - w, z1 + 1)])
    holte = vlak_xz(binnen, y0 + w + 1.0, y1 - w) ^ extrude(rounded_rect(-x + w, y0 + w + 1.0, x - w, y1 - w, 1.5), z0, z1 + 2)
    m = m - holte
    # greep: halve maan uit de bovenrand van de voorkant
    return m - vlak_xz(Point(0, z1).buffer(11.0, quad_segs=24), y0 - 1, y0 + w + 2.0)


def hoorn():
    """Hoorntje, staand (onderkant op z = 0, midden op (0, 0)), met een gat voor een paspen."""
    m = rev_z([(0.0, 0.0), (8.0, 0.0), (7.6, 2.0), (6.0, 8.0), (3.8, 14.0), (1.8, 18.5), (0.0, 19.8)])
    gat = rev_z([(0.0, -1.0), (2.45, -1.0), (2.45, 8.0), (0.0, 10.45)], n=32)
    return m - gat


def paspen():
    r, L = P.PASPEN_D / 2, P.PASPEN_L
    return rev_z([(0.0, 0.0), (r - 0.4, 0.0), (r, 0.4), (r, L - 0.4), (r - 0.4, L), (0.0, L)], n=32)


# --- montage ------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def delen():
    """{naam: (kleur, Manifold, beweging)}; beweging: 'vast', 'onder', 'boven', 'oog', 'stang'."""
    d = {
        "romp voor": (P.ORANJE, romp_voor(), "vast"),
        "romp achter": (P.ORANJE, romp_achter(), "vast"),
        "lade": (P.WIT, lade(), "vast"),
        "mondstuk": (P.ZWART, mondstuk(), "vast"),
        "tanden": (P.WIT, tanden(), "vast"),
        "rol onder": (P.ZWART, rol_onder(), "onder"),
        "kruk": (P.ORANJE, kruk(), "onder"),
        "knop": (P.WIT, knop(), "onder"),
        "rol boven": (P.ZWART, rol_boven(), "boven"),
        "nokas": (P.ZWART, nokas(), "boven"),
        "ogen": (P.WIT, ogen(), "oog"),
        "pupil links": (P.ZWART, pupil(-P.OOG_X), "oog"),
        "pupil rechts": (P.ZWART, pupil(P.OOG_X), "oog"),
        "tuimelaar": (P.ZWART, tuimelaar(), "oog"),
        "drijfstang": (P.ZWART, drijfstang(), "stang"),
    }
    for i, gx in enumerate(P.GROEF):
        d[f"o-ring {i + 1}"] = ("rubber", oring(gx), "onder")
    for kant, x in (("links", -P.HOORN_X), ("rechts", P.HOORN_X)):
        d[f"hoorn {kant}"] = (P.WIT, hoorn().translate([x, P.HOORN_Y, P.H]), "vast")
        d[f"paspen hoorn {kant}"] = (P.ORANJE, paspen().translate([x, P.HOORN_Y, P.H - 8.0]), "vast")
    for i, (x, z) in enumerate(P.PASPENNEN):
        pen = paspen().transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float))  # z -> y
        d[f"paspen {i + 1}"] = (P.ORANJE, pen.translate([x, P.Y_DEEL - P.PASPEN_L / 2, z]), "vast")
    return d


RUBBER = "rubber"


def stand(phi=0.0, zonder=()):
    """Alle onderdelen bij krukhoek phi: {naam: (kleur, Manifold)}."""
    pk, q, hoek = vierstang(phi)
    dh = hoek - hoek_midden()
    stang_hoek = math.degrees(math.atan2(pk[1] - q[1], pk[0] - q[0]))
    out = {}
    for naam, (kleur, m, beweging) in delen().items():
        if naam in zonder:
            continue
        if beweging == "onder":
            m = op(draai_x(m, -phi), A_ONDER)
        elif beweging == "boven":
            m = op(draai_x(m, phi), A_BOVEN)
        elif beweging == "oog":
            m = op(draai_x(m, dh), B_OOG)
        elif beweging == "stang":
            m = op(draai_x(m, stang_hoek), q)
        out[naam] = (kleur, m)
    return out
