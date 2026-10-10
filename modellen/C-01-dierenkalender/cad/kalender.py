"""Gedeelde geometrie van C-01: kaartgeleiders, dierenfront, voet, kaartjes, pasproef en montage.

Alle functies geven {filamentcode: Manifold} terug. bouw.py schrijft er STL- en 3MF-bestanden van.
"""
import math

import numpy as np
from manifold3d import Manifold
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from dieren import PER_SLEUTEL
from dieren.kit import Laag, blad, ster
import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import (FONT_SMAL, box3, ellipse, extrude, extrude_profile, rot_x, rounded_rect, smooth, stroke, text,
                  translate, union)
from params import *  # noqa: F403
from talen import TALEN


# ---------------------------------------------------------------------------
# Kaartgeleiders (gedeeld door front en pasproefstrook)
# ---------------------------------------------------------------------------

def layout(card_widths, rail_w=RAIL_W):
    """Kanalen en ribben naast elkaar, gecentreerd rond x = 0.

    Een rib is (xa, xb, kanaal links, kanaal rechts); None = geen kanaal aan die kant.
    """
    chans = [w + 2 * SIDE_CLEAR for w in card_widths]
    total = sum(chans) + rail_w * (len(chans) + 1)
    x = -total / 2
    channels, rails = [], []
    for i, c in enumerate(chans):
        rails.append((x, x + rail_w, i - 1 if i else None, i))
        x += rail_w
        channels.append((x, x + c))
        x += c
    rails.append((x, x + rail_w, len(chans) - 1, None))
    return channels, rails


def lip(x_face, z_low, z_top, direction):
    """45°-lip vanaf een ribvlak: onderkant loopt onder 45° omhoog, dus printbaar zonder support."""
    d = direction * LIP
    return Polygon([(x_face, z_low), (x_face + d, z_low + LIP), (x_face + d, z_top), (x_face, z_top)])


def guides(channels, rails, rim_y0, depths, gap=RIM_GAP):
    """Ribben met lippen en een draagrand per kanaal. Geeft (geleiders, uitsparingen).

    De uitsparingen (invoerschuintes en nagelkuiltjes) gaan van plaat + geleiders af.
    depths: vrije kanaaldiepte per kanaal (2,4 mm op het front, 2,2/2,4/2,6 op de proef).
    """
    E = 0.01  # kleine overlap, zodat delen elkaar echt raken
    card_y0 = rim_y0 + RIM_H
    top_y = card_y0 + CARD_H - CARD_EXPOSED
    zl = [PLATE_T + d for d in depths]
    zt = [guide_top_z(d) for d in depths]
    parts = []
    for xa, xb, left, right in rails:
        top = max(zt[i] for i in (left, right) if i is not None)
        prof = box(xa, PLATE_T - E, xb, top)
        if right is not None:
            prof = prof.union(lip(xb - E, zl[right], zt[right], +1))
        if left is not None:
            prof = prof.union(lip(xa + E, zl[left], zt[left], -1))
        # lokaal (X, Y=z, Z=lengte)  ->  wereld (x, y = top_y - Z, z = Y)
        m = [[1, 0, 0, 0], [0, 0, -1, top_y], [0, 1, 0, 0], [0, 0, 0, 1]]
        parts.append(extrude_profile(prof, top_y - rim_y0, m))
    cuts = []
    for (ca, cb), z_low, z_top in zip(channels, zl, zt):
        rim = box(rim_y0, PLATE_T - E, card_y0, z_top).union(
            Polygon([(card_y0 - E, z_low), (card_y0 + LIP, z_low + LIP), (card_y0 + LIP, z_top),
                     (card_y0 - E, z_top)]))
        cc = (ca + cb) / 2
        for xa, xb in ((ca - E, cc - gap / 2), (cc + gap / 2, cb + E)):
            # lokaal (X=y, Y=z, Z=lengte)  ->  wereld (x = xa + Z, y = X, z = Y)
            m = [[0, 0, 1, xa], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            parts.append(extrude_profile(rim, xb - xa, m))
        # nagelkuiltje achter de onderrand van het kaartje, in de opening van de draagrand
        cup = ellipse(cc, card_y0, gap / 2 - 0.5, 4.0).intersection(box(cc - 9, rim_y0 + 0.6, cc + 9, card_y0 + 4))
        cuts.append(extrude(cup, PLATE_T - 1.0, PLATE_T + 0.01))
    # 45°-afloop aan het boveneind van alle geleiders, van de plaat tot boven de ribben
    top = max(zt) + 2
    ramp = Polygon([(top_y, PLATE_T + E), (top_y + 1, PLATE_T + E), (top_y + 1, top), (top_y - (top - PLATE_T - E), top)])
    x0, x1 = rails[0][0] - 1, rails[-1][1] + 1
    m = [[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    cuts.append(extrude_profile(ramp, x1 - x0, m))
    return union(parts), union(cuts)


def tongue(width, y_top=0.5):
    """2D-tong met afgeronde onderhoeken (gaat 0,5 mm het bord in)."""
    r = TONGUE_R
    return unary_union([rounded_rect(-width / 2, -TONGUE_L, width / 2, -TONGUE_L + 3 * r, r),
                        box(-width / 2, -TONGUE_L + r, width / 2, y_top)])


def tongue_chamfers(width):
    """45°-afschuining langs de onderrand van de tong, aan voor- en achterzijde."""
    c, y, E = TONGUE_CHAMFER, -TONGUE_L, 0.01
    m = [[0, 0, 1, -width / 2 - 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    tris = [Polygon([(y - E, -E), (y + c, -E), (y - E, c)]),
            Polygon([(y - E, PLATE_T + E), (y + c, PLATE_T + E), (y - E, PLATE_T - c)])]
    return union([extrude_profile(t, width + 2, m) for t in tris])


def schilder(lagen):
    """Schildervolgorde: een laag verliest wat latere lagen van een andere kleur bedekken.

    Lagen van dezelfde kleur worden alleen samengevoegd (niet van elkaar afgetrokken), zodat er
    geen flinterdunne randjes ontstaan. Geeft {filament: Manifold}.
    """
    later, out = {}, {}
    for laag in reversed(lagen):
        m = extrude(laag.vorm, laag.z0, laag.z1)
        if m.is_empty():
            continue
        others = [v for f, v in later.items() if f != laag.filament]
        zichtbaar = m - union(others) if others else m
        later.setdefault(laag.filament, Manifold())
        later[laag.filament] = union([later[laag.filament], m])
        out.setdefault(laag.filament, []).append(zichtbaar)
    return {f: union(ms) for f, ms in out.items()}


def bord2d():
    hw = BOARD_W / 2
    return unary_union([rounded_rect(-hw, 0, hw, BOARD_TOP, BOARD_R), box(-hw, 0, hw, BOARD_R)])


def tekening(dier):
    """De tekening van het dier, geschaald (rond het midden van de bordrand) tot hij in 194 x 196 mm past."""
    T = dier.tekening()
    x0, _, x1, top = T.silhouet.bounds
    k = min(1.0, (ART_TOP - BOARD_TOP) / (top - BOARD_TOP), ART_MAX_W / (x1 - x0))
    if k < 1.0:
        sc = lambda g: affinity.scale(g, k, k, origin=(0, BOARD_TOP))  # noqa: E731
        T.silhouet = sc(T.silhouet)
        T.lagen = [Laag(l.filament, sc(l.vorm), l.z0, l.z1) for l in T.lagen]
    return T


def decor(dier, centers, y=21.0):
    """Drie blaadjes of sterretjes onder de vensters; geeft lagen."""
    fil, soort = dier.decor
    if not soort:
        return []
    xs = [centers[0], (centers[1] + centers[2]) / 2, centers[3]]
    if soort == "blad":
        vormen = [blad(x, y, rot=r) for x, r in zip(xs, (20, -20, 20))]
        bladeren = unary_union([v for v, _ in vormen])
        nerven = unary_union([n for _, n in vormen])
        return [Laag(fil, bladeren, PLATE_T - 0.01, PLATE_T + 0.8),
                Laag(dier.bord, nerven, PLATE_T + 0.4, PLATE_T + 0.9)]   # nerf als groefje
    if soort == "ster":
        sterren = unary_union([ster(x, y, r=10.5 if i != 1 else 12.5, rot=r)
                               for i, (x, r) in enumerate(zip(xs, (-8, 0, 8)))])
        return [Laag(fil, sterren, PLATE_T - 0.01, PLATE_T + 0.8)]
    if soort == "ring":
        ringen = unary_union([ellipse(x, y, 9, 9).difference(ellipse(x, y, 5.6, 5.6)) for x in xs])
        return [Laag(fil, ringen, PLATE_T - 0.01, PLATE_T + 0.8)]
    if soort == "web":
        webben = []
        for x in xs:
            spaken = [stroke([(x, y), (x + 11 * math.cos(math.radians(a)), y + 11 * math.sin(math.radians(a)))], 1.2)
                      for a in range(0, 360, 45)]
            ringen = [stroke([(x + r * math.cos(math.radians(a)), y + r * math.sin(math.radians(a)))
                              for a in range(0, 361, 45)], 1.2) for r in (4.5, 8.5)]
            webben += spaken + ringen
        return [Laag(fil, unary_union(webben), PLATE_T - 0.01, PLATE_T + 0.6)]
    if soort == "tnt":
        lagen = []
        for x in xs:
            blok = rounded_rect(x - 10, y - 10, x + 10, y + 10, 1.0)
            band = box(x - 10, y - 3.6, x + 10, y + 3.6)
            lagen += [Laag(ROOD, blok, PLATE_T - 0.01, PLATE_T + 0.8),
                      Laag(WIT, band.intersection(blok), PLATE_T + 0.2, PLATE_T + 0.8),
                      Laag(ZWART, text("TNT", 4.6, x, y), PLATE_T + 0.4, PLATE_T + 0.8)]
        return lagen
    raise ValueError(soort)


def front(sleutel, taal, channel=CHANNEL):
    """Dierenfront voor een dier en een taal. Geeft ({filament: Manifold}, kanalen)."""
    dier = PER_SLEUTEL[sleutel]
    T = tekening(dier)
    t = TALEN[taal]
    board = bord2d()
    tong = tongue(TONGUE_W)
    # kleine sluiting: voorkomt punten waar twee vormen elkaar maar net raken (niet-manifold in STL)
    plate2d = smooth(unary_union([board, tong, T.silhouet]), close=0.4)

    channels, rails = layout([CARD_W[k] for k in KINDS])
    g, cuts = guides(channels, rails, WINDOW_Y0, [channel] * len(channels))
    centers = [(a + b) / 2 for a, b in channels]

    lagen = [Laag(dier.bord, plate2d, 0, PLATE_T)]
    if dier.kop != dier.bord:
        # kop in eigen kleur door de hele plaatdikte, zodat ook de zijkant de goede kleur heeft
        # (bord en tong iets groter afgetrokken: anders blijven er flinterdunne randjes langs de bordrand over)
        kop2d = plate2d.difference(unary_union([board, tong]).buffer(0.05)).intersection(box(-150, 1, 150, 300))
        lagen.append(Laag(dier.kop, kop2d, 0, PLATE_T))
    label_y = WINDOW_Y0 + RIM_H + CARD_H + 7.0
    l = t["labels"]
    labels = unary_union([text(l[0], 6.5, centers[0], label_y), text(l[1], 6.5, (centers[1] + centers[2]) / 2, label_y),
                          text(l[2], 6.5, centers[3], label_y)])
    lagen.append(Laag(dier.label, labels, PLATE_T - LABEL_INLAY, PLATE_T))
    lagen += decor(dier, centers)
    # dierlagen 0,05 mm binnen de plaatrand: anders vallen zijwanden net niet samen (niet-manifold).
    # Lagen door de volle dikte (manen, stekels) nemen de buitenrand van de plaat in hun buurt over,
    # zodat er in de inhammen tussen lokken geen haarfijne randjes van de kopkleur achterblijven.
    binnen = plate2d.buffer(-0.05)
    rand = plate2d.difference(plate2d.buffer(-0.8))
    for l in T.lagen:
        vorm = l.vorm
        if l.z0 < 0.5:
            vorm = unary_union([vorm, rand.intersection(vorm.buffer(0.8))])
        lagen.append(Laag(l.filament, vorm.intersection(binnen) if l.z0 >= 0.5 else vorm.intersection(plate2d),
                          l.z0, l.z1))

    kleuren = schilder(lagen)
    kleuren[dier.bord] = union([kleuren.get(dier.bord, Manifold()), g - union(list(kleuren.values()))])

    # versielabel achterop, verzonken in de onderkant en gespiegeld zodat het van achteren leesbaar is
    label = f"C-01 {dier.naam[taal]} {VERSIE}" + (" fan-art" if dier.fanart else "")
    tag = affinity.scale(text(label, 4.5, 0, 12), -1, 1, origin=(0, 0))
    weg = union([cuts, tongue_chamfers(TONGUE_W), extrude(tag, -0.01, 0.4)])
    return {f: m - weg for f, m in kleuren.items() if not (m - weg).is_empty()}, channels


# ---------------------------------------------------------------------------
# Voet met vier opbergbakken, sleuf en achtervak
# ---------------------------------------------------------------------------

def slot_cut(x_len, y_front, z_top, width=SLOT_W, depth=SLOT_DEPTH, tilt=TILT_DEG, chamfer=SLOT_CHAMFER):
    """Sleuf die `tilt` graden achterover leunt; voorkant op (y_front, z_top), mond 45° afgeschuind."""
    s = box3(-x_len / 2, 0, -depth, x_len / 2, width, 6)
    # trechter: afgeknotte piramide met 45°-wanden; onderkant iets kleiner dan de sleuf,
    # zodat de schuine vlakken de sleufwanden snijden in plaats van erop te vallen
    c, t, hx = chamfer, 6.0, x_len / 2
    lo, hi = -c - 0.5, t
    pts = [(sx * (hx - 0.5), y, lo) for sx in (-1, 1) for y in (0.5, width - 0.5)]
    pts += [(sx * (hx + c + t), y, hi) for sx in (-1, 1) for y in (-c - t, width + c + t)]
    s = union([s, Manifold.hull_points(pts)])
    return s.transform((translate(0, y_front, z_top) @ rot_x(-tilt))[:3, :])


def foot():
    hw = FOOT_W / 2
    body = extrude(rounded_rect(-hw, 0, hw, FOOT_D, FOOT_R), 0, FOOT_H)

    bins = []
    total = sum(BIN_INNER_W) + WALL * (len(BIN_INNER_W) - 1)
    x = -total / 2
    bin_back = WALL + BIN_INNER_D
    for w in BIN_INNER_W:
        # binnenhoek R0,8: een kaartje kan helemaal tegen de achterwand (56 mm in 58 mm)
        bins.append(extrude(rounded_rect(x, WALL, x + w, bin_back, 0.8), FLOOR, FOOT_H + 1))
        x += w + WALL

    # lage voorwand met vlakke bovenkant, daarachter lopen de wanden schuin op naar de achterwand
    wedge = Polygon([(-1, BIN_FRONT_H), (WALL, BIN_FRONT_H), (bin_back, FOOT_H), (bin_back, FOOT_H + 2),
                     (-1, FOOT_H + 2)])
    m = [[0, 0, 1, -hw - 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    slope = extrude_profile(wedge, FOOT_W + 2, m)

    slot = slot_cut(SLOT_L, SLOT_FRONT_Y, FOOT_H)

    back = extrude(rounded_rect(-hw + WALL, SLOT_BLOCK_BACK_Y, hw - WALL, FOOT_D - WALL, FOOT_R - WALL),
                   FLOOR, FOOT_H + 1)
    return {"voet": body - union(bins) - slope - slot - back}


# ---------------------------------------------------------------------------
# Kaartjes
# ---------------------------------------------------------------------------

def card(label, kind, x=0.0, y=0.0):
    w = CARD_W[kind]
    outline = rounded_rect(x - w / 2, y, x + w / 2, y + CARD_H, CARD_R)
    vis_cy = y + (LIP + CARD_H) / 2  # midden van het zichtbare deel; de onderlip dekt 2 mm
    if kind == "cijfer":
        t = text(label, 18.0, x, vis_cy, max_w=w - 6)
    else:
        # smalle Fredoka; alleen DONDERDAG wordt nog ~5 % extra versmald
        t = text(label, 8.0, x, vis_cy, max_w=w - 2 * (LIP - SIDE_CLEAR) - 2, font=FONT_SMAL)
    t = t.intersection(outline.buffer(-1.0))
    body = extrude(outline, 0, CARD_T) - extrude(t, CARD_BASE, CARD_T + 0.01)
    ink = extrude(t, CARD_BASE, CARD_T)
    return {KAART_WIT: body, KAART_ZWART: ink}


def card_plate(items, cols, gap=4.0):
    """items: lijst (label, soort). Legt ze in een raster, geeft {kleur: Manifold}."""
    whites, blacks = [], []
    row_h = CARD_H + gap
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    for r, row in enumerate(rows):
        widths = [CARD_W[k] for _, k in row]
        x = -(sum(widths) + gap * (len(row) - 1)) / 2
        for (label, kind), w in zip(row, widths):
            c = card(label, kind, x + w / 2, -r * row_h)
            whites.append(c[KAART_WIT])
            blacks.append(c[KAART_ZWART])
            x += w + gap
    return {KAART_WIT: union(whites), KAART_ZWART: union(blacks)}


def kaartjes(taal, soort):
    """soort: 'maanden', 'dagen' of 'cijfers' (cijfers zijn voor elke taal gelijk)."""
    t = TALEN[taal]
    if soort == "maanden":
        return card_plate([(m, "maand") for m in t["maanden"]], cols=3)
    if soort == "dagen":
        return card_plate([(d, "dag") for d in t["dagen"]], cols=4)
    return card_plate([(d, "cijfer") for d in TIENTALLEN + EENHEDEN], cols=7)


# ---------------------------------------------------------------------------
# Montage: alle delen in gebruiksstand (gedeeld door controle.py en render.py)
# ---------------------------------------------------------------------------

KINDS = ["dag", "cijfer", "cijfer", "maand"]
POSITIES = ["dag", "tiental", "eenheid", "maand"]


def mounted(slot_front_y=SLOT_FRONT_Y, z_top=FOOT_H, slot_w=SLOT_W, tilt=TILT_DEG):
    """4x4-matrix: liggend gemodelleerd front -> in de sleuf, `tilt`° achterover.

    De achterste onderrand van de schouders rust op de voet en de tong hangt midden in de sleuf.
    """
    lean = rot_x(90 - tilt)
    gap = (slot_w - PLATE_T) / 2
    p = (lean @ [0, 0, PLATE_T, 1])[:3]  # voorvlak, onderrand
    a = math.radians(tilt)
    ty = slot_front_y - p[1] + (gap + math.sin(a) * p[2]) / math.cos(a)
    return translate(0, ty, z_top) @ lean


def stand_card(x, y_back, z=FLOOR):
    """Kaartje rechtop, tekst naar voren (-y), achterkant op y_back, onderkant op z."""
    return translate(x, y_back, z) @ rot_x(90)


def bin_stacks(taal, date):
    t = TALEN[taal]
    return [[d for d in t["dagen"] if d != date[0]], [d for d in TIENTALLEN if d != date[1]],
            [d for d in EENHEDEN if d != date[2]], [m for m in t["maanden"] if m != date[3]]]


def bin_centers():
    x = -(sum(BIN_INNER_W) + WALL * (len(BIN_INNER_W) - 1)) / 2
    out = []
    for w in BIN_INNER_W:
        out.append(x + w / 2)
        x += w + WALL
    return out


def assembly(sleutel="vos", taal="nl", date=None, explode=0.0, full_bins=False):
    """Alle onderdelen in gemonteerde stand (mm): lijst (naam, filament, Manifold).

    explode > 0 schuift de delen uit elkaar voor een explosietekening.
    full_bins=True zet alle 33 kaartjes in de bakken (vulproef).
    """
    def T(m, M):
        return m.transform(np.asarray(M)[:3, :])

    t = TALEN[taal]
    date = date or t["vandaag"]
    dier = PER_SLEUTEL[sleutel]
    parts = [("voet", dier.bord, m) for m in foot().values()]
    fr, channels = front(sleutel, taal)
    a = math.radians(TILT_DEG)
    up = np.array([0, math.sin(a), math.cos(a)])          # langs het schuine front omhoog
    out = np.array([0, -math.cos(a), math.sin(a)])        # loodrecht uit het front
    M = translate(*(up * 70 * explode)) @ mounted()
    parts += [("front", c, T(m, M)) for c, m in fr.items()]
    if not full_bins:
        for (ca, cb), label, kind in zip(channels, date, KINDS):
            place = translate(*((up * 25 + out * 45) * explode)) @ M @ translate(
                (ca + cb) / 2, WINDOW_Y0 + RIM_H, PLATE_T + 0.002)
            parts += [(f"kanaal {label}", c, T(m, place)) for c, m in card(label, kind).items()]
    stacks = ([t["dagen"], TIENTALLEN, EENHEDEN, t["maanden"]] if full_bins else bin_stacks(taal, date))
    for cx, kind, pos, labels in zip(bin_centers(), KINDS, POSITIES, stacks):
        y = WALL + BIN_INNER_D - 0.3                       # achterste kaartje 0,3 mm van de achterwand
        for i, label in enumerate(reversed(labels)):       # eerste van de lijst staat vooraan
            place = translate(0, -20 * explode, 40 * explode * (1 + 0.04 * i)) @ stand_card(cx, y, FLOOR + 0.002)
            parts += [(f"bak {pos} {label}", c, T(m, place)) for c, m in card(label, kind).items()]
            y -= CARD_T + 0.05
    return parts, channels


# ---------------------------------------------------------------------------
# Pasproefstroken
# ---------------------------------------------------------------------------

PROEF_KANALEN = [2.2, 2.4, 2.6]
PROEF_SLEUVEN = [3.6, 3.8, 4.0]
PROEF_TONG_W = 28.0


def proef_kanalen():
    """Drie cijferkanalen naast elkaar (2,2 / 2,4 / 2,6 mm) met onderaan een tong van 3 mm."""
    channels, rails = layout([CARD_W["cijfer"]] * len(PROEF_KANALEN))
    rim_y0 = 4.0
    g, cuts = guides(channels, rails, rim_y0, PROEF_KANALEN)
    w = rails[-1][1] - rails[0][0] + 4
    h = rim_y0 + RIM_H + CARD_H + 9
    plate = unary_union([rounded_rect(-w / 2, 0, w / 2, h, 3), box(-w / 2, 0, w / 2, 3), tongue(PROEF_TONG_W)])
    labels = unary_union([text(f"{d:.1f}".replace(".", ","), 4.5, (a + b) / 2, h - 4.5)
                          for (a, b), d in zip(channels, PROEF_KANALEN)])
    m = (union([extrude(plate, 0, PLATE_T), g]) - cuts - tongue_chamfers(PROEF_TONG_W)
         - extrude(labels, PLATE_T - 0.6, PLATE_T + 1))
    return {"proef": m}


def proef_sleuven():
    """Blokje met drie schuine sleuven (3,6 / 3,8 / 4,0 mm) op dezelfde 8° als de voet."""
    n = len(PROEF_SLEUVEN)
    slot_len = PROEF_TONG_W + (SLOT_L - TONGUE_W)
    pitch = slot_len + 6
    w, d, h = n * pitch + 4, 24.0, SLOT_DEPTH + 6
    body = extrude(rounded_rect(-w / 2, 0, w / 2, d, 3), 0, h)
    cuts, labels = [], []
    for i, sw in enumerate(PROEF_SLEUVEN):
        cx = (i - (n - 1) / 2) * pitch
        s = slot_cut(slot_len, 10.0, h, width=sw)
        cuts.append(s.translate([cx, 0, 0]))
        labels.append(text(f"{sw:.1f}".replace(".", ","), 4.5, cx, 4.5))
    # opschrift in de bovenkant, voor de sleuven
    lab = extrude(unary_union(labels), h - 0.6, h + 1)
    return {"proef": body - union(cuts) - lab}


def proef_kaartjes(taal):
    return card_plate([("0", "cijfer"), ("1", "cijfer"), ("2", "cijfer"), (TALEN[taal]["proefdag"], "dag")], cols=4)
