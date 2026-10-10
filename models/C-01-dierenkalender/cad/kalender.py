"""Bouwt alle printonderdelen van C-01 en schrijft STL-bestanden per kleur.

Gebruik:  python kalender.py            (alles)
          python kalender.py proef      (alleen de pasproefstroken)
"""
import json
import sys
from pathlib import Path

import trimesh
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import fox
from geom import (FONT_SMAL, box3, ellipse, extrude, extrude_profile, rot_x, rounded_rect, save_stl,
                  smooth, stroke, text, translate, union)
from params import *  # noqa: F403

ROOT = Path(__file__).resolve().parent.parent
STL = ROOT / "stl"

ORANJE, WIT, ZWART, GROEN = ("Pumpkin Orange 10301", "Ivory White 11100",
                             "Charcoal 11101", "Mistletoe Green 10502")


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
    """Ribben met lippen en een draagrand per kanaal. Geeft (geleiders, nagelkuiltjes).

    depths: vrije kanaaldiepte per kanaal (2,2 mm op het front, 2,0/2,2/2,4 op de proef).
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
    scoops = []
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
        scoops.append(extrude(cup, PLATE_T - 1.0, PLATE_T + 0.01))
    return union(parts), union(scoops)


def layered(colors):
    """Hogere kleuren winnen: zwart > wit > groen > oranje. Geeft {kleur: Manifold}."""
    order = [ZWART, WIT, GROEN, ORANJE]
    out, taken = {}, None
    for c in order:
        if c not in colors:
            continue
        m = colors[c]
        if taken is not None:
            m = m - taken
        out[c] = m
        taken = m if taken is None else union([taken, m])
    return out


# ---------------------------------------------------------------------------
# Dierenfront (vos)
# ---------------------------------------------------------------------------

def leaf(cx, cy, length=24.0, width=11.0, rot=0.0):
    r = (length ** 2 / 4 + width ** 2 / 4) / width
    d = r - width / 2
    a = affinity.rotate(ellipse(0, d, r, r).intersection(ellipse(0, -d, r, r)), rot, origin=(0, 0))
    vein = stroke([(-length * 0.36, 0), (length * 0.36, 0)], 1.2)
    vein = affinity.rotate(vein, rot, origin=(0, 0))
    return affinity.translate(a, cx, cy), affinity.translate(vein, cx, cy)


def front_fox(channel=CHANNEL):
    P = fox.fox_parts(top=TOTAL_H - TONGUE_L)
    hw = BOARD_W / 2
    board = unary_union([rounded_rect(-hw, 0, hw, BOARD_TOP, BOARD_R), box(-hw, 0, hw, BOARD_R)])
    tongue = box(-TONGUE_W / 2, -TONGUE_L, TONGUE_W / 2, 0.5)
    # kleine sluiting: voorkomt punten waar twee vormen elkaar maar net raken (niet-manifold in STL)
    plate2d = smooth(unary_union([board, tongue, P["outline"], P["paws"]]), close=0.4)

    channels, rails = layout([CARD_W["dag"], CARD_W["cijfer"], CARD_W["cijfer"], CARD_W["maand"]])
    g, scoops = guides(channels, rails, WINDOW_Y0, [channel] * len(channels))

    head_top = PLATE_T + HEAD_RELIEF
    relief = smooth(unary_union([P["outline"].buffer(-0.01), P["paws"]]), close=0.4)
    orange = union([
        extrude(plate2d, 0, PLATE_T),
        extrude(relief, PLATE_T - 0.01, head_top),
        extrude(P["paws"], head_top - 0.01, PAW_TOP),
        g,
    ]) - scoops

    # kleurvlakken houden 0,05 mm afstand van de pootjes, zodat er geen raakpunten ontstaan
    paws_free = P["paws"].buffer(0.05)
    white = extrude(unary_union([P["mask"], P["inner_ears"]]).difference(paws_free),
                    head_top - 0.01, head_top + COLOR_LAYER)
    black = union([
        extrude(P["details"].difference(paws_free), head_top - 0.01, DETAIL_TOP),
        extrude(P["toes"], PAW_TOP - 0.4, PAW_TOP + 0.2),
    ])

    # labels boven de vensters
    label_y = WINDOW_Y0 + RIM_H + CARD_H + 7.0
    c = [(a + b) / 2 for a, b in channels]
    labels = unary_union([text("Dag", 6.5, c[0], label_y), text("Datum", 6.5, (c[1] + c[2]) / 2, label_y),
                          text("Maand", 6.5, c[3], label_y)])
    black = union([black, extrude(labels, PLATE_T - 0.01, PLATE_T + 0.6)])

    # drie blaadjes onder de vensters
    leaves, veins = zip(*[leaf(c[0], 21, rot=20), leaf((c[1] + c[2]) / 2, 21, rot=-20), leaf(c[3], 21, rot=20)])
    green = extrude(unary_union(leaves), PLATE_T - 0.01, PLATE_T + 0.8) - extrude(
        unary_union(veins), PLATE_T + 0.4, PLATE_T + 0.9)

    return layered({ORANJE: orange, WIT: white, ZWART: black, GROEN: green}), channels


# ---------------------------------------------------------------------------
# Voet met vier opbergbakken, sleuf en achtervak
# ---------------------------------------------------------------------------

def slot_cut(x_len, y_front, z_top, width=SLOT_W, depth=SLOT_DEPTH, tilt=TILT_DEG):
    """Sleuf die `tilt` graden achterover leunt; voorkant op (y_front, z_top)."""
    s = box3(-x_len / 2, 0, -depth, x_len / 2, width, 6)
    return s.transform((translate(0, y_front, z_top) @ rot_x(-tilt))[:3, :])


def foot():
    hw = FOOT_W / 2
    body = extrude(rounded_rect(-hw, 0, hw, FOOT_D, FOOT_R), 0, FOOT_H)

    bins = []
    total = sum(BIN_INNER_W) + WALL * (len(BIN_INNER_W) - 1)
    x = -total / 2
    bin_back = WALL + BIN_INNER_D
    for w in BIN_INNER_W:
        bins.append(extrude(rounded_rect(x, WALL, x + w, bin_back, 1.5), FLOOR, FOOT_H + 1))
        x += w + WALL

    # schuine bovenkant: lage voorkant, oplopend naar de achterwand van de bakken
    wedge = Polygon([(-1, BIN_FRONT_H), (bin_back, FOOT_H), (bin_back, FOOT_H + 2), (-1, FOOT_H + 2)])
    m = [[0, 0, 1, -hw - 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    slope = extrude_profile(wedge, FOOT_W + 2, m)

    slot = slot_cut(SLOT_L, SLOT_FRONT_Y, FOOT_H)

    back = extrude(rounded_rect(-hw + WALL, SLOT_BLOCK_BACK_Y, hw - WALL, FOOT_D - WALL, FOOT_R - WALL),
                   FLOOR, FOOT_H + 1)
    return {ORANJE: body - union(bins) - slope - slot - back}


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
    return {WIT: body, ZWART: ink}


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
            whites.append(c[WIT])
            blacks.append(c[ZWART])
            x += w + gap
    return {WIT: union(whites), ZWART: union(blacks)}


# ---------------------------------------------------------------------------
# Pasproefstroken
# ---------------------------------------------------------------------------

PROEF_KANALEN = [2.0, 2.2, 2.4]
PROEF_SLEUVEN = [3.6, 3.8, 4.0]
PROEF_TONG_W = 28.0


def proef_kanalen():
    """Drie cijferkanalen naast elkaar (2,0 / 2,2 / 2,4 mm) met onderaan een tong van 3 mm."""
    channels, rails = layout([CARD_W["cijfer"]] * len(PROEF_KANALEN))
    rim_y0 = 4.0
    g, scoops = guides(channels, rails, rim_y0, PROEF_KANALEN)
    w = rails[-1][1] - rails[0][0] + 4
    h = rim_y0 + RIM_H + CARD_H + 9
    plate = unary_union([rounded_rect(-w / 2, 0, w / 2, h, 3), box(-w / 2, 0, w / 2, 3),
                         box(-PROEF_TONG_W / 2, -TONGUE_L, PROEF_TONG_W / 2, 0.5)])
    labels = unary_union([text(f"{d:.1f}".replace(".", ","), 4.5, (a + b) / 2, h - 4.5)
                          for (a, b), d in zip(channels, PROEF_KANALEN)])
    m = union([extrude(plate, 0, PLATE_T), g]) - scoops - extrude(labels, PLATE_T - 0.6, PLATE_T + 1)
    return {ORANJE: m}


def proef_sleuven():
    """Blokje met drie schuine sleuven (3,6 / 3,8 / 4,0 mm) op dezelfde 8° als de voet."""
    n = len(PROEF_SLEUVEN)
    slot_len = PROEF_TONG_W + 0.8
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
    return {ORANJE: body - union(cuts) - lab}


def proef_kaartjes():
    return card_plate([("0", "cijfer"), ("1", "cijfer"), ("2", "cijfer"), ("WOENSDAG", "dag")], cols=4)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

SHORT = {ORANJE: "oranje", WIT: "wit", ZWART: "zwart", GROEN: "groen"}


def export(name, colors, folder, info):
    files = {}
    for c, m in colors.items():
        if m.is_empty():
            continue
        path = folder / f"{name}__{SHORT[c]}.stl"
        save_stl(m, path)
        tm = trimesh.load(path)  # opnieuw inlezen zoals een slicer dat doet
        assert tm.is_watertight and tm.is_winding_consistent, path
        files[SHORT[c]] = {"bestand": str(path.relative_to(ROOT)), "filament": c,
                           "volume_cm3": round(float(tm.volume) / 1000, 2),
                           "maat_mm": [round(float(v), 1) for v in tm.extents]}
    info[name] = files
    print(f"{name:18s}", ", ".join(f"{k} {v['maat_mm']}" for k, v in files.items()))


def main(which="alles"):
    index = STL / "onderdelen.json"
    info = json.loads(index.read_text()) if index.exists() else {}
    proef = STL / "1-pasproef"
    export("proef_kanalen", proef_kanalen(), proef, info)
    export("proef_sleuven", proef_sleuven(), proef, info)
    export("proef_kaartjes", proef_kaartjes(), proef, info)
    if which != "proef":
        front, _ = front_fox()
        export("front_vos", front, STL / "2-vos", info)
        export("voet", foot(), STL / "3-voet", info)
        k = STL / "4-kaartjes"
        months = [(m, "maand") for m in MAANDEN]
        days = [(d, "dag") for d in DAGEN]
        export("kaartjes_maanden", card_plate(months, cols=3), k, info)
        export("kaartjes_dagen", card_plate(days, cols=4), k, info)
        digits = [(d, "cijfer") for d in TIENTALLEN] + [(d, "cijfer") for d in EENHEDEN]
        export("kaartjes_cijfers", card_plate(digits, cols=7), k, info)
    index.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "alles")
