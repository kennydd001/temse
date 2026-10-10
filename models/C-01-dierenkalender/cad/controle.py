"""Montage- en printcontrole voor C-01, op de echte geometrie.

Gebruik:  python controle.py
Schrijft  controle/RAPPORT.md, controle/controle.json en renders/doorsneden.png.

Wat wordt gecontroleerd
  1. Botsingen in gemonteerde stand (front in voet, kaartjes in kanalen en bakken).
  2. Gemeten spelingen: kaart-rib, kaart-lip, lipoverlap, tong-sleuf, kaartjes-bakken.
  3. Uitneemroute: elk kaartje omhoog schuiven tot het vrij is van de geleiders.
  4. Printbaarheid per STL: bedmaat A1, overhang > 45°, details smaller dan de nozzle,
     lagen met meerdere kleuren (AMS-wissels).
  5. Stabiliteit: zwaartepunt en kantelhoeken (met slicergewichten als die er zijn).
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kalender as K  # noqa: E402
from geom import rot_x, to_trimesh, translate, union  # noqa: E402
from params import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
BED = (256.0, 256.0)
NOZZLE = 0.4
LAYER = 0.2
PLA = 1.24  # g/cm3

KLEUR = {K.ORANJE: "#FF9016", K.WIT: "#F4F1EA", K.ZWART: "#222222", K.GROEN: "#3F8E43"}


def T(m, M):
    return m.transform(np.asarray(M, dtype=float)[:3, :])


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def cs_to_shapely(cs):
    """CrossSection -> shapely (positieve lussen zijn vulling, negatieve gaten)."""
    fills, holes = [], []
    for loop in cs.to_polygons():
        if len(loop) < 3:
            continue
        p = Polygon(loop)
        (fills if p.exterior.is_ccw else holes).append(p.buffer(0))
    g = unary_union(fills)
    return g.difference(unary_union(holes)) if holes else g


def section(m, plane, value):
    """Doorsnede als shapely-geometrie. plane 'x' -> (y, z); 'y' -> (x, z); 'z' -> (x, y)."""
    if plane == "z":
        return cs_to_shapely(m.slice(value))
    if plane == "y":  # (x, y, z) -> (x, z, -y)
        return cs_to_shapely(T(m, rot_x(-90)).slice(-value))
    # plane 'x': (x, y, z) -> (z, y, -x), daarna assen wisselen naar (y, z)
    g = cs_to_shapely(T(m, rot_y(90)).slice(-value))
    from shapely import affinity
    return affinity.affine_transform(g, [0, 1, 1, 0, 0, 0])


def gap(a, b, search=10.0):
    return float(a.min_gap(b, search))


def fmt(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# 1 + 2 + 3: montage
# ---------------------------------------------------------------------------

def check_montage():
    res = {"botsingen": [], "spelingen": {}, "uitnemen": []}
    parts, channels = K.assembly()
    voet = union([m for n, c, m in parts if n == "voet"])
    front = union([m for n, c, m in parts if n == "front"])
    cards = {}
    for n, c, m in parts:
        if n.startswith(("kanaal", "bak")):
            cards.setdefault(n, []).append(m)
    cards = {n: union(ms) for n, ms in cards.items()}

    def overlap(name, a, b):
        v = (a ^ b).volume()
        res["botsingen"].append({"paar": name, "overlap_mm3": round(v, 4), "ok": v < 0.01})

    overlap("front ↔ voet", front, voet)
    for n, m in cards.items():
        overlap(f"{n} ↔ front", m, front)
        overlap(f"{n} ↔ voet", m, voet)

    # volle bakken: alle 33 kaartjes tegelijk in de voet
    full, _ = K.assembly(full_bins=True)
    stack = {}
    for n, c, m in full:
        if n.startswith("bak"):
            stack.setdefault(n, []).append(m)
    stack = {n: union(ms) for n, ms in stack.items()}
    names = list(stack)
    worst = max((stack[n] ^ voet).volume() for n in names)
    res["botsingen"].append({"paar": "alle 33 kaartjes in de bakken ↔ voet", "overlap_mm3": round(worst, 4),
                             "ok": worst < 0.01})
    worst_cc = 0.0
    for a, b in zip(names, names[1:]):
        worst_cc = max(worst_cc, (stack[a] ^ stack[b]).volume())
    res["botsingen"].append({"paar": "kaartjes onderling in de bakken", "overlap_mm3": round(worst_cc, 4),
                             "ok": worst_cc < 0.01})

    # --- spelingen in het kanaal (front liggend, kaartje in rust tegen de plaat) ---
    fl, _ = K.front_fox()
    fl = union(list(fl.values()))
    card_y0 = WINDOW_Y0 + RIM_H
    top_y = card_y0 + CARD_H - CARD_EXPOSED
    zl = PLATE_T + CHANNEL
    sp = {}
    for (ca, cb), kind, label in zip(channels, K.KINDS, K.VANDAAG):
        c = union(list(K.card(label, kind).values()))
        c = T(c, translate((ca + cb) / 2, card_y0, PLATE_T + 0.002))
        y = card_y0 + 8.0
        S = section(fl, "y", y)
        C = section(c, "y", y)
        cx0, _, cx1, cz1 = C.bounds
        ribs = S.intersection(box(ca - 5, PLATE_T + 0.05, cb + 5, zl - 0.05))
        lips = S.intersection(box(cx0, cz1 + 0.01, cx1, 20))  # alleen wat boven het kaartje hangt
        side = C.distance(ribs)
        up = C.distance(lips)
        over_l = max(0.0, lips.intersection(box(cx0, 0, (cx0 + cx1) / 2, 20)).bounds[2] - cx0) if not lips.is_empty else 0
        over_r = max(0.0, cx1 - lips.intersection(box((cx0 + cx1) / 2, 0, cx1, 20)).bounds[0]) if not lips.is_empty else 0
        sp[f"kanaal {kind} ({label})"] = {
            "zijspeling_mm": round(side, 3),
            "ruimte_kaart_tot_lip_mm": round(up, 3),
            "lipoverlap_links_mm": round(over_l, 3),
            "lipoverlap_rechts_mm": round(over_r, 3),
            "lipoverlap_slechtst_mm": round(min(over_l, over_r) - SIDE_CLEAR, 3),
        }
    res["spelingen"]["kanalen"] = sp

    # --- tong in de sleuf (doorsnede x = 0, in sleufassen gedraaid) ---
    a = TILT_DEG
    Mslot = rot_x(a) @ translate(0, -SLOT_FRONT_Y, -FOOT_H)  # sleufvoorvlak -> y = 0, sleuf staat recht
    fs = T(front, Mslot)
    vs = T(voet, Mslot)
    Fs, Vs = section(fs, "x", 0.0), section(vs, "x", 0.0)
    tongue = Fs.intersection(box(-20, -SLOT_DEPTH - 2, 20, -1.0))
    front_wall = Vs.intersection(box(-30, -SLOT_DEPTH - 1, 0.0 + 1e-3, -1.0))
    back_wall = Vs.intersection(box(SLOT_W - 1e-3, -SLOT_DEPTH - 1, 30, -1.0))
    bottom = Vs.intersection(box(-1, -40, SLOT_W + 1, -SLOT_DEPTH + 1e-3))
    res["spelingen"]["tong_in_sleuf"] = {
        "voorzijde_mm": round(tongue.distance(front_wall), 3),
        "achterzijde_mm": round(tongue.distance(back_wall), 3),
        "onderkant_tong_tot_sleufbodem_mm": round(tongue.distance(bottom), 3),
    }
    # lengterichting: horizontale doorsnede halverwege de sleuf
    zc = FOOT_H - 8.0
    Fz, Vz = section(front, "z", zc), section(voet, "z", zc)
    ends = [Fz.distance(Vz.intersection(box(s * 90, 0, s * 110, FOOT_D))) for s in (-1, 1)]
    res["spelingen"]["tong_in_sleuf"]["uiteinden_links_rechts_mm"] = [round(e, 3) for e in ends]
    # schouders: rust de achterste onderrand op de voet?
    shoulders = front ^ box_m(-BOARD_W / 2 - 1, 0, FOOT_H - 1, -TONGUE_W / 2 - 1, FOOT_D, FOOT_H + 5)
    zmin = shoulders.bounding_box()[2]
    res["spelingen"]["tong_in_sleuf"]["schouder_boven_voet_mm"] = round(zmin - FOOT_H, 3)

    # --- kaartjes in de bakken (alle 33) ---
    zc = FLOOR + 14.0
    Vb = section(voet, "z", zc)
    bins = {}
    for cx, w, pos, labels in zip(K.bin_centers(), BIN_INNER_W, K.POSITIES,
                                   [DAGEN, TIENTALLEN, EENHEDEN, MAANDEN]):
        cs = unary_union([section(stack[f"bak {pos} {lab}"], "z", zc) for lab in labels])
        x0, y0, x1, y1 = cs.bounds
        # zijspeling midden in de stapel (de afgeronde binnenhoeken niet meegerekend)
        band = Vb.intersection(box(cx - w, y0 + 1, cx + w, y1 - 1))
        side = min(cs.distance(band.intersection(box(cx - w, -1, x0, 99))),
                   cs.distance(band.intersection(box(x1, -1, cx + w, 99))))
        corner = cs.distance(Vb.intersection(box(cx - w, y1 - 3, cx + w, y1 + 5)))
        low = FLOOR + 4.0  # onder de bovenkant van de lage voorwand
        cs_low = unary_union([section(stack[f"bak {pos} {lab}"], "z", low) for lab in labels])
        front_room = cs_low.distance(section(voet, "z", low).intersection(box(x0, -1, x1, cs_low.bounds[1])))
        bins[f"bak {pos} ({len(labels)} st.)"] = {"zijspeling_mm": round(side, 3),
                                                  "kleinste_afstand_achterhoek_mm": round(corner, 3),
                                                  "ruimte_voor_stapel_mm": round(front_room, 3)}
    res["spelingen"]["bakken_vol"] = bins

    # --- uitneemroute: kaartje 0,05 mm van de plaat omhoog schuiven tot het vrij is ---
    relief = fl ^ box_m(-120, -30, PLATE_T + 0.01, 120, 230, 20)
    for (ca, cb), kind, label in zip(channels, K.KINDS, K.VANDAAG):
        c = union(list(K.card(label, kind).values()))
        c = T(c, translate((ca + cb) / 2, card_y0, PLATE_T + 0.05))
        stroke_len = top_y - card_y0 + 0.5
        worst, at = 99.0, 0.0
        for t in np.linspace(1.0, stroke_len, 40):  # start 1 mm boven de draagrand
            g = gap(c.translate([0, t, 0]), relief, 5.0)
            if g < worst:
                worst, at = g, t
        top = c.translate([0, stroke_len, 0])
        animal = relief ^ box_m(-120, top_y + 2, -1, 120, 230, 30)
        res["uitnemen"].append({"kaart": f"{kind} ({label})", "slag_mm": round(stroke_len, 1),
                                "kleinste_afstand_onderweg_mm": round(worst, 3),
                                "bij_slag_mm": round(at, 1),
                                "afstand_tot_dier_bovenaan_mm": round(gap(top, animal, 30.0), 2)})
    return res


def box_m(x0, y0, z0, x1, y1, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


# ---------------------------------------------------------------------------
# 4: printbaarheid per STL
# ---------------------------------------------------------------------------

def load_manifold(path):
    tm = trimesh.load(path)
    return tm, Manifold(Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                             tri_verts=np.asarray(tm.faces, dtype=np.uint32)))


def check_print():
    out = {}
    groups = {}
    for f in sorted((ROOT / "stl").glob("*/*.stl")):
        groups.setdefault(f.stem.split("__")[0], []).append(f)
    for name, files in groups.items():
        info = {"kleuren": {}, "overhang": [], "dunne_details": []}
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        sections, solids = {}, []
        for f in files:
            color = f.stem.split("__")[1]
            tm, m = load_manifold(f)
            solids.append(m)
            lo, hi = np.minimum(lo, tm.bounds[0]), np.maximum(hi, tm.bounds[1])
            info["kleuren"][color] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                      "z_mm": [round(float(tm.bounds[0][2]), 2), round(float(tm.bounds[1][2]), 2)]}
            # dunne details: per laag doorsnede, openen met halve nozzle; wat verdwijnt is te smal
            z0, z1 = tm.bounds[0][2], tm.bounds[1][2]
            lost_max, where = 0.0, None
            for z in (np.arange(math.ceil(z1 / LAYER)) + 0.5) * LAYER:  # vast lagenraster voor alle kleuren
                if not z0 < z < z1:
                    continue
                s = cs_to_shapely(m.slice(z))
                if s.is_empty:
                    continue
                sections.setdefault(round(z, 2), set()).add(color)
                opened = s.buffer(-NOZZLE / 2 + 0.01).buffer(NOZZLE / 2 - 0.01)
                lost = s.difference(opened.buffer(0.02))
                for p in getattr(lost, "geoms", [lost]):
                    if p.area > lost_max:
                        lost_max, where = p.area, (round(p.centroid.x, 1), round(p.centroid.y, 1), round(z, 2))
            if lost_max > 0.2:  # kleiner dan 0,2 mm²: slicer laat het weg, niet zichtbaar
                info["dunne_details"].append({"kleur": color, "grootste_te_smalle_stuk_mm2": round(lost_max, 2),
                                              "plek_mm": [float(v) for v in where]})
        # overhang op de plaat als geheel (een kleur die op een andere kleur ligt is ondersteund):
        # vlakken die steiler dan ~48° naar beneden kijken en niet op het bed liggen
        tm = to_trimesh(union(solids).simplify(1e-3))
        n, zc = tm.face_normals, tm.triangles_center[:, 2]
        edge = (n[:, 2] < -0.64) & (n[:, 2] >= -0.74) & (zc > 0.3)   # 40-48°: bewust 45°-vlakken
        info["vlakken_rond_45_graden_mm2"] = round(float(tm.area_faces[edge].sum()), 0)
        mask = (n[:, 2] < -0.74) & (zc > 0.3)
        if mask.any():
            sub = tm.submesh([np.where(mask)[0]], append=True)
            bridges = 0.0
            for comp in sub.split(only_watertight=False):
                if comp.area < 1.0:
                    continue
                # helling t.o.v. horizontaal: 0° = plafond, 45° = schuine lip, 90° = wand
                ang = math.degrees(math.acos(min(1, -float(np.mean(comp.face_normals[:, 2])))))
                b = comp.bounds
                # korte brug: (bijna) horizontaal plafond dat smaller is dan 3 mm, b.v. verzonken letters achterop
                foot = unary_union([Polygon(t[:, :2]).buffer(0) for t in comp.triangles])
                span = 2 * foot.area / max(foot.length, 1e-6)
                if ang < 15 and span <= 3.0:
                    bridges += float(comp.area)
                    continue
                info["overhang"].append({"oppervlak_mm2": round(float(comp.area), 1),
                                         "hoek_tov_horizontaal_graden": round(ang, 1),
                                         "plek_mm": [round(float(v), 1) for v in (b[0] + b[1]) / 2]})
            info["korte_bruggen_mm2"] = round(bridges, 0)
        size = hi - lo
        info["maat_mm"] = [round(float(v), 1) for v in size]
        info["past_op_A1"] = bool(size[0] <= BED[0] - 6 and size[1] <= BED[1] - 6)
        info["vrije_strook_mm"] = [round(BED[0] - float(size[0]), 1), round(BED[1] - float(size[1]), 1)]
        multi = sorted(z for z, cs in sections.items() if len(cs) > 1)
        info["lagen_met_kleurwissel"] = len(multi)
        info["geschatte_wissels"] = int(sum(len(sections[z]) - 1 for z in multi))
        if multi:
            info["kleurwissel_tussen_z_mm"] = [multi[0], multi[-1]]
        out[name] = info
    return out


# ---------------------------------------------------------------------------
# 5: stabiliteit
# ---------------------------------------------------------------------------

def check_stabiliteit(slicer):
    parts, _ = K.assembly()
    # vulgraad (massa / massief volume): uit de slicer als die er is, anders een aanname
    fill = {"voet": 0.45, "front": 0.85, "kaart": 1.0}
    if slicer:
        for key, plate in (("voet", "voet"), ("front", "front_vos"), ("kaart", "kaartjes_maanden")):
            r = slicer.get(plate)
            if r and r.get("gram") and r.get("volume_cm3"):
                fill[key] = r["gram"] / (r["volume_cm3"] * PLA)
    tot_m, mom = 0.0, np.zeros(3)
    for n, c, m in parts:
        key = "voet" if n == "voet" else "front" if n == "front" else "kaart"
        tm = to_trimesh(m)
        mass = tm.volume / 1000 * PLA * fill[key]
        tot_m += mass
        mom += mass * tm.center_mass
    com = mom / tot_m
    fwd = math.degrees(math.atan2(com[1] - 0.0, com[2]))
    back = math.degrees(math.atan2(FOOT_D - com[1], com[2]))
    return {"massa_g": round(tot_m, 0), "zwaartepunt_mm": [round(float(v), 1) for v in com],
            "kantelhoek_voorover_graden": round(fwd, 1), "kantelhoek_achterover_graden": round(back, 1),
            "vulgraad_gebruikt": {k: round(v, 2) for k, v in fill.items()},
            "bron": "slicer" if slicer else "aanname"}


# ---------------------------------------------------------------------------
# Doorsnedetekening
# ---------------------------------------------------------------------------

def draw(res):
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1.2, 1])
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    parts, channels = K.assembly()
    arrow = dict(arrowstyle="->", lw=1.1, color="#1a1a1a")

    def fill(ax, g, color, z=1, alpha=1.0):
        for p in getattr(g, "geoms", [g]):
            if p.is_empty or not hasattr(p, "exterior"):
                continue
            ax.fill(*p.exterior.xy, color=color, ec="#333", lw=0.6, zorder=z, alpha=alpha)
            for i in p.interiors:
                ax.fill(*i.xy, color="white", ec="#333", lw=0.6, zorder=z + 0.1)

    # A: kaartgeleider, doorsnede over de breedte (front liggend zoals geprint, kaartjes in rust)
    ax = axA
    fl, _ = K.front_fox()
    card_y0 = WINDOW_Y0 + RIM_H
    y = card_y0 + 8.0
    for c, m in fl.items():
        fill(ax, section(m, "y", y), KLEUR[c])
    for (ca, cb), kind, label in zip(channels[:2], K.KINDS, K.VANDAAG):
        mm = T(union(list(K.card(label, kind).values())), translate((ca + cb) / 2, card_y0, PLATE_T + 0.002))
        fill(ax, section(mm, "y", y), "#F4F1EA", z=3)
    cb = channels[0][1]
    s = res["spelingen"]["kanalen"][f"kanaal dag ({K.VANDAAG[0]})"]
    ct = PLATE_T + CARD_T
    ax.set_xlim(cb - 7.5, cb + 7.5)
    ax.set_ylim(-0.3, 10.2)
    ax.annotate(f"zijspeling {fmt(s['zijspeling_mm'])} mm", (cb - SIDE_CLEAR / 2, PLATE_T + 0.8),
                (cb - 7.2, 1.2), arrowprops=arrow, fontsize=10)
    ax.annotate(f"vrije ruimte boven het kaartje {fmt(s['ruimte_kaart_tot_lip_mm'])} mm\n(kaartje kan iets naar voren)",
                (cb - SIDE_CLEAR - 0.3, ct + 0.6), (cb - 7.2, 9.0), arrowprops=arrow, fontsize=10)
    ax.annotate(f"45°-lip ligt ≥ {fmt(s['lipoverlap_slechtst_mm'])} mm over de kaartrand,\n"
                "ook als het kaartje helemaal opzij schuift", (cb - LIP + 0.2, PLATE_T + CHANNEL + LIP + 0.3),
                (cb + 0.6, 9.0), arrowprops=arrow, fontsize=10)
    ax.text(cb - 7.2, PLATE_T + 0.25, "kaartje DAG", fontsize=9, zorder=5)
    ax.text(cb + 3.5, PLATE_T + 0.25, "kaartje tiental", fontsize=9, zorder=5)
    ax.set_title("A  Kaartgeleider, doorsnede over de breedte (front ligt plat zoals geprint)", fontsize=11, loc="left")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")

    # B: tong in sleuf, doorsnede x = 0
    ax = axB
    voet = union([m for n, c, m in parts if n == "voet"])
    front = union([m for n, c, m in parts if n == "front"])
    fill(ax, section(voet, "x", 0.0), "#FFB45E")
    fill(ax, section(front, "x", 0.0), KLEUR[K.ORANJE], z=2)
    t = res["spelingen"]["tong_in_sleuf"]
    ax.set_xlim(SLOT_FRONT_Y - 9, SLOT_FRONT_Y + 13)
    ax.set_ylim(FOOT_H - 21, FOOT_H + 7)
    a = math.radians(TILT_DEG)
    zc = FOOT_H - 8
    yf = SLOT_FRONT_Y - (FOOT_H - zc) * math.tan(a)
    ax.annotate(f"voor {fmt(t['voorzijde_mm'])} mm", (yf + 0.2, zc), (SLOT_FRONT_Y - 8.5, zc + 2),
                arrowprops=arrow, fontsize=10)
    ax.annotate(f"achter {fmt(t['achterzijde_mm'])} mm", (yf + SLOT_W - 0.2, zc), (SLOT_FRONT_Y + 6, zc + 4),
                arrowprops=arrow, fontsize=10)
    zb = FOOT_H - SLOT_DEPTH * math.cos(a) + 0.5
    yb = SLOT_FRONT_Y - SLOT_DEPTH * math.sin(a) + SLOT_W / 2
    ax.annotate(f"tong raakt de bodem niet: {fmt(t['onderkant_tong_tot_sleufbodem_mm'])} mm", (yb, zb),
                (SLOT_FRONT_Y + 1.5, FOOT_H - 20), arrowprops=arrow, fontsize=10)
    ax.text(SLOT_FRONT_Y - 8.5, FOOT_H + 6.3,
            f"front leunt {fmt(TILT_DEG, 0)}° achterover; schouders rusten op de voet\n"
            f"sleufmond 45° afgeschuind; uiteinden {fmt(t['uiteinden_links_rechts_mm'][0])} / "
            f"{fmt(t['uiteinden_links_rechts_mm'][1])} mm speling", fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85), zorder=6)
    ax.set_title("B  Tong in de sleuf (doorsnede midden)", fontsize=11, loc="left")
    ax.set_xlabel("y (mm), voorkant voet = 0")
    ax.set_ylabel("z (mm)")

    # C: alle 33 kaartjes in de bakken, bovenaanzicht net boven de voorwand
    ax = axC
    full, _ = K.assembly(full_bins=True)
    zc = FLOOR + 14.0
    fill(ax, section(voet, "z", zc), "#FFB45E")
    cards = {}
    for n, c, m in full:
        if n.startswith("bak"):
            cards.setdefault(n, []).append(m)
    for ms in cards.values():
        fill(ax, section(union(ms), "z", zc), "#F4F1EA", z=3)
    for i, ((k, v), cx) in enumerate(zip(res["spelingen"]["bakken_vol"].items(), K.bin_centers())):
        ax.text(cx, -3 - 9 * (i == 2), f"{k[4:]}\nopzij {fmt(v['zijspeling_mm'])} mm · "
                f"vóór de stapel {fmt(v['ruimte_voor_stapel_mm'], 1)} mm", ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-FOOT_W / 2 - 3, FOOT_W / 2 + 3)
    ax.set_ylim(-22, FOOT_D + 2)
    ax.text(0, (SLOT_BLOCK_BACK_Y + FOOT_D - WALL) / 2, "open achtervak", ha="center", va="center", fontsize=10)
    ax.text(0, (WALL + BIN_INNER_D + SLOT_BLOCK_BACK_Y) / 2, "sleufblok (sleuf begint hoger, op 19,7 mm)",
            ha="center", va="center", fontsize=9)
    ax.set_title("C  Vulproef: alle 33 kaartjes tegelijk in de bakken (bovenaanzicht op 16 mm hoogte)",
                 fontsize=11, loc="left")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    for a_ in (axA, axB, axC):
        a_.set_aspect("equal")
        a_.grid(alpha=0.25)
    fig.suptitle("C-01 dierenkalender: spelingen gemeten in de printbestanden", fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "renders" / "doorsneden.png", dpi=120)


# ---------------------------------------------------------------------------
# Printplaten: bovenaanzicht op het A1-bed
# ---------------------------------------------------------------------------

VOLG = ["oranje", "groen", "wit", "zwart"]
HEX = {"oranje": "#FF9016", "wit": "#F7F5F0", "zwart": "#1E1E1E", "groen": "#3F8E43"}
TITELS = {"proef_kanalen": "1a  Pasproef: kanalen", "proef_sleuven": "1b  Pasproef: sleuven",
          "proef_kaartjes": "1c  Pasproef: kaartjes", "kaartjes_maanden": "2a  Kaartjes: maanden",
          "kaartjes_dagen": "2b  Kaartjes: dagen", "kaartjes_cijfers": "2c  Kaartjes: cijfers",
          "voet": "3  Voet", "front_vos": "4  Front vos (4 kleuren)"}


def printplaten(slicer):
    groups = {}
    for f in sorted((ROOT / "stl").glob("*/*.stl")):
        groups.setdefault(f.stem.split("__")[0], []).append(f)
    order = [n for n in TITELS if n in groups]
    fig, axs = plt.subplots(2, 4, figsize=(18, 10.6))
    for ax, name in zip(axs.flat, order):
        shapes, lo, hi = [], np.full(2, 1e9), np.full(2, -1e9)
        for f in sorted(groups[name], key=lambda f: VOLG.index(f.stem.split("__")[1])):
            color = f.stem.split("__")[1]
            tm, m = load_manifold(f)
            z0, z1 = tm.bounds[0][2], tm.bounds[1][2]
            zs = np.arange(z0 + 0.05, z1, 0.2)
            g = unary_union([cs_to_shapely(m.slice(z)) for z in zs])
            shapes.append((color, g))
            lo, hi = np.minimum(lo, tm.bounds[0][:2]), np.maximum(hi, tm.bounds[1][:2])
        multi = len(shapes) > 1
        w, h = hi - lo
        off = np.array([3, 3]) - lo if multi else np.array([128, 128]) - (lo + hi) / 2
        ax.add_patch(plt.Rectangle((0, 0), 256, 256, color="#3A3A3E", zorder=0))
        for i in range(0, 257, 32):
            ax.plot([i, i], [0, 256], color="#46464B", lw=0.5, zorder=0.5)
            ax.plot([0, 256], [i, i], color="#46464B", lw=0.5, zorder=0.5)
        for color, g in shapes:
            from shapely import affinity
            g = affinity.translate(g, *off)
            for p in getattr(g, "geoms", [g]):
                if p.is_empty or not hasattr(p, "exterior"):
                    continue
                ax.fill(*p.exterior.xy, color=HEX[color], lw=0, zorder=2)
                for r in p.interiors:
                    ax.fill(*r.xy, color="#3A3A3E" if color == shapes[0][0] else HEX[shapes[0][0]], lw=0, zorder=2.1)
        if multi:
            tx, ty = (3 + w + 6, 20) if 256 - (3 + w) >= 45 else (20, 3 + h + 6)
            ax.add_patch(plt.Rectangle((tx, ty), 35, 35, fill=False, ls="--", ec="#DDDDDD", lw=1, zorder=3))
            ax.text(tx + 17.5, ty + 17.5, "spoel-\ntoren", color="#DDDDDD", ha="center", va="center", fontsize=8)
        r = slicer.get(name, {})
        sub = f"{fmt(r['gram'], 0)} g · {r['tijd']}" if r else ""
        if r and r.get("wissels"):
            sub += f" · {r['wissels']} wissels"
        ax.set_title(f"{TITELS[name]}\n{' × '.join(fmt(v, 0) for v in (w, h))} mm   {sub}", fontsize=10, loc="left")
        ax.set_xlim(-2, 258)
        ax.set_ylim(-2, 258)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axs.flat[len(order):]:
        ax.axis("off")
    fig.suptitle("Printplaten op de Bambu Lab A1 (256 × 256 mm). Gewicht en tijd: PrusaSlicer met A1-achtig "
                 "profiel, tijd is een indicatie", fontsize=12, x=0.01, y=0.995, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    fig.savefig(ROOT / "renders" / "printplaten.png", dpi=110)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def rapport(mont, prnt, stab, slicer):
    L = ["# Controle C-01 dierenkalender (vos)", "",
         "Automatisch gegenereerd door `cad/controle.py` uit de STL's en de montage. "
         "Maten in mm. ✅ = in orde, ⚠️ = bekijken.", ""]
    ok = all(b["ok"] for b in mont["botsingen"])
    L += ["## 1. Botsingen in gemonteerde stand", "",
          f"{'✅' if ok else '⚠️'} {len(mont['botsingen'])} combinaties gecontroleerd; "
          f"grootste overlap {fmt(max(b['overlap_mm3'] for b in mont['botsingen']), 4)} mm³.", ""]
    bad = [b for b in mont["botsingen"] if not b["ok"]]
    for b in bad:
        L.append(f"- ⚠️ {b['paar']}: {fmt(b['overlap_mm3'], 3)} mm³")
    L += ["", "## 2. Spelingen", "", "| Kanaal | Kaart ↔ rib | Kaart ↔ lip | Lip over kaartrand (slechtst) |",
          "|---|---|---|---|"]
    for k, v in mont["spelingen"]["kanalen"].items():
        L.append(f"| {k} | {fmt(v['zijspeling_mm'])} | {fmt(v['ruimte_kaart_tot_lip_mm'])} | "
                 f"{fmt(v['lipoverlap_slechtst_mm'])} |")
    t = mont["spelingen"]["tong_in_sleuf"]
    L += ["", "| Tong in sleuf | Speling |", "|---|---|",
          f"| voorzijde | {fmt(t['voorzijde_mm'])} |", f"| achterzijde | {fmt(t['achterzijde_mm'])} |",
          f"| onderkant tong ↔ sleufbodem | {fmt(t['onderkant_tong_tot_sleufbodem_mm'])} |",
          f"| uiteinden links / rechts | {fmt(t['uiteinden_links_rechts_mm'][0])} / "
          f"{fmt(t['uiteinden_links_rechts_mm'][1])} |",
          f"| schouder boven voet (0 = rust erop) | {fmt(t['schouder_boven_voet_mm'], 3)} |",
          "", "| Bak, alle kaartjes erin | Zijspeling | Kleinste afstand in de achterhoek | Ruimte vóór de stapel |",
          "|---|---|---|---|"]
    for k, v in mont["spelingen"]["bakken_vol"].items():
        L.append(f"| {k} | {fmt(v['zijspeling_mm'])} | {fmt(v['kleinste_afstand_achterhoek_mm'])} | "
                 f"{fmt(v['ruimte_voor_stapel_mm'], 1)} |")
    L += ["", "![Doorsneden](../renders/doorsneden.png)", "",
          "## 3. Uitneemroute", "",
          "Elk kaartje 0,05 mm van de plaat en vanaf 1 mm boven de draagrand omhoog geschoven tot de onderrand "
          "boven de geleiders is. De kleinste afstand onderweg is meestal de zijspeling langs de ribben.", "",
          "| Kaartje | Slag | Kleinste afstand onderweg | Afstand tot het dier bovenaan |", "|---|---|---|---|"]
    for u in mont["uitnemen"]:
        L.append(f"| {u['kaart']} | {fmt(u['slag_mm'], 1)} | {fmt(u['kleinste_afstand_onderweg_mm'])} | "
                 f"{fmt(u['afstand_tot_dier_bovenaan_mm'])} |")
    L += ["", "## 4. Printbaarheid (Bambu Lab A1, 256 × 256 mm, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Plaat | Maat | Past | Overhang > 48° (mm²) | 45°-vlakken (mm²) | Korte bruggen < 3 mm (mm²) | Te smal (< 0,4 mm) | Lagen met kleurwissel | Wissels (schatting) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, i in prnt.items():
        ov = sum(o["oppervlak_mm2"] for o in i["overhang"])
        thin = max([d["grootste_te_smalle_stuk_mm2"] for d in i["dunne_details"]] or [0])
        L.append(f"| {name} | {' × '.join(fmt(v, 0) for v in i['maat_mm'])} | {'✅' if i['past_op_A1'] else '⚠️'} | "
                 f"{'✅ ' if ov == 0 else '⚠️ '}{fmt(ov, 0)} | {fmt(i['vlakken_rond_45_graden_mm2'], 0)} | "
                 f"{fmt(i.get('korte_bruggen_mm2', 0), 0)} | "
                 f"{'✅ geen' if thin == 0 else '⚠️ ' + fmt(thin) + ' mm²'} | "
                 f"{i['lagen_met_kleurwissel']} | {i['geschatte_wissels']} |")
    L += ["", "45°-vlakken zijn de bewust schuine lippen, afschuiningen en afloop van de geleiders: printbaar zonder "
          "support. Korte bruggen zijn de verzonken letters van het versielabel aan de achterkant (op het bed)."]
    details = [(n, o) for n, i in prnt.items() for o in i["overhang"]]
    if details:
        L += ["", "Overhangende vlakken (op het bed liggende vlakken niet meegeteld):", ""]
        for n, o in details:
            L.append(f"- {n}: {fmt(o['oppervlak_mm2'], 0)} mm² onder {fmt(o['hoek_tov_horizontaal_graden'], 0)}° "
                     f"rond {o['plek_mm']}")
    thin = [(n, d) for n, i in prnt.items() for d in i["dunne_details"]]
    if thin:
        L += ["", "Te smalle stukjes:", ""]
        for n, d in thin:
            L.append(f"- {n} / {d['kleur']}: {fmt(d['grootste_te_smalle_stuk_mm2'])} mm² bij {d['plek_mm']}")
    if slicer:
        L += ["", "### Slicer (PrusaSlicer 2.7, A1-achtig profiel)", "",
              "Bambu Studio is hier niet beschikbaar; de tijden zijn een indicatie, het gewicht is betrouwbaarder.", "",
              "| Plaat | Filament (g) | Tijd (indicatie) | Support nodig | Wissels |", "|---|---|---|---|---|"]
        for name, r in slicer.items():
            L.append(f"| {name} | {fmt(r.get('gram', 0), 1)} | {r.get('tijd', '?')} | "
                     f"{'⚠️ ja' if r.get('support') else '✅ nee'} | {r.get('wissels', '-')} |")
        L += ["", "Support is gecontroleerd met automatische support bij een drempel van 30° (Bambu-standaard): "
              "de slicer vond nergens iets om te ondersteunen.", "", "![Printplaten](../renders/printplaten.png)"]
    L += ["", "## 5. Stabiliteit", "",
          f"- Totaalgewicht ongeveer {fmt(stab['massa_g'], 0)} g ({stab['bron']}).",
          f"- Zwaartepunt (x, y, z) = {stab['zwaartepunt_mm']} mm; voet is {fmt(FOOT_D, 0)} mm diep.",
          f"- Kantelt pas voorover bij {fmt(stab['kantelhoek_voorover_graden'], 0)}° en achterover bij "
          f"{fmt(stab['kantelhoek_achterover_graden'], 0)}° scheefstand.", ""]
    (OUT / "RAPPORT.md").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    slicer_file = ROOT / "controle" / "slicer.json"
    slicer = json.loads(slicer_file.read_text()) if slicer_file.exists() else {}
    mont = check_montage()
    prnt = check_print()
    stab = check_stabiliteit(slicer)
    (OUT / "controle.json").write_text(json.dumps({"montage": mont, "print": prnt, "stabiliteit": stab},
                                                  indent=2, ensure_ascii=False) + "\n")
    draw(mont)
    printplaten(slicer)
    rapport(mont, prnt, stab, slicer)
    print((OUT / "RAPPORT.md").read_text())


if __name__ == "__main__":
    main()
