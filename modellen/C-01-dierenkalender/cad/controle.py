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
from shapely.geometry import box
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kalender as K  # noqa: E402
from dieren import ALLE, EIGEN, FANART, bouwbaar  # noqa: E402
import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely, rot_x, to_trimesh, translate, union  # noqa: E402
from params import *  # noqa: E402,F403
from talen import TALEN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
BED = (256.0, 256.0)
NOZZLE = 0.4
LAYER = 0.2
PLA = 1.24  # g/cm3

KLEUR = {code: ("#F4F1EA" if code == "11100" else "#222222" if code == "11101" else v[2])
         for code, v in FILAMENTEN.items()}


def T(m, M):
    return m.transform(np.asarray(M, dtype=float)[:3, :])


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


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

def check_montage(sleutel="vos", taal="nl"):
    res = {"botsingen": [], "spelingen": {}, "uitnemen": []}
    t = TALEN[taal]
    parts, channels = K.assembly(sleutel, taal)
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
    full, _ = K.assembly(sleutel, taal, full_bins=True)
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
    fl, _ = K.front(sleutel, taal)
    fl = union(list(fl.values()))
    card_y0 = WINDOW_Y0 + RIM_H
    zl = PLATE_T + CHANNEL
    sp = {}
    for (ca, cb), kind, label in zip(channels, K.KINDS, t["vandaag"]):
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
                                   [t["dagen"], TIENTALLEN, EENHEDEN, t["maanden"]]):
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

    res["uitnemen"] = check_uitnemen(sleutel, taal)
    return res


def check_uitnemen(sleutel, taal="nl"):
    """Kaartje 0,05 mm van de plaat omhoog schuiven tot het vrij is; afstand tot het dier bovenaan."""
    fl, channels = K.front(sleutel, taal)
    fl = union(list(fl.values()))
    card_y0 = WINDOW_Y0 + RIM_H
    top_y = card_y0 + CARD_H - CARD_EXPOSED
    relief = fl ^ box_m(-120, -30, PLATE_T + 0.01, 120, 230, 20)
    animal = relief ^ box_m(-120, top_y + 2, -1, 120, 230, 30)
    out = []
    for (ca, cb), kind, label in zip(channels, K.KINDS, TALEN[taal]["vandaag"]):
        c = union(list(K.card(label, kind).values()))
        c = T(c, translate((ca + cb) / 2, card_y0, PLATE_T + 0.05))
        stroke_len = top_y - card_y0 + 0.5
        worst, at = 99.0, 0.0
        for s in np.linspace(1.0, stroke_len, 30):  # start 1 mm boven de draagrand
            g = gap(c.translate([0, s, 0]), relief, 5.0)
            if g < worst:
                worst, at = g, s
        top = c.translate([0, stroke_len, 0])
        out.append({"kaart": f"{kind} ({label})", "slag_mm": round(stroke_len, 1),
                    "kleinste_afstand_onderweg_mm": round(worst, 3), "bij_slag_mm": round(at, 1),
                    "afstand_tot_dier_bovenaan_mm": round(gap(top, animal, 30.0), 2)})
    return out


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
    """Printbaarheid per printplaat (alle kleuren van één onderdeel samen), zie gereedschap/printcheck.py."""
    groups = {}
    for f in sorted(p for map_ in ("gedeeld", "varianten", "fanart") for p in (ROOT / map_).rglob("*.stl")):
        key = str(f.parent.relative_to(ROOT) / f.stem.split("__")[0])
        groups.setdefault(key, []).append(f)
    return {name: printcheck.plaat({(f.stem.split("__")[1] if "__" in f.stem else "enkel"): load_manifold(f)[1]
                                    for f in files})
            for name, files in groups.items()}


# ---------------------------------------------------------------------------
# 5: stabiliteit
# ---------------------------------------------------------------------------

def check_stabiliteit(slicer):
    """Zwaartepunt en kantelhoeken per dier (vulgraad uit de slicer als die er is, anders een aanname)."""
    fill = {"voet": 0.45, "front": 0.85, "kaart": 1.0}
    bron = "aanname"
    for key, plate in (("voet", "gedeeld/voet/voet"), ("front", "varianten/01-vos/nl/front_vos"),
                       ("kaart", "gedeeld/kaartjes/nl/maanden")):
        r = slicer.get(plate)
        if r and r.get("gram") and r.get("volume_cm3"):
            fill[key] = r["gram"] / (r["volume_cm3"] * PLA)
            bron = "slicer"
    out = {}
    for d in bouwbaar(ALLE):
        parts, _ = K.assembly(d.sleutel, "nl")
        tot_m, mom = 0.0, np.zeros(3)
        for n, c, m in parts:
            key = "voet" if n == "voet" else "front" if n == "front" else "kaart"
            tm = to_trimesh(m)
            mass = tm.volume / 1000 * PLA * fill[key]
            tot_m += mass
            mom += mass * tm.center_mass
        com = mom / tot_m
        out[f"{d.nr:02d} {d.naam['nl']}"] = {"massa_g": round(tot_m, 0), "zwaartepunt_mm": [round(float(v), 1) for v in com],
                          "kantelhoek_voorover_graden": round(math.degrees(math.atan2(com[1], com[2])), 1),
                          "kantelhoek_achterover_graden": round(math.degrees(math.atan2(FOOT_D - com[1], com[2])), 1)}
    return {"per_dier": out, "vulgraad_gebruikt": {k: round(v, 2) for k, v in fill.items()}, "bron": bron}


# ---------------------------------------------------------------------------
# Doorsnedetekening
# ---------------------------------------------------------------------------

def draw(res):
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1.2, 1])
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    parts, channels = K.assembly("vos", "nl")
    vandaag = TALEN["nl"]["vandaag"]
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
    fl, _ = K.front("vos", "nl")
    card_y0 = WINDOW_Y0 + RIM_H
    y = card_y0 + 8.0
    for c, m in fl.items():
        fill(ax, section(m, "y", y), KLEUR[c])
    for (ca, cb), kind, label in zip(channels[:2], K.KINDS, vandaag):
        mm = T(union(list(K.card(label, kind).values())), translate((ca + cb) / 2, card_y0, PLATE_T + 0.002))
        fill(ax, section(mm, "y", y), "#F4F1EA", z=3)
    cb = channels[0][1]
    s = res["spelingen"]["kanalen"][f"kanaal dag ({vandaag[0]})"]
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
    fill(ax, section(front, "x", 0.0), KLEUR[ORANJE], z=2)
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
    full, _ = K.assembly("vos", "nl", full_bins=True)
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
# Printplaten en fronten: rechtstreeks uit de miniaturen in de Bambu-projecten
# ---------------------------------------------------------------------------

def _platen(pad):
    """[(plaatnaam, PNG-bytes)] uit een Bambu-project."""
    import xml.etree.ElementTree as ET
    import zipfile
    z = zipfile.ZipFile(pad)
    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    out = []
    for k, pl in enumerate(cfg.findall("plate"), start=1):
        naam = pl.find("metadata[@key='plater_name']").get("value")
        out.append((naam, z.read(f"Metadata/plate_{k}.png")))
    return out


def _png(ax, data):
    import io
    ax.imshow(plt.imread(io.BytesIO(data)), extent=(0, 256, 0, 256))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")


SLICER_SLEUTEL = {"Voet": "gedeeld/voet/voet", "Kaartjes maanden": "gedeeld/kaartjes/nl/maanden",
                  "Kaartjes dagen": "gedeeld/kaartjes/nl/dagen", "Kaartjes cijfers": "gedeeld/kaartjes/cijfers"}


def _onderschrift(r):
    if not r:
        return ""
    s = f"{fmt(r['gram'], 0)} g · {r['tijd']}"
    return s + (f" · {r['wissels']} wissels" if r.get("wissels") else "")


def printplaten(slicer):
    """De vijf platen van het vos-project plus de pasproef, zoals ze in de 3MF-bestanden staan."""
    from bouw import GEDEELD, bestandsnaam, variant_map
    from dieren import PER_SLEUTEL
    vos = PER_SLEUTEL["vos"]
    platen = [("1 " + n, d) for n, d in _platen(GEDEELD / "pasproef" / "nl" / "pasproef_nl.3mf")]
    platen += [(f"{i + 2} {n}", d) for i, (n, d) in
               enumerate(_platen(variant_map(vos, "nl") / f"{bestandsnaam(vos, 'nl')}.3mf"))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.8))
    for ax, (naam, data) in zip(axs.flat, platen):
        _png(ax, data)
        sleutel = naam.split(" ", 1)[1]
        if sleutel.startswith("Front"):
            r = slicer.get("varianten/01-vos/nl/front_vos")
        elif sleutel == "Pasproef":
            delen = [slicer.get(k) for k in ("gedeeld/pasproef/kanalen", "gedeeld/pasproef/sleuven",
                                             "gedeeld/pasproef/nl/proefkaartjes")]
            r = {"gram": sum(d["gram"] for d in delen if d), "tijd": "3 delen", "wissels": 0} if all(delen) else None
        else:
            r = slicer.get(SLICER_SLEUTEL.get(sleutel, ""))
        ax.set_title(f"{naam}\n{_onderschrift(r)}", fontsize=11, loc="left")
    fig.suptitle("Printplaten in het Bambu-project (vos, NL) en de pasproef. Stippellijn = spoeltoren. "
                 "Gewicht en tijd: PrusaSlicer met A1-achtig profiel (tijd is een indicatie)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ROOT / "renders" / "printplaten.png", dpi=100)
    plt.close(fig)


def fronten(slicer, dieren, pad):
    """Plaat 1 (het front) van elk project, in het Nederlands."""
    from bouw import bestandsnaam, variant_map
    n = len(dieren)
    cols = n if n <= 5 else (n + 1) // 2   # 7 dieren: 4 + 3
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.6 * rows), squeeze=False)
    for ax, d in zip(axs.flat, dieren):
        naam, data = _platen(variant_map(d, "nl") / f"{bestandsnaam(d, 'nl')}.3mf")[0]
        _png(ax, data)
        r = slicer.get(str((variant_map(d, "nl") / f"front_{d.naam['nl']}").relative_to(ROOT)))
        ax.set_title(f"{d.nr:02d} {d.naam['nl']}{' (fan-art)' if d.fanart else ''}\n{_onderschrift(r)}",
                     fontsize=9, loc="left")
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(pad, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def rapport(mont, uitnemen, prnt, stab, slicer, projecten):
    L = ["# Controle C-01 dierenkalender", "",
         "Automatisch gegenereerd door `cad/controle.py` uit de STL's, de Bambu-projecten en de montage. "
         "Maten in mm. ✅ = in orde, ⚠️ = bekijken.", ""]
    ok = all(b["ok"] for b in mont["botsingen"])
    L += ["## 1. Botsingen in gemonteerde stand (vos; voet, geleiders en kaartjes zijn voor elk dier gelijk)", "",
          f"{'✅' if ok else '⚠️'} {len(mont['botsingen'])} combinaties gecontroleerd; "
          f"grootste overlap {fmt(max(b['overlap_mm3'] for b in mont['botsingen']), 4)} mm³.", ""]
    for b in [b for b in mont["botsingen"] if not b["ok"]]:
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
          "## 3. Uitneemroute per variant", "",
          "Elk kaartje 0,05 mm van de plaat en vanaf 1 mm boven de draagrand omhoog geschoven tot de onderrand "
          "boven de geleiders is. Onderweg is de kleinste afstand de zijspeling langs de ribben; bovenaan telt de "
          "afstand tot pootjes, hoefjes of kin.", "",
          "| Variant | Kleinste afstand onderweg | Kleinste afstand tot het dier bovenaan |", "|---|---|---|"]
    for naam, rij in uitnemen.items():
        a = min(u["kleinste_afstand_onderweg_mm"] for u in rij)
        b = min(u["afstand_tot_dier_bovenaan_mm"] for u in rij)
        L.append(f"| {naam} | {'✅' if a > 0.1 else '⚠️'} {fmt(a)} | {'✅' if b >= 3 else '⚠️'} {fmt(b)} |")
    L += ["", "## 4. Printbaarheid (Bambu Lab A1, 256 × 256 mm, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Onderdeel | Maat | Past | Overhang > 48° (mm²) | 45°-vlakken (mm²) | Korte bruggen < 3 mm (mm²) | "
          "Te smal (< 0,4 mm) | Lagen met kleurwissel |", "|---|---|---|---|---|---|---|---|"]
    for name, i in prnt.items():
        ov = sum(o["oppervlak_mm2"] for o in i["overhang"])
        thin = max([d["grootste_te_smalle_stuk_mm2"] for d in i["dunne_details"]] or [0])
        L.append(f"| {name} | {' × '.join(fmt(v, 0) for v in i['maat_mm'])} | {'✅' if i['past_op_A1'] else '⚠️'} | "
                 f"{'✅ ' if ov == 0 else '⚠️ '}{fmt(ov, 0)} | {fmt(i['vlakken_rond_45_graden_mm2'], 0)} | "
                 f"{fmt(i.get('korte_bruggen_mm2', 0), 0)} | "
                 f"{'✅ geen' if thin == 0 else '⚠️ ' + fmt(thin) + ' mm²'} | {i['lagen_met_kleurwissel']} |")
    L += ["", "45°-vlakken zijn de bewust schuine lippen, afschuiningen en afloop van de geleiders: printbaar zonder "
          "support. Korte bruggen zijn de verzonken letters van het versielabel aan de achterkant (op het bed)."]
    for n, o in [(n, o) for n, i in prnt.items() for o in i["overhang"]]:
        L.append(f"- {n}: {fmt(o['oppervlak_mm2'], 0)} mm² onder {fmt(o['hoek_tov_horizontaal_graden'], 0)}° "
                 f"rond {o['plek_mm']}")
    for n, d in [(n, d) for n, i in prnt.items() for d in i["dunne_details"]]:
        L.append(f"- te smal: {n} / {d['kleur']}: {fmt(d['grootste_te_smalle_stuk_mm2'])} mm² bij {d['plek_mm']}")
    if slicer:
        L += ["", "### Slicer (PrusaSlicer 2.7, A1-achtig profiel)", "",
              "Bambu Studio draait hier niet; het gewicht is betrouwbaar, de tijd en het aantal wissels zijn een "
              "indicatie. Engelse fronten verschillen alleen in de labels en zijn niet apart geslicet.", "",
              "| Onderdeel | Filament (g) | Tijd (indicatie) | Support nodig | Wissels |", "|---|---|---|---|---|"]
        for name, r in slicer.items():
            L.append(f"| {name} | {fmt(r.get('gram', 0), 1)} | {r.get('tijd', '?')} | "
                     f"{'⚠️ ja' if r.get('support') else '✅ nee'} | {r.get('wissels', '-')} |")
        L += ["", "Support is gecontroleerd met automatische support bij een drempel van 30° (Bambu-standaard).", "",
              "![Printplaten](../renders/printplaten.png)", "", "![Fronten](../renders/fronten.png)"]
    L += ["", "## 5. Bambu Studio-projecten", "",
          "Elk .3mf-bestand is gecontroleerd op de regels van de Bambu Studio-lezer (bbs_3mf.cpp): verplichte "
          "bestanden, id's van objecten en delen, filamentslots, lijsten per filament, spoelmatrix, en of elk "
          "object volledig op zijn eigen plaat ligt zonder de spoeltoren te raken. Niet getest in Bambu Studio zelf.", ""]
    for pad, fouten in projecten:
        L.append(f"- {'✅' if not fouten else '⚠️'} `{pad}`" + ("" if not fouten else ": " + "; ".join(fouten)))
    L += ["", "## 6. Stabiliteit", "", f"Vulgraad: {stab['bron']} {stab['vulgraad_gebruikt']}.", "",
          "| Variant | Gewicht (g) | Zwaartepunt hoogte (mm) | Kantelt voorover bij | Kantelt achterover bij |",
          "|---|---|---|---|---|"]
    for naam, r in stab["per_dier"].items():
        L.append(f"| {naam} | {fmt(r['massa_g'], 0)} | {fmt(r['zwaartepunt_mm'][2], 0)} | "
                 f"{fmt(r['kantelhoek_voorover_graden'], 0)}° | {fmt(r['kantelhoek_achterover_graden'], 0)}° |")
    (OUT / "RAPPORT.md").write_text("\n".join(L) + "\n")


def main():
    from gereedschap.bambu.valideer import valideer
    OUT.mkdir(exist_ok=True)
    slicer_file = ROOT / "controle" / "slicer.json"
    slicer = json.loads(slicer_file.read_text()) if slicer_file.exists() else {}
    mont = check_montage()
    uitnemen = {f"{d.nr:02d} {d.naam['nl']}": check_uitnemen(d.sleutel, "nl") for d in bouwbaar(ALLE)}
    prnt = check_print()
    stab = check_stabiliteit(slicer)
    projecten = [(str(p.relative_to(ROOT)), valideer(p)[0]) for p in sorted(ROOT.rglob("*.3mf"))]
    (OUT / "controle.json").write_text(json.dumps({"montage": mont, "uitnemen": uitnemen, "print": prnt,
                                                   "stabiliteit": stab, "projecten": projecten},
                                                  indent=2, ensure_ascii=False) + "\n")
    draw(mont)
    printplaten(slicer)
    fronten(slicer, EIGEN, ROOT / "renders" / "fronten.png")
    fronten(slicer, FANART, ROOT / "fanart" / "fronten.png")
    rapport(mont, uitnemen, prnt, stab, slicer, projecten)
    print((OUT / "RAPPORT.md").read_text()[:3000])


if __name__ == "__main__":
    main()
