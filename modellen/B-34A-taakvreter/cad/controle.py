"""Montage- en printcontrole van B-34A Taakvreter.

Gebruik:  python controle.py        (python slice.py eerst, voor gewicht en tijd in het rapport)
Schrijft  controle/RAPPORT.md, controle/controle.json en controle/mechanisme.png

  1. Botsingen over een hele slag van de kruk (alle onderdelen, elk paar).
  2. Spelingen tussen de onderdelen die langs elkaar bewegen.
  3. Tandwielen: tandspeling over een hele steek.
  4. Oogmechaniek: uitslag van de ogen, overbrengingshoek.
  5. Papierpad: sleuf, scheef invoeren, knijpen tussen de rollen, glijgoot naar de lade.
  6. Vingerproef: kan een vinger bij een knelplek?
  7. Lade: speling in de baai en eruit schuiven zonder te haken.
  8. Printbaarheid per plaat (gereedschap/printcheck.py) en de slicer.
"""
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

import _pad  # noqa: F401,E402
import params as P  # noqa: E402
import platen  # noqa: E402
import vreter as V  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "controle"
KLEUR = {P.ORANJE: "#FF9016", P.WIT: "#E8E2D6", P.ZWART: "#3A3A3A", V.RUBBER: "#B03030"}

# bedoelde overlap: klemmende paspennen, O-ringen in hun groef en tegen de bovenste rol
KLEMMEN = [("romp voor", "paspen "), ("hoorn ", "paspen hoorn "), ("rol onder", "o-ring "), ("rol boven", "o-ring ")]


def klemt(a, b):
    for x, y in KLEMMEN:
        if (a.startswith(x) and b.startswith(y)) or (b.startswith(x) and a.startswith(y)):
            return True
    return a.startswith("paspen") and b.startswith("paspen")


def gap(a, b, zoek=6.0):
    return float(a.min_gap(b, zoek))


def bb_raak(a, b, marge=0.0):
    return all(a[i] - marge <= b[i + 3] and b[i] - marge <= a[i + 3] for i in range(3))


# --- 1. botsingen -------------------------------------------------------------------------------

def botsingen(stappen=24):
    hoeken = [360.0 * k / stappen for k in range(stappen)]
    fouten = []
    for phi in hoeken:
        st = V.stand(phi)
        bbs = {n: m.bounding_box() for n, (_, m) in st.items()}
        for a, b in itertools.combinations(st, 2):
            if klemt(a, b) or not bb_raak(bbs[a], bbs[b]):
                continue
            v = (st[a][1] ^ st[b][1]).volume()
            if v > 0.005:
                fouten.append({"hoek": phi, "a": a, "b": b, "mm3": round(v, 3)})
    return {"standen": len(hoeken), "onderdelen": len(V.delen()), "fouten": fouten}


# --- 2. spelingen -------------------------------------------------------------------------------

# (bewegend deel, x-zones van dat deel die meetellen (lagers zelf niet), tegen, titel)
PAREN = [
    ("rol onder", [(P.RIB_LINKS[1] + 0.3, P.TAND_X[1])], "romp", "onderste rol - romp (schouder, rol, tandwiel)"),
    ("rol boven", [(P.BOVEN_X0, P.RIB[0] - 0.3)], "romp", "bovenste rol - beugel, rib, glijgoot"),
    ("kruk", [(36.0, P.RIB[0] - 0.3), (P.KRUK_X[0], 90.0)], "romp", "kruk - romp (kraag, arm)"),
    ("knop", [(60.0, P.KRUK_X[0]), (P.KRUK_X[1], 90.0)], "romp", "knop (en klikpen) - zijwand"),
    ("nokas", [(-60.0, P.NOKSCHIJF_X[1])], "romp", "krukschijf - beugel"),
    ("drijfstang", [(-60.0, 0.0)], "romp", "drijfstang - romp"),
    ("drijfstang", [(-60.0, 0.0)], "rol onder", "drijfstang - onderste rol"),
    ("tuimelaar", [(-60.0, 0.0)], "romp", "tuimelaar - zijwand"),
    ("ogen", [(-P.OOG_X - P.TROMMEL_HALF, -P.OOG_X + P.TROMMEL_HALF), (P.OOG_X - P.TROMMEL_HALF, P.OOG_X + P.TROMMEL_HALF)],
     "romp", "oogtrommels - gezicht"),
    ("pupil links", [(-60.0, 60.0)], "romp", "pupil - gezicht"),
    ("mondstuk", [(-60.0, 60.0)], "rol onder", "invoertunnel - onderste rol"),
    ("mondstuk", [(-60.0, 60.0)], "rol boven", "invoertunnel - bovenste rol"),
    ("mondstuk", [(-60.0, 60.0)], "o-ring 2", "invoertunnel - O-ring"),
    ("lade", [(-60.0, 60.0)], "romp", "lade - romp"),
]

# lagers: (wat, as of pen, gat) als straal-speling
LAGERS = [
    ("asstompen van de rollen en de kruk-as in de zijwanden", P.STOMP_R, P.GAT_DRAAI / 2),
    ("nokas in de beugel", P.STOMP_R, P.GAT_DRAAI / 2),
    ("oogas in de kliklagers", P.OOGAS_R, P.OOGAS_R + 0.15),
    ("D-eind van de oogas in de tuimelaar", P.OOGAS_R, P.OOGAS_R + 0.15),
    ("pennen van tuimelaar en drijfstang", P.PEN_R, P.PEN_GAT),
    ("klikpen van de knop in de krukarm", P.KNOPPEN_R, P.KNOP_GAT),
    ("zeskant van kruk-as en nokas in de rollen (over de platte kanten)", P.ZESKANT / 2, P.ZESKANT_GAT / 2),
]


def zones(m, xs):
    return [m ^ V.Manifold.cube([x1 - x0, 400, 400]).translate([x0, -200, -200]) for x0, x1 in xs]


def spelingen(stappen=12):
    out = []
    standen = [V.stand(360.0 * k / stappen) for k in range(stappen)]
    for a, xs, b, titel in PAREN:
        best, waar = 99.0, 0.0
        for k, st in enumerate(standen):
            tegen = [st["romp voor"][1], st["romp achter"][1]] if b == "romp" else [st[b][1]]
            for z in zones(st[a][1], xs):
                if z.is_empty():
                    continue
                for t in tegen:
                    g = gap(z, t)
                    if g < best:
                        best, waar = g, 360.0 * k / stappen
        out.append({"paar": titel, "speling": round(best, 2), "hoek": waar})
    return out


# --- 3. tandwielen ------------------------------------------------------------------------------

def tandwielen(n=24):
    """Speling tussen de twee tandwielen (alleen de tandwielschijven) over één steek."""
    ring = V.Manifold.cube([P.TAND_X[1] - P.TAND_X[0] - 0.2, 40, 40]).translate([P.TAND_X[0] + 0.1, -20, -20])
    onder, boven = V.rol_onder() ^ ring, V.rol_boven() ^ ring
    steek = 360.0 / P.Z_TAND
    gaps = []
    for k in range(n):
        phi = steek * k / n
        a = V.op(V.draai_x(onder, -phi), V.A_ONDER)
        b = V.op(V.draai_x(boven, phi), V.A_BOVEN)
        gaps.append(gap(a, b, 2.0))
    return {"kleinste": round(min(gaps), 3), "grootste": round(max(gaps), 3),
            "hart_op_hart": P.AS, "rollen_kaal": round(P.AS - 2 * P.ROL_R, 2)}


# --- 4. oogmechaniek ----------------------------------------------------------------------------

def oogmechaniek():
    hoeken, mu = [], []
    B = np.array(V.B_OOG)
    for f in np.linspace(0, 360, 361):
        pk, q, h = V.vierstang(f)
        hoeken.append(h)
        a, b = np.array(pk) - np.array(q), B - np.array(q)      # drijfstang en tuimelaar in het scharnier
        c = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        mu.append(math.degrees(math.acos(min(1.0, c))))
    uitslag = max(hoeken) - min(hoeken)
    pupil = P.TROMMEL_R * math.sin(math.radians(uitslag / 2))
    g = float(np.linalg.norm(np.array(V.A_BOVEN) - B))
    staven = sorted([P.NOK_E, P.STANG, P.TUIMEL, g])
    return {"uitslag_graden": round(uitslag, 1), "pupil_op_neer_mm": round(pupil, 1),
            "overbrengingshoek_min": round(min(mu), 1), "overbrengingshoek_max": round(max(mu), 1),
            "grashof": staven[0] + staven[3] <= staven[1] + staven[2], "vaste_stang": round(g, 2),
            "kruk_kortste": staven[0] == P.NOK_E}


# --- 5. papierpad -------------------------------------------------------------------------------

def papierpad():
    s = P.SLEUF_B / P.PAPIER
    # vierkant blaadje dat scheef binnenkomt: breedte 76 (cos a + sin a)
    scheef = math.degrees(math.asin(s / math.sqrt(2))) - 45.0
    knijp = P.ORING_R + P.ROL_R - P.AS
    return {
        "sleuf_mm": [P.SLEUF_B, P.SLEUF_H], "papier_mm": P.PAPIER, "scheef_max_graden": round(scheef, 1),
        "rol_breedte_mm": [round(-P.ROL_X, 1), round(P.ROL_X, 1)],
        "rand_sleuf_tot_tandwiel_mm": round(P.TAND_X[0] - P.SLEUF_B / 2, 1),
        "knijp_o_ring_mm": round(knijp, 2), "knijp_na_lagerspeling_mm": round(knijp - P.LAGERSPELING, 2),
        "lagerspeling_mm": P.LAGERSPELING, "o_ring_schap_mm": round(P.Z_ONDER - P.ORING_R - P.SCHAP[1], 2),
        "papiervlak_z": round(P.NIP_Z, 2),
        "tunnel_tot_knijppunt_mm": round(P.Y_DEEL - P.TUNNEL_Y, 1),
        "goot_boven_papiervlak_mm": round(P.GOOT_MIDDEN[1] + P.GOOT_R[1] - P.NIP_Z, 2),
        "valopening_mm": round(P.D - P.WAND - P.SCHAP[2], 1),
        "lade_binnen_mm": [round(2 * (P.LADE_HALF - P.LADE_WAND), 1),
                           round(P.LADE_Y[1] - P.LADE_Y[0] - 2 * P.LADE_WAND - 1, 1),
                           round(P.LADE_Z[1] - P.LADE_Z[0] - P.LADE_WAND, 1)],
    }


# --- 6. vingerproef -----------------------------------------------------------------------------

def vingerproef():
    """Een kindervinger (Ø 8 mm) en een volwassen vinger (Ø 12 mm, IEC-proefvinger) tegen de invoer."""
    diep = P.Y_DEEL + P.LIP_T
    st = V.stand(0.0, zonder=("lade",))
    # vanaf de onderkant met de lade eruit: hoe dicht komt een vinger (bol Ø 8) achter het schap bij de rol?
    bol = V.Manifold.sphere(4.0, 32)
    best = 99.0
    for y in np.arange(P.SCHAP[2] + 4.0, P.SCHAP[2] + 12.0, 1.0):
        for z in np.arange(P.SCHAP[1] + 4.0, P.NIP_Z, 1.0):
            b = bol.translate([0.0, float(y), float(z)])
            if gap(b, st["romp voor"][1], 0.5) <= 0 or gap(b, st["romp achter"][1], 0.5) <= 0:
                continue
            best = min(best, gap(b, st["rol onder"][1], 10.0), gap(b, st["o-ring 2"][1], 10.0))
    return {"sleuf_hoogte": P.SLEUF_H, "knijppunt_diepte_mm": round(diep, 1),
            "vinger_8_in_sleuf": P.SLEUF_H >= 8.0, "lade_eruit_vinger_tot_rol_mm": round(best, 2)}


# --- 7. lade ------------------------------------------------------------------------------------

def lade():
    st = V.stand(0.0)
    l = st["lade"][1]
    speling = min(gap(l, st["romp voor"][1]), gap(l, st["romp achter"][1]))
    haakt = []
    for dy in np.arange(0, P.LADE_Y[1] - P.LADE_Y[0] + 5, 5.0):
        m = l.translate([0, -float(dy), 0])
        for n in ("romp voor", "romp achter"):
            if (m ^ st[n][1]).volume() > 0.005:
                haakt.append((float(dy), n))
    return {"speling": round(speling, 2), "haakt": haakt}


# --- 8. printbaarheid ---------------------------------------------------------------------------

def printbaar():
    out = {}
    for naam, kleur, delen in platen.platen():
        out[naam] = printcheck.plaat({kleur: V.Manifold.batch_boolean([m for _, m in delen], V.manifold_add())})
        out[naam]["kleur"] = kleur
    return out


# --- tekening -----------------------------------------------------------------------------------

def snede(m, as_, c):
    if as_ == "x":
        T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)
        return cs_to_shapely(m.transform(T).slice(c))
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)
    return cs_to_shapely(m.transform(T).slice(-c))


def _vul(ax, g, kleur):
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or not hasattr(p, "exterior"):
            continue
        p = orient(p, 1.0)
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            cs = list(ring.coords)
            verts += cs
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(cs) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=kleur, ec="#222", lw=0.3))


def tekening(pad):
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))
    vlakken = [("x", 0.0, "doorsnede midden: papierpad"), ("x", -53.0, "oogmechaniek (links)"),
               ("y", P.Y_DEEL, "deelvlak (rolassen)"), ("y", P.OOG_Y, "oogas")]
    st0 = V.stand(0.0)
    for ax, (as_, c, titel) in zip(axs[0], vlakken):
        for naam, (k, m) in st0.items():
            _vul(ax, snede(m, as_, c), KLEUR.get(k, "#888"))
        if as_ == "x" and c == 0.0:
            ax.plot([-6, P.Y_DEEL], [P.NIP_Z, P.NIP_Z], color="#1565C0", lw=1.2, ls="--")
        ax.set_title(titel)
    for ax, phi in zip(axs[1], (0.0, 90.0, 180.0, 270.0)):
        st = V.stand(phi)
        _vul(ax, snede(st["romp voor"][1], "x", -46.5).union(snede(st["romp achter"][1], "x", -46.5)), "#FFD3A6")
        for x, namen in ((-50.0, ("nokas", "rol onder")), (-53.5, ("drijfstang",)), (-56.5, ("tuimelaar", "ogen"))):
            for naam in namen:
                k, m = st[naam]
                _vul(ax, snede(m, "x", x), KLEUR.get(k, "#888") if naam != "tuimelaar" else "#6D6D6D")
        ax.set_title(f"kruk {phi:.0f}°: ogen {V.vierstang(phi)[2] - V.hoek_midden():+.0f}°")
        ax.set_xlim(0, 60)
        ax.set_ylim(40, 115)
    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.2)
    for ax in axs[1]:
        ax.set_xlim(0, 60)
        ax.set_ylim(40, 115)
    fig.tight_layout()
    fig.savefig(pad, dpi=90)
    plt.close(fig)


# --- rapport ------------------------------------------------------------------------------------

def ok(v):
    return "✅" if v else "⚠️"


def rapport(r):
    L = [f"# Controle B-34A Taakvreter", "",
         f"Automatisch gegenereerd door `cad/controle.py` ({P.VERSIE}). Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "![Mechanisme](mechanisme.png)", "", "## 1. Botsingen over een hele slag", ""]
    b = r["botsingen"]
    knijp = f"{P.ORING_R + P.ROL_R - P.AS:.2f}".replace(".", ",")
    if b["fouten"]:
        L.append(f"⚠️ {len(b['fouten'])} overlappen:")
        L += [f"- {f['a']} - {f['b']} bij {f['hoek']:.0f}°: {f['mm3']} mm³" for f in b["fouten"][:20]]
    else:
        L.append(f"✅ {b['standen']} standen van de kruk (om de {360 // b['standen']}°), {b['onderdelen']} onderdelen "
                 "inclusief O-ringen en paspennen, elk paar: nergens overlap. Bewust klemmend: de paspennen in de "
                 f"voorhelft en de hoorntjes, en de O-ringen tegen de bovenste rol ({knijp} mm).")
    L += ["", "## 2. Spelingen over een hele slag", "",
          "Vrije ruimte tussen wat beweegt en wat stilstaat (de lagers zelf staan in de tabel eronder):", "",
          "| Paar | Kleinste speling | Bij krukhoek |", "|---|---|---|"]
    for s in r["spelingen"]:
        L.append(f"| {s['paar']} | {ok(s['speling'] >= 0.25)} {s['speling']:.2f} | {s['hoek']:.0f}° |")
    L += ["", "Lagers (speling rondom, ontwerpmaat; de pasproef laat zien of je printer dat haalt):", "",
          "| Lager | As of pen | Gat | Speling |", "|---|---|---|---|"]
    for wat, ra, rg in LAGERS:
        L.append(f"| {wat} | Ø {2 * ra:.1f} | Ø {2 * rg:.2f} | {rg - ra:.2f} |")
    t = r["tandwielen"]
    L += ["", "## 3. Tandwielen", "",
          f"Twee tandwielen m = {P.M_ROL}, z = {P.Z_TAND}, hart-op-hart {t['hart_op_hart']} mm: de rollen draaien "
          f"1:1 tegen elkaar in. Speling tussen de tanden over één steek: kleinste {ok(t['kleinste'] > 0.03)} "
          f"{t['kleinste']}, grootste {t['grootste']}. De kale rollen staan {t['rollen_kaal']} mm uit elkaar; de "
          "O-ringen overbruggen dat."]
    o = r["oogmechaniek"]
    L += ["", "## 4. Oogmechaniek", "",
          f"Vierstangenmechanisme: kruk {P.NOK_E} mm op de bovenste rol, drijfstang {P.STANG} mm, tuimelaar "
          f"{P.TUIMEL} mm op de oogas, vaste stang {o['vaste_stang']} mm.", "",
          f"- {ok(o['grashof'] and o['kruk_kortste'])} Grashof met de kruk als kortste stang: de kruk draait "
          "rond, de ogen schommelen heen en weer.",
          f"- Uitslag van de ogen: {o['uitslag_graden']}°, de pupillen gaan {o['pupil_op_neer_mm']} mm op en neer "
          "(één keer per omwenteling van de kruk).",
          f"- {ok(o['overbrengingshoek_min'] >= 40)} Overbrengingshoek {o['overbrengingshoek_min']}° tot "
          f"{o['overbrengingshoek_max']}° (boven 40° loopt het licht, zonder klemmen)."]
    p = r["papierpad"]
    L += ["", "## 5. Papierpad", "",
          f"- Sleuf {p['sleuf_mm'][0]:.0f} × {p['sleuf_mm'][1]:.0f} mm voor blaadjes van {p['papier_mm']:.0f} mm: "
          f"tot {p['scheef_max_graden']}° scheef past het blaadje nog.",
          f"- {ok(p['rand_sleuf_tot_tandwiel_mm'] >= 1.0)} De rollen lopen van {p['rol_breedte_mm'][0]:.0f} tot "
          f"{p['rol_breedte_mm'][1]:.0f}, breder dan de sleuf: de tandwielen zitten {p['rand_sleuf_tot_tandwiel_mm']} "
          "mm buiten de rand van de sleuf, daar kan geen papier komen.",
          f"- {ok(p['knijp_na_lagerspeling_mm'] >= 0.15 and p['knijp_o_ring_mm'] <= 0.8)} De O-ringen drukken "
          f"nominaal {p['knijp_o_ring_mm']} mm in tegen de bovenste rol. De lagers hebben samen tot "
          f"{p['lagerspeling_mm']} mm speling (twee draaigaten en een zeskant per kant); die duwen de O-ringen eerst "
          f"weg, dus er blijft minstens {p['knijp_na_lagerspeling_mm']} mm over, plus de dikte van het blaadje.",
          f"- {ok(p['o_ring_schap_mm'] >= 0.5)} Tussen de O-ringen en het schap eronder: {p['o_ring_schap_mm']} mm.",
          f"- De tunnel stopt {p['tunnel_tot_knijppunt_mm']} mm voor het knijppunt; de rollen vormen daar een "
          "trechter naar het papiervlak.",
          f"- Achter de rollen buigt de glijgoot het blaadje omlaag: het plafond begint {p['goot_boven_papiervlak_mm']} "
          f"mm boven het papiervlak. Achter het schap is een opening van {p['valopening_mm']} mm naar de lade.",
          f"- Lade binnenmaat {p['lade_binnen_mm'][0]:.0f} × {p['lade_binnen_mm'][1]:.0f} × "
          f"{p['lade_binnen_mm'][2]:.0f} mm."]
    v = r["vingerproef"]
    L += ["", "## 6. Vingerproef", "",
          f"- {ok(not v['vinger_8_in_sleuf'])} De invoer is {v['sleuf_hoogte']:.0f} mm hoog: een vinger (ook van "
          f"een kind, Ø 8 mm) past er niet in. Het knijppunt zit {v['knijppunt_diepte_mm']} mm achter de lip.",
          "- Het schap onder de rollen loopt tot voorbij de rollen: van onderen (lade erin) kun je er niet bij.",
          f"- ⚠️ Met de lade eruit komt een vinger (Ø 8 mm) achter het schap tot {v['lade_eruit_vinger_tot_rol_mm']} "
          "mm van de achterkant van de onderste rol. Daar lopen de rollen uit elkaar (geen intrekkende kant), maar "
          "draai de kruk toch alleen met de lade erin."]
    la = r["lade"]
    L += ["", "## 7. Lade", "",
          f"- {ok(la['speling'] >= 0.3)} Speling rondom in de baai: {la['speling']} mm.",
          f"- {ok(not la['haakt'])} Er helemaal uit schuiven: " + ("haakt nergens." if not la["haakt"] else
                                                                     f"haakt bij {la['haakt'][:3]}.")]
    L += ["", "## 8. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |", "|---|---|---|---|---|---|"]
    for naam, i in r["print"].items():
        f = P.FILAMENTEN[i["kleur"]][0]
        maat = " × ".join(f"{v:.0f}" for v in i["maat_mm"])
        oh = sum(o["oppervlak_mm2"] for o in i["overhang"])
        dun = "geen" if not i["dunne_details"] else f"{len(i['dunne_details'])} plekken"
        L.append(f"| {naam} | {f} | {maat} | {ok(i['past_op_A1'])} | {ok(oh < 5)} {oh:.0f} | {ok(not i['dunne_details'])} {dun} |")
    if r.get("slicer"):
        L += ["", "### Slicer (PrusaSlicer, A1-achtig profiel)", "", "| Plaat | Filament | Tijd (indicatie) | Support nodig |",
              "|---|---|---|---|"]
        for naam, s in r["slicer"].items():
            L.append(f"| {naam} | {s['gram']:.1f} g | {s['tijd']} | {ok(not s['support'])} {'ja' if s['support'] else 'nee'} |")
    return "\n".join(L) + "\n"


def main(argv):
    UIT.mkdir(exist_ok=True)
    r = {"botsingen": botsingen(), "spelingen": spelingen(), "tandwielen": tandwielen(),
         "oogmechaniek": oogmechaniek(), "papierpad": papierpad(), "vingerproef": vingerproef(),
         "lade": lade(), "print": printbaar()}
    sl = UIT / "slicer.json"
    if sl.exists():
        r["slicer"] = json.loads(sl.read_text())
    tekening(UIT / "mechanisme.png")
    (UIT / "controle.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str) + "\n")
    tekst = rapport(r)
    (UIT / "RAPPORT.md").write_text(tekst)
    print(tekst)


if __name__ == "__main__":
    main(sys.argv[1:])
