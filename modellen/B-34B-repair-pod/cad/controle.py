"""Mechanisme-, montage- en printcontrole voor B-34B Repair Pod, op de echte geometrie.

Gebruik:  python controle.py          schrijft controle/RAPPORT.md, controle/controle.json en
                                       controle/mechanisme.png (vooraanzichten over de slag)

Wat wordt gecontroleerd
  1. Botsingen: alle 38 onderdelen (pennen meegeteld) in 33 standen van dicht tot open, elk paar.
  2. Spelingen over de hele slag: bak-dak, bak-voet, arm-voet, arm-bak, schuif-voet, schuif-dak.
  3. Tandsector en heugel: nergens overlap, en overal in ingrijping (kleine speling, nooit los).
  4. Eindstops: knop op het dak (dicht), binnenarm tegen de nok (open).
  5. Horizontaal: het parallellogram houdt de bak exact recht; de pen-speling geeft een kleine scheefstand.
  6. Kracht op de knop en kantelen, met een vol en een leeg bakje.
  7. Printbaarheid per plaat (gereedschap/printcheck.py).
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
from shapely import affinity  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

import _pad  # noqa: F401,E402
import params as P  # noqa: E402
import platen  # noqa: E402
import pod  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely, to_trimesh  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
STANDEN = np.linspace(P.THETA_OPEN, P.THETA_DICHT, 33)
PLA = 1.24
SCHROEVEN_G = 150.0        # zware lading in één bakje (b.v. 150 g schroefjes)


def gap(a, b, zoek=10.0):
    return float(a.min_gap(b, zoek))


def bb(m):
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


# 1 ---------------------------------------------------------------------------------------------
def botsingen():
    erg = {}
    for th in STANDEN:
        items = list(pod.stand(th).items())
        boxes = {n: bb(m) for n, (_, m) in items}
        for (na, (_, a)), (nb, (_, b)) in itertools.combinations(items, 2):
            (la, ha), (lb, hb) = boxes[na], boxes[nb]
            if np.any(la > hb + 0.01) or np.any(lb > ha + 0.01):
                continue
            v = (a ^ b).volume()
            if v > 1e-3 and v > erg.get((na, nb), (0, 0))[0]:
                erg[(na, nb)] = (round(v, 3), round(float(th), 1))
    return {"standen": len(STANDEN), "paren_met_overlap": [{"a": a, "b": b, "mm3": v, "hoek": t}
                                                         for (a, b), (v, t) in erg.items()]}


# 2-4 -------------------------------------------------------------------------------------------
PAREN = [
    ("bak rechts", "dak", "bak - dak"),
    ("bak rechts", "voet", "bak - voet (rug en flank)"),
    ("bak rechts", "bak links", "bak - bak"),
    ("binnenarm voor rechts", "voet", "binnenarm - voet"),
    ("buitenarm voor rechts", "bak rechts", "buitenarm - bak"),
    ("binnenarm voor rechts", "buitenarm voor rechts", "binnenarm - buitenarm"),
    ("schuif", "voet", "schuif - voet (zwaluwstaart)"),
]


def spelingen():
    """Kleinste speling per paar over de slag; de open stand telt niet voor arm-voet (daar raakt de nok, zie 4)."""
    out = {}
    for a, b, label in PAREN:
        laagst, waar = 99.0, None
        for th in (STANDEN[1:] if "arm" in a and b == "voet" else STANDEN):
            st = pod.stand(th)
            g = gap(st[a][1], st[b][1])
            if g < laagst:
                laagst, waar = g, float(th)
        out[label] = {"kleinste_mm": round(laagst, 2), "bij_hoek": round(waar, 1)}
    return out


def tandwielen():
    out = {}
    for kant in ("rechts", "links"):
        gs = []
        for th in np.linspace(P.THETA_OPEN, P.THETA_DICHT, 65):
            st = pod.stand(th)
            gs.append(gap(st["schuif"][1], st[f"binnenarm voor {kant}"][1], 3.0))
        out[kant] = {"kleinste_mm": round(min(gs), 3), "grootste_mm": round(max(gs), 3)}
    out["slag_mm"] = round(P.SLAG, 2)
    return out


def aanslagen():
    o = pod.stand(P.THETA_OPEN)
    o2 = pod.stand(P.THETA_OPEN + 2)
    d = pod.stand(P.THETA_DICHT)
    d2 = pod.stand(P.THETA_DICHT - 2)
    return {"open_binnenarm_tegen_nok_mm": round(gap(o["binnenarm voor rechts"][1], o["voet"][1]), 3),
            "open_plus_2_graden_mm": round(gap(o2["binnenarm voor rechts"][1], o2["voet"][1]), 3),
            "dicht_knop_op_dak_mm": round(gap(d["schuif"][1], d["dak"][1]), 3),
            "dicht_min_2_graden_knop_mm": round(gap(d2["schuif"][1], d2["dak"][1]), 3)}


# 5-6 -------------------------------------------------------------------------------------------
def horizontaal():
    # elke arm heeft aan beide eindes speling (gat in de arm min de pen, en gat in de wand min de pen)
    s = (P.GAT_ARM - P.PEN_D) / 2 + (P.GAT_WAND - P.PEN_D) / 2
    kantel = math.degrees(math.atan(2 * 2 * s / P.ARM_D))
    return {"parallellogram": "bak verschuift alleen, draait niet (exact)",
            "speling_per_scharnier_mm": round(s, 3), "grootste_scheefstand_graden": round(kantel, 2)}


def massa(slicer=None):
    """Gram per onderdeel: uit de slicer (als die er is), anders volume x dichtheid x vulgraad."""
    d = pod.onderdelen()
    vul = {"voet": 0.35, "dak": 0.55}
    out = {}
    for n, (_, m) in d.items():
        out[n] = m.volume() / 1000 * PLA * vul.get(n, 0.9)
    out["pennen"] = 16 * pod.pen().volume() / 1000 * PLA
    if slicer:
        for k, r in slicer.items():
            if k == "voet" and "voet" in out:
                out["voet"] = r["gram"]
            if k == "dak":
                out["dak"] = r["gram"]
    return out


def zwaartepunt(theta, lading_r, lading_l, g):
    st = pod.stand(theta)
    tot, mom = 0.0, 0.0
    for n, (_, m) in st.items():
        gram = g.get(n, 0.0)
        if n.startswith("pen"):
            gram = g["pennen"] / 16
        if gram:
            tm = to_trimesh(m)
            tot += gram
            mom += gram * float(tm.center_mass[0])
    # lading in het midden van een bak
    dx, dz = P.verschuiving(theta)
    cx = P.BAK_X0 + P.BAK_B / 2 + dx
    tot += lading_r + lading_l
    mom += lading_r * cx - lading_l * cx
    return mom / tot, tot


def kantelen(g):
    out = []
    rand = P.VOET_HALF
    for th in (P.THETA_DICHT, 75, 60, 45, P.THETA_OPEN):
        x, tot = zwaartepunt(th, SCHROEVEN_G, 0.0, g)
        out.append({"hoek": round(float(th), 1), "zwaartepunt_x_mm": round(x, 1), "rand_voet_mm": rand,
                    "marge_mm": round(rand - abs(x), 1), "totaal_g": round(tot)})
    return out


def knopkracht(g):
    """Kracht op de knop om de pod in een stand te houden (N), met beide bakjes vol of leeg."""
    out = []
    bak_g = g["bak rechts"]
    for lading in (0.0, SCHROEVEN_G):
        for th in (P.THETA_DICHT, 80, 60, 45, P.THETA_OPEN):
            # parallellogram: zwaartekrachtmoment van een bak om de voetpunten = W * L * cos(theta)
            w = (bak_g + lading) / 1000 * 9.81
            moment = 2 * w * P.L / 1000 * math.cos(math.radians(th))
            out.append({"lading_per_bak_g": lading, "hoek": round(float(th), 1),
                        "kracht_N": round(moment / (P.R_WIEL / 1000), 1)})
    return out


# 7 ---------------------------------------------------------------------------------------------
def printbaar():
    out = {}
    for naam, kleur, delen in platen.platen():
        out[naam] = printcheck.plaat({n: m for n, m in delen})
        out[naam]["kleur"] = kleur
        del out[naam]["kleuren"]
    return out


# tekening --------------------------------------------------------------------------------------
KL = {P.ZWART: "#3a3a3a", P.ORANJE: "#ff9016", P.WIT: "#e8e4da"}


def snede(m, y):
    T = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float)
    return affinity.scale(cs_to_shapely(m.transform(T).slice(y)), 1, -1, origin=(0, 0))


def _vul(ax, g, c, z):
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or not hasattr(p, "exterior"):
            continue
        p = orient(p, 1.0)
        pad = MPath.make_compound_path(*[MPath(np.asarray(r.coords), closed=True) for r in [p.exterior, *p.interiors]])
        ax.add_patch(PathPatch(pad, facecolor=c, edgecolor="#222", lw=0.3, zorder=z))


def tekening(pad):
    hoeken = [P.THETA_DICHT, 70, 50, P.THETA_OPEN]
    fig, axs = plt.subplots(2, len(hoeken), figsize=(4.4 * len(hoeken), 6.6), squeeze=False)
    for j, th in enumerate(hoeken):
        st = pod.stand(th)
        for i, (y, titel) in enumerate(((-2.75, "armen en heugel (voorkant)"), (40.0, "doorsnede midden"))):
            ax = axs[i][j]
            for naam, (kl, m) in st.items():
                _vul(ax, snede(m, y), KL[kl], 1 + ("arm" in naam) + 2 * ("pen" in naam))
            ax.set_aspect("equal")
            ax.set_xlim(-112, 112)
            ax.set_ylim(-3, 110)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{th:.0f}°  ·  {titel}", fontsize=9)
    fig.tight_layout()
    fig.savefig(pad, dpi=90)
    plt.close(fig)


# rapport ---------------------------------------------------------------------------------------
def ok(v):
    return "✅" if v else "⚠️"


def rapport(r):
    L = ["# Controle B-34B Repair Pod", "",
         f"Automatisch gegenereerd door `cad/controle.py` ({P.VERSIE}). Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "![Mechanisme](mechanisme.png)", "",
         "## 1. Botsingen over de hele slag", ""]
    b = r["botsingen"]
    if b["paren_met_overlap"]:
        L += [f"⚠️ {len(b['paren_met_overlap'])} paren raken elkaar:", ""]
        L += [f"- {p['a']} / {p['b']}: {p['mm3']} mm³ bij {p['hoek']}°" for p in b["paren_met_overlap"]]
    else:
        L += [f"✅ {b['standen']} standen van {P.THETA_DICHT:.0f}° (dicht) tot {P.THETA_OPEN:.0f}° (open), "
              "38 onderdelen inclusief pennen, elk paar: nergens overlap."]
    L += ["", "## 2. Spelingen over de hele slag", "", "| Paar | Kleinste speling | Bij hoek |", "|---|---|---|"]
    for k, v in r["spelingen"].items():
        L.append(f"| {k} | {ok(v['kleinste_mm'] >= 0.3)} {v['kleinste_mm']:.2f} | {v['bij_hoek']}° |")
    t = r["tandwielen"]
    L += ["", "## 3. Tandsector en heugel", "",
          f"Slag van de knop: {t['slag_mm']} mm. Speling tussen de tanden over de hele slag (0 = overlap, "
          "groot = los):", "", "| Kant | Kleinste | Grootste |", "|---|---|---|"]
    for kant in ("rechts", "links"):
        v = t[kant]
        goed = v["kleinste_mm"] > 0.02 and v["grootste_mm"] < 0.35
        L.append(f"| {kant} | {ok(goed)} {v['kleinste_mm']:.3f} | {v['grootste_mm']:.3f} |")
    a = r["aanslagen"]
    L += ["", "## 4. Eindstops", "", "| Stand | Afstand | Zo hoort het |", "|---|---|---|",
          f"| open: binnenarm tegen de nok | {ok(a['open_binnenarm_tegen_nok_mm'] < 0.05)} "
          f"{a['open_binnenarm_tegen_nok_mm']:.3f} | 0 (raakt) |",
          f"| 2° voor open | {ok(a['open_plus_2_graden_mm'] > 0.1)} {a['open_plus_2_graden_mm']:.3f} | vrij |",
          f"| dicht: knop op het dak | {ok(a['dicht_knop_op_dak_mm'] < 0.05)} {a['dicht_knop_op_dak_mm']:.3f} | 0 (raakt) |",
          f"| 2° voor dicht | {ok(a['dicht_min_2_graden_knop_mm'] > 0.1)} {a['dicht_min_2_graden_knop_mm']:.3f} | vrij |"]
    h = r["horizontaal"]
    L += ["", "## 5. Horizontaal", "",
          "- Parallellogram: de bak verschuift alleen en draait niet, in elke stand.",
          f"- Speling per scharnier {h['speling_per_scharnier_mm']} mm (gat in arm en wand min de pen): de bak kan "
          f"daardoor hooguit {h['grootste_scheefstand_graden']}° scheef hangen. Pas `GAT_ARM` aan na de pasproef.",
          "", "## 6. Kracht op de knop en kantelen", "",
          "Kracht om de pod in een stand te houden (zwaartekracht trekt hem open; dicht houdt hij zichzelf):", "",
          "| Lading per bak | " + " | ".join(f"{v['hoek']:.0f}°" for v in r["knopkracht"][:5]) + " |",
          "|---|" + "---|" * 5]
    for lading in (0.0, SCHROEVEN_G):
        rij = [v for v in r["knopkracht"] if v["lading_per_bak_g"] == lading]
        L.append(f"| {lading:.0f} g | " + " | ".join(f"{v['kracht_N']:.1f} N" for v in rij) + " |")
    L += ["", f"Kantelen met {SCHROEVEN_G:.0f} g in één bakje en het andere leeg (zwaartepunt t.o.v. de voetrand "
          f"op {P.VOET_HALF:.0f} mm):", "", "| Hoek | Zwaartepunt x | Marge tot de rand | Totaal |", "|---|---|---|---|"]
    for v in r["kantelen"]:
        L.append(f"| {v['hoek']:.0f}° | {v['zwaartepunt_x_mm']} | {ok(v['marge_mm'] > 10)} {v['marge_mm']} | {v['totaal_g']} g |")
    L += ["", "## 7. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |", "|---|---|---|---|---|---|"]
    for naam, v in r["print"].items():
        over = sum(o["oppervlak_mm2"] for o in v["overhang"])
        maat = " × ".join(f"{x:.0f}" for x in v["maat_mm"])
        L.append(f"| {naam} | {P.FILAMENTEN[v['kleur']][0]} | {maat} | {ok(v['past_op_A1'])} | "
                 f"{ok(over < 30)} {over:.0f} | {ok(not v['dunne_details'])} "
                 f"{'geen' if not v['dunne_details'] else v['dunne_details'][0]['grootste_te_smalle_stuk_mm2']} |")
    if "slicer" in r:
        L += ["", "### Slicer (PrusaSlicer, A1-achtig profiel)", "",
              "| Plaat | Filament | Tijd (indicatie) | Support nodig |", "|---|---|---|---|"]
        for naam, v in r["slicer"].items():
            L.append(f"| {naam} | {v['gram']:.1f} g | {v['tijd']} | {ok(not v['support'])} {'ja' if v['support'] else 'nee'} |")
    return "\n".join(L) + "\n"


def main(argv):
    OUT.mkdir(exist_ok=True)
    sl_pad = OUT / "slicer.json"
    slicer = json.loads(sl_pad.read_text()) if sl_pad.exists() else None
    g = massa()
    r = {"botsingen": botsingen(), "spelingen": spelingen(), "tandwielen": tandwielen(), "aanslagen": aanslagen(),
         "horizontaal": horizontaal(), "massa_g": {k: round(v, 1) for k, v in g.items()},
         "knopkracht": knopkracht(g), "kantelen": kantelen(g), "print": printbaar()}
    if slicer:
        r["slicer"] = slicer
    (OUT / "controle.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    tekening(OUT / "mechanisme.png")
    (OUT / "RAPPORT.md").write_text(rapport(r))
    print((OUT / "RAPPORT.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1:])
