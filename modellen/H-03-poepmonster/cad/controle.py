"""Controle van H-03 Poepmonster, op de echte geometrie.

Gebruik:  python controle.py
Schrijft  controle/RAPPORT.md, controle/controle.json en controle/gezicht.png.

Wat wordt gecontroleerd
  1. Gezicht: wat het zelfdragend maken aan de gaten verandert, hoe breed de spleetjes in de buik zijn (houden ze
     de poep binnen?) en het dunste stukje wand tussen twee gaten.
  2. Rand en inhoud: hoogte van de rand achteraan en vooraan, en hoeveel erin past tot de achterrand.
  3. Printbaarheid (gereedschap/printcheck.py): overhang, bruggen, te dunne details, maat; één stuk en
     waterdicht, ook als STL; gewicht geschat.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _pad  # noqa: F401,E402
import gezicht as G  # noqa: E402
import monster as M  # noqa: E402
import params as P  # noqa: E402
from bouw import INST  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import polys  # noqa: E402
from gereedschap.printcheck import gram, stukken  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
MINI = 180.0
POEP_MIN = 8.0          # de kleinste klodders van de A1 zijn ongeveer zo breed


def fmt(v, n=1):
    return f"{v:.{n}f}".replace(".", ",")


def gezicht(sleutel):
    g0, g = G.ontwerp(sleutel), G.gaten(sleutel)
    ps = polys(g)
    v0, v1 = P.VARIANTEN[sleutel][4]
    spleten = [p for p in ps if p.bounds[1] < v1 and p.bounds[3] > v0 + 2]
    return {"dichtgemaakt_mm2": round(g0.difference(g).area, 1), "bijgemaakt_mm2": round(g.difference(g0).area, 1),
            "spleetjes": len(spleten), "breedste_spleet_mm": round(max(p.bounds[2] - p.bounds[0] for p in spleten), 2),
            "dunste_stuk_mm": round(min(a.distance(b) for i, a in enumerate(ps) for b in ps[i + 1:]), 2)}


def teken():
    fig, axs = plt.subplots(1, len(P.VARIANTEN), figsize=(6.5 * len(P.VARIANTEN), 7))
    for ax, k in zip(axs, P.VARIANTEN):
        ax.set_facecolor("#FF9016")
        for p in polys(G.gaten(k)):
            ax.fill(*p.exterior.xy, color="#2A1A0A", lw=0)
            for r in p.interiors:
                ax.fill(*r.xy, color="#FF9016", lw=0)
        for p in polys(G.ontwerp(k)):
            ax.plot(*p.exterior.xy, color="white", lw=0.6)
        ax.axhline(P.VARIANTEN[k][3], color="#7A3A00", lw=1, ls="--")
        ax.set_aspect("equal")
        ax.set_xlim(-45, 45)
        ax.set_ylim(0, P.VARIANTEN[k][3] + P.TAND_H + 3)
        ax.set_title(f"{P.VARIANTEN[k][0]}: rand vooraan op {fmt(P.VARIANTEN[k][3], 0)} mm (stippellijn)", loc="left")
    fig.suptitle("Gaten vooraan zoals geprint (donker) en zoals getekend (witte lijn). Maten in mm boven het bed.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "gezicht.png", dpi=70)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for k, (nl, en, achter, voor, buik) in P.VARIANTEN.items():
        m = M.monster(k)
        pc = printcheck.plaat({P.ORANJE: m})
        n, dicht = stukken(m)
        res[k] = {"naam": nl, "rand_achter_mm": achter, "rand_voor_mm": voor, "tand_mm": P.TAND_H,
                  "inhoud_cm3": M.inhoud_cm3(k), "gezicht": gezicht(k), "stukken": n, "waterdicht": dicht,
                  "past_op_A1_mini": bool(max(pc["maat_mm"][:2]) <= MINI - 6),
                  "overhang_mm2": round(sum(o["oppervlak_mm2"] for o in pc["overhang"]
                                            if o["hoek_tov_horizontaal_graden"] >= 15), 1),
                  "vlakke_plafonds_mm2": round(sum(o["oppervlak_mm2"] for o in pc["overhang"]
                                                   if o["hoek_tov_horizontaal_graden"] < 15), 1),
                  "gram_geschat": gram(m, wand=0.42 * int(INST["wall_loops"])), "printcheck": pc}
    (OUT / "controle.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    teken()
    ok = lambda b: "✅" if b else "⚠️"  # noqa: E731

    L = ["# Controle H-03 Poepmonster", "",
         "Automatisch gegenereerd door `cad/controle.py`. Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "## 1. Gezicht", "",
         "*Dichtgemaakt* is wat het zelfdragend maken van de getekende gaten afhaalt (spitse bovenkant van de ogen, "
         "randjes van één rij), *bijgemaakt* wat het toevoegt. De spleetjes in de buik moeten smaller zijn dan de "
         f"kleinste poepklodders (~{fmt(POEP_MIN, 0)} mm).", "",
         "| Variant | Dichtgemaakt | Bijgemaakt | Spleetjes in de buik | Breedste spleet | Dunste stuk tussen twee gaten |",
         "|---|---|---|---|---|---|"]
    for r in res.values():
        g = r["gezicht"]
        L.append(f"| {r['naam']} | {fmt(g['dichtgemaakt_mm2'])} mm² | {fmt(g['bijgemaakt_mm2'])} mm² | {g['spleetjes']} | "
                 f"{ok(g['breedste_spleet_mm'] < POEP_MIN)} {fmt(g['breedste_spleet_mm'])} | "
                 f"{ok(g['dunste_stuk_mm'] >= 2.0)} {fmt(g['dunste_stuk_mm'])} |")
    L += ["", "![Gezicht](gezicht.png)", "", "## 2. Rand en inhoud", "",
          "| Variant | Rand achteraan (dal / punt) | Rand vooraan (dal / punt) | Inhoud tot de achterrand |", "|---|---|---|---|"]
    for r in res.values():
        L.append(f"| {r['naam']} | {fmt(r['rand_achter_mm'], 0)} / {fmt(r['rand_achter_mm'] + r['tand_mm'], 0)} | "
                 f"{fmt(r['rand_voor_mm'], 0)} / {fmt(r['rand_voor_mm'] + r['tand_mm'], 0)} | ~{fmt(r['inhoud_cm3'], 0)} cm³ |")
    L += ["", "De afvalgoot van de printer moet hoger liggen dan de punten van de tanden achteraan.", "",
          "## 3. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Variant | Maat | Eén stuk, waterdicht | A1 / A1 mini | Overhang > 48° | Vlakke plafonds (bruggen) | Te smal | Gewicht (schatting) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res.values():
        pc, dun = r["printcheck"], r["printcheck"]["dunne_details"]
        L.append(f"| {r['naam']} | {' × '.join(fmt(v, 0) for v in pc['maat_mm'])} | "
                 f"{ok(r['stukken'] == 1 and r['waterdicht'])} {r['stukken']} stuk | "
                 f"{ok(pc['past_op_A1'])} / {ok(r['past_op_A1_mini'])} | {ok(r['overhang_mm2'] < 5)} {fmt(r['overhang_mm2'], 0)} mm² | "
                 f"{fmt(r['vlakke_plafonds_mm2'], 0)} mm² | "
                 f"{'✅ geen' if not dun or dun[0]['grootste_te_smalle_stuk_mm2'] < 1.5 else '⚠️ ' + fmt(dun[0]['grootste_te_smalle_stuk_mm2']) + ' mm²'} | "
                 f"~{fmt(r['gram_geschat'], 0)} g |")
    L += ["", f"*Vlakke plafonds*: de bovenkant van de spleetjes in de buik, {fmt(P.HECHTING_STAP - P.HECHTING_B)} mm breed "
          "van hechting tot hechting. In een wand van 2 mm telt printcheck ze als korte brug.", "",
          "Het gewicht is geschat uit de doorsneden (3 wanden, 15 % infill), niet met een slicer.", "",
          "Niet getest: een echte print, en of hij bij jouw A1 goed onder de afvalgoot staat. Meet dat na (zie de README)."]
    (OUT / "RAPPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
