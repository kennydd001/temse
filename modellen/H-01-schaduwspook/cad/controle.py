"""Controle van H-01 Schaduwspook, op de echte geometrie.

Gebruik:  python controle.py
Schrijft  controle/RAPPORT.md, controle/controle.json en controle/silhouetten.png.

Wat wordt gecontroleerd
  1. Schaduw: de projectie van het beeldje in elke kijkrichting, rij per rij, tegen het silhouet zoals het
     geprint wordt, en hoeveel het zelfdragend maken afwijkt van het getekende silhouet.
  2. Eén stuk en waterdicht, ook na het wegschrijven als STL (float32).
  3. Printbaarheid (gereedschap/printcheck.py): overhang, te dunne details, maat; plus de A1 mini (180 mm).
  4. Gewicht: geschat uit de doorsneden (2 wanden van 0,42 mm, 5 boven- en 3 onderlagen massief, de rest 15 %
     infill, PLA 1,24 g/cm³).
     Een schatting, geen slicer.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _pad  # noqa: F401,E402
import beeldje as BJ  # noqa: E402
import params as P  # noqa: E402
import schaduw as S  # noqa: E402
import silhouetten  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import polys  # noqa: E402
from gereedschap.printcheck import gram, stukken  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
MINI = 180.0


def afwijking_ontwerp(ontwerp, print_):
    """Oppervlak (mm²) dat de 45°-aanpassingen toevoegen aan het getekende silhouet."""
    return round(print_.difference(ontwerp).area, 1)


def teken(res):
    n = max(len(v[2]) for v in P.VARIANTEN.values())
    fig, axs = plt.subplots(len(P.VARIANTEN), n, figsize=(4.2 * n, 4.4 * len(P.VARIANTEN)), squeeze=False)
    for row, sleutel in zip(axs, P.VARIANTEN):
        hoeken = P.VARIANTEN[sleutel][3]
        for ax in row[len(hoeken):]:
            ax.axis("off")
        for ax, g, g0, h in zip(row, BJ.silhouetten_van(sleutel), BJ.ontwerp(sleutel), hoeken):
            for p in polys(g):
                ax.fill(*p.exterior.xy, color="#FF9016", lw=0)
                for r in p.interiors:
                    ax.fill(*r.xy, color="white", lw=0)
            for p in polys(g0):
                ax.plot(*p.exterior.xy, color="black", lw=0.6)
                for r in p.interiors:
                    ax.plot(*r.xy, color="black", lw=0.6)
            ax.set_aspect("equal")
            ax.set_title(f"{P.VARIANTEN[sleutel][0]}: {h:.0f}°", fontsize=10, loc="left")
    fig.suptitle("Oranje = zoals het geprint wordt (zelfdragend), zwarte lijn = getekend silhouet", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "silhouetten.png", dpi=80)
    plt.close(fig)


def fmt(v, n=1):
    return f"{v:.{n}f}".replace(".", ",")


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for sleutel, (nl, en, vormen, hoeken, hoogte, kleur) in P.VARIANTEN.items():
        m = BJ.beeldje(sleutel)
        sil, ontw = BJ.silhouetten_van(sleutel), BJ.ontwerp(sleutel)
        n, dicht = stukken(m)
        pc = printcheck.plaat({kleur: m})
        schaduw = S.verschil(m, sil, hoeken)
        for d, v, g, g0 in zip(schaduw, vormen, sil, ontw):
            d["vorm"] = silhouetten.leesbaar(v)
            d["aanpassing_mm2"] = afwijking_ontwerp(g0, g)
        res[sleutel] = {"naam": nl, "schaduw": schaduw, "stukken": n, "waterdicht": dicht, "gram_geschat": gram(m),
                        "past_op_A1_mini": bool(max(pc["maat_mm"][:2]) <= MINI - 6), "printcheck": pc}
    (OUT / "controle.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    teken(res)

    L = ["# Controle H-01 Schaduwspook", "",
         "Automatisch gegenereerd door `cad/controle.py`. Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "## 1. Schaduw", "",
         "De projectie van het beeldje in elke kijkrichting, rij per rij (0,2 mm), tegen het silhouet zoals het "
         "geprint wordt. *Ontbreekt* is wat de andere silhouetten wegsnijden (bij vier schaduwen een randje aan de "
         "breedste plekken). *Aanpassing* is wat het zelfdragend maken toevoegt aan het getekende silhouet: spitse "
         "bovenkanten van gaten, wiggen onder overhangen en dunne steunpilaartjes.", "",
         "| Variant | Schaduw | Hoek | Ontbreekt | Te veel | Aanpassing |", "|---|---|---|---|---|---|"]
    for r in res.values():
        for d in r["schaduw"]:
            pct = 100 * d["ontbreekt_mm2"] / d["doel_mm2"]
            L.append(f"| {r['naam']} | {'✅' if pct < 3 else '⚠️'} {d['vorm']} | {d['hoek']:.0f}° | "
                     f"{fmt(d['ontbreekt_mm2'])} mm² ({fmt(pct)} %) | {fmt(d['te_veel_mm2'])} mm² | "
                     f"{fmt(d['aanpassing_mm2'])} mm² |")
    L += ["", "![Silhouetten](silhouetten.png)", "",
          "## 2. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Variant | Maat | Eén stuk, waterdicht | A1 / A1 mini | Overhang > 48° (mm²) | Te smal (< 0,4 mm) | Gewicht (schatting) |",
          "|---|---|---|---|---|---|---|"]
    for r in res.values():
        pc = r["printcheck"]
        maat = " × ".join(fmt(v, 0) for v in pc["maat_mm"])
        over = sum(o["oppervlak_mm2"] for o in pc["overhang"])
        dun = pc["dunne_details"]
        L.append(f"| {r['naam']} | {maat} | {'✅' if r['stukken'] == 1 and r['waterdicht'] else '⚠️'} "
                 f"{r['stukken']} stuk | {'✅' if pc['past_op_A1'] else '⚠️'} / {'✅' if r['past_op_A1_mini'] else '⚠️ te groot'} | "
                 f"{'✅' if over < 5 else '⚠️'} {fmt(over, 0)} | "
                 f"{'✅ geen' if not dun or dun[0]['grootste_te_smalle_stuk_mm2'] < 1.5 else '⚠️ ' + fmt(dun[0]['grootste_te_smalle_stuk_mm2']) + ' mm²'} | "
                 f"~{fmt(r['gram_geschat'], 0)} g |")
    L += ["", "*Te smal*: kleiner dan 1,5 mm² per laag telt niet (de punt van een boog of steeltje; de slicer laat dat "
          "weg). Het gewicht is geschat uit de doorsneden (wanden, boven- en onderlagen, 15 % infill), niet met een slicer.", "",
          "Niet getest: de schaduw met een echte gsm-zaklamp. De renders gebruiken een puntlamp van 2 mm op 45 cm; "
          "een grotere of dichterbij gehouden lamp geeft een zachtere, iets vertekende schaduw."]
    (OUT / "RAPPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
