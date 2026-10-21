"""Controle van H-02 Moodkin, op de echte geometrie.

Gebruik:  python controle.py
Schrijft  controle/RAPPORT.md, controle/controle.json, controle/gezichten.png en controle/doorsnede.png.

Wat wordt gecontroleerd
  1. Gezichten: elk gezicht valt helemaal binnen de kassen van de schil (met MARGE), en wat het zelfdragend maken
     verandert. De dunste stukjes tussen twee gaten.
  2. Licht: kan je door de kassen een gat van een ander gezicht zien? Hoek tussen wat door de kassen te zien is
     en het dichtste gat van de buren.
  3. Passen: speling trommel-schil over de hele hoogte, en in 3D geen overlap tussen de onderdelen, in elke stand
     en halverwege een klik (deksel opgetild op de nokjes). Pennetjes, steeltje, nokjes en gleufjes, LED.
  4. Printbaarheid per onderdeel in printstand (gereedschap/printcheck.py): overhang, bruggen, te dunne details,
     maat; één stuk en waterdicht; gewicht geschat.
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _pad  # noqa: F401,E402
import gezichten as G  # noqa: E402
import lantaarn as L  # noqa: E402
import params as P  # noqa: E402
from bouw import ONDERDELEN  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely, polys  # noqa: E402
from gereedschap.printcheck import gram, stukken  # noqa: E402
from gereedschap.zelfdragend import zelfdragend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
MINI = 180.0
LED_H = 45.0            # hoogste LED-theelichtjes, met het plastic vlammetje
KLEUR = {"schil": "#FF9016", "trommel": "#FEC600", "deksel": "#E07A10", "steel": "#3F8E43"}


def fmt(v, n=1):
    return f"{v:.{n}f}".replace(".", ",")


def dunste_stuk(gaten):
    """Kleinste afstand tussen twee gaten (mm): het dunste stukje wand dat ertussen staat."""
    ps = polys(gaten)
    return min((a.distance(b) for i, a in enumerate(ps) for b in ps[i + 1:]), default=None)


# --- 1. gezichten ----------------------------------------------------------------------------------

def gezichten():
    kassen = G.kassen()
    w = G._wand()
    kas_aanp = zelfdragend(w.difference(kassen), P.RES, P.OVERHANG, brug=P.BRUG).intersection(kassen).area
    binnen = kassen.buffer(-G.MARGE + 0.05)
    uit = {"kassen": {"oppervlak_mm2": round(kassen.area, 0), "printaanpassing_mm2": round(kas_aanp, 2),
                      "dunste_tand_mm": round(dunste_stuk(kassen), 2)}, "gezichten": []}
    for n in P.GEZICHTEN:
        g0, g = G.VORMEN[n](), G.gezicht(n)
        uit["gezichten"].append({"naam": n, "oppervlak_mm2": round(g.area, 0),
                                 "buiten_kassen_mm2": round(g.difference(binnen).area, 3),
                                 "dichtgemaakt_mm2": round(g0.difference(g).area, 1),
                                 "bijgemaakt_mm2": round(g.difference(g0).area, 1),
                                 "dunste_stuk_mm": round(dunste_stuk(g), 2),
                                 "breedte_u_mm": round(max(abs(g.bounds[0]), abs(g.bounds[2])), 1)})
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
    for ax, n in zip(axs, P.GEZICHTEN):
        ax.set_facecolor(KLEUR["schil"])
        for p in polys(kassen):
            ax.fill(*p.exterior.xy, color=KLEUR["trommel"], lw=0)
        for p in polys(G.gezicht(n)):
            ax.fill(*p.exterior.xy, color="#2A1A0A", lw=0)
            for r in p.interiors:
                ax.fill(*r.xy, color=KLEUR["trommel"], lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(-G.B, G.B)
        ax.set_ylim(-1.5, G.H + 1)
        ax.set_title(n, loc="left")
    fig.suptitle("Oranje = schil, geel = trommel door de kassen heen, donker = gaten (die geven 's nachts licht). "
                 "Maten in mm, v = 0 is de onderkant van de band.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "gezichten.png", dpi=80)
    plt.close(fig)
    return uit


# --- 2. licht --------------------------------------------------------------------------------------

def licht(gz):
    """Wat je door een kas op de trommel kan zien, tegen het dichtste gat van een ander gezicht.

    Een kas loopt tot |x| = x_kas aan de binnenkant van de schil (straal R_BINNEN_SCHIL). Wie er schuin door kijkt,
    ziet de trommel nog een stukje verder: hoogstens SPELING·tan(80°) opzij (een blik van 80° naast loodrecht).
    Een gat van gezicht k zit op de trommel tot |u| = breedte_u van zijn eigen midden; dat van de buur dus op
    90° - asin(breedte_u / R_TROMMEL) van vooraan.
    """
    x_kas = max(abs(G.kassen().bounds[0]), abs(G.kassen().bounds[2]))
    zicht = math.degrees(math.asin(x_kas / P.R_BINNEN_SCHIL)) + math.degrees(P.SPELING * math.tan(math.radians(80))
                                                                                / P.R_TROMMEL)
    u_max = max(g["breedte_u_mm"] for g in gz["gezichten"])
    buur = 90.0 - math.degrees(math.asin(u_max / P.R_TROMMEL))
    return {"kas_tot_x_mm": round(x_kas, 1), "zichtbaar_tot_graden": round(zicht, 1),
            "dichtste_burengat_graden": round(buur, 1), "marge_graden": round(buur - zicht, 1)}


# --- 3. passen -------------------------------------------------------------------------------------

def overlap(a, b):
    return round(max(0.0, (a ^ b).volume()), 3)


def passen():
    zs = np.arange(P.Z_TROMMEL, P.H_SCHIL + 0.01, 0.1)
    speling = min(L.r_binnen(z) - P.R_TROMMEL for z in zs)
    standen = []
    for k, (stand, op) in enumerate([(0, 0.0), (0.5, P.NOK_H), (1, 0.0), (1.5, P.NOK_H), (2, 0.0), (3, 0.0)]):
        d = L.samen(stand, op)
        standen.append({"stand": stand, "opgetild_mm": op,
                        "trommel_schil_mm3": overlap(d["trommel"], d["schil"]),
                        "deksel_schil_mm3": overlap(d["deksel"], d["schil"]),
                        "trommel_deksel_mm3": overlap(d["trommel"], d["deksel"]),
                        "steel_deksel_mm3": overlap(d["steel"], d["deksel"])})
    return {"speling_trommel_schil_min_mm": round(speling, 2),
            "trommel_eruit_mm": round(P.R_BINNEN_SCHIL - P.R_TROMMEL, 2),
            "pen_speling_per_kant_mm": round((P.PEN_GAT - P.PEN_D) / 2, 2),
            "steel_speling_per_kant_mm": round((P.STEEL_GAT - P.STEEL_PEN) / 2, 2),
            "nok_in_gleuf": {"nok_h": P.NOK_H, "gleuf_diep": P.GLEUF_DIEP, "nok_b": P.NOK_B, "gleuf_b": P.GLEUF_B},
            "led": {"ring_d": P.LED_RING_D, "led_d": P.LED_D, "speling_per_kant_mm": round((P.LED_RING_D - P.LED_D) / 2, 2),
                    "vrije_hoogte_mm": round(P.H_SCHIL - P.VLOER, 1), "led_h_max_mm": LED_H,
                    "trommel_binnen_r_mm": P.R_TROMMEL - P.TROMMEL_WAND},
            "standen": standen}


def doorsnede():
    d = L.samen()
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5))
    for naam, m in d.items():
        for ax, s in ((axs[0], m.rotate([0, 0, 90]).rotate([90, 0, 0]).slice(0.0)), (axs[1], m.slice(48.0))):
            for p in polys(cs_to_shapely(s)):
                ax.fill(*p.exterior.xy, color=KLEUR[naam], lw=0)
                for r in p.interiors:
                    ax.fill(*r.xy, color="white", lw=0)
    axs[0].set_title("Doorsnede van opzij (vooraan links): schil, trommel aan het deksel, steeltje", loc="left")
    axs[1].set_title("Doorsnede op z = 48 mm (vooraan onder): de vier gezichten in de trommel", loc="left")
    for ax in axs:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "doorsnede.png", dpi=70)
    plt.close(fig)


# --- 4. printbaarheid ------------------------------------------------------------------------------

def printbaarheid():
    uit = {}
    for naam, m in L.printstand().items():
        nl, kleur, inst = ONDERDELEN[naam]
        pc = printcheck.plaat({kleur: m})
        bruggen = [o for o in pc["overhang"] if o["hoek_tov_horizontaal_graden"] < 15]
        over = [o for o in pc["overhang"] if o["hoek_tov_horizontaal_graden"] >= 15]
        n, dicht = stukken(m)
        uit[naam] = {"naam": nl, "stukken": n, "waterdicht": dicht, "maat_mm": pc["maat_mm"],
                     "past_op_A1": pc["past_op_A1"], "past_op_A1_mini": bool(max(pc["maat_mm"][:2]) <= MINI - 6),
                     "overhang_mm2": round(sum(o["oppervlak_mm2"] for o in over), 1),
                     "vlakke_plafonds_mm2": round(sum(o["oppervlak_mm2"] for o in bruggen), 1),
                     "dunne_details": pc["dunne_details"],
                     "gram_geschat": gram(m, wand=0.42 * int(inst["wall_loops"])), "printcheck": pc}
    return uit


def main():
    OUT.mkdir(exist_ok=True)
    gz = gezichten()
    res = {"gezichten": gz, "licht": licht(gz), "passen": passen(), "printbaarheid": printbaarheid()}
    doorsnede()
    (OUT / "controle.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    ok = lambda b: "✅" if b else "⚠️"  # noqa: E731

    L_ = ["# Controle H-02 Moodkin", "",
          "Automatisch gegenereerd door `cad/controle.py`. Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
          "## 1. Gezichten", "",
          f"De schil heeft vaste kassen ({fmt(gz['kassen']['oppervlak_mm2'], 0)} mm²); de gezichten zijn gaten in de "
          f"trommel daarachter. Elk gat moet minstens {fmt(G.MARGE)} mm binnen de rand van de kassen blijven. "
          f"*Dichtgemaakt* is wat het zelfdragend maken van de getekende gaten afhaalt (een randje van één rij hoog "
          f"bovenaan en onderaan), *bijgemaakt* wat het toevoegt.", "",
          "| Gezicht | Binnen de kassen | Dichtgemaakt | Bijgemaakt | Dunste stuk tussen twee gaten |", "|---|---|---|---|---|"]
    for g in gz["gezichten"]:
        L_.append(f"| {g['naam']} | {ok(g['buiten_kassen_mm2'] < 0.01)} {fmt(g['buiten_kassen_mm2'], 2)} mm² erbuiten | "
                  f"{fmt(g['dichtgemaakt_mm2'])} mm² | {fmt(g['bijgemaakt_mm2'])} mm² | "
                  f"{ok(g['dunste_stuk_mm'] >= 1.2)} {fmt(g['dunste_stuk_mm'], 2)} |")
    k = gz["kassen"]
    L_ += ["", f"- Kassen in de schil: het zelfdragend maken verandert nog {fmt(k['printaanpassing_mm2'], 2)} mm² "
               f"{ok(k['printaanpassing_mm2'] < 2)} (randjes van één rij). Dunste tand tussen twee kassen: "
               f"{fmt(k['dunste_tand_mm'], 2)} mm {ok(k['dunste_tand_mm'] >= 1.2)}.", "",
           "![Gezichten](gezichten.png)", "",
           "## 2. Licht", ""]
    li = res["licht"]
    L_ += [f"- Door de kassen zie je de trommel tot hoogstens {fmt(li['zichtbaar_tot_graden'])}° naast vooraan, ook als "
           f"je er schuin (80°) door kijkt.",
           f"- Het dichtste gat van een ander gezicht zit op {fmt(li['dichtste_burengat_graden'])}°. "
           f"{ok(li['marge_graden'] > 10)} Marge {fmt(li['marge_graden'])}°: je ziet altijd maar één gezicht.", "",
           "## 3. Passen", ""]
    pa = res["passen"]
    L_ += [f"- Speling trommel-schil: minstens {fmt(pa['speling_trommel_schil_min_mm'], 2)} mm rondom over de hele hoogte "
           f"{ok(pa['speling_trommel_schil_min_mm'] >= 0.8)}. Deksel met trommel komt er recht uit "
           f"({fmt(pa['trommel_eruit_mm'])} mm speling bij de rand).",
           f"- Pennetjes trommel-deksel: {fmt(pa['pen_speling_per_kant_mm'], 2)} mm per kant (klemt met duimdruk). "
           f"Steeltje: {fmt(pa['steel_speling_per_kant_mm'], 2)} mm per kant, vierkant.",
           f"- Nokje {fmt(P.NOK_H)} hoog en {fmt(P.NOK_B)} breed in een gleufje van {fmt(P.GLEUF_DIEP)} diep en "
           f"{fmt(P.GLEUF_B)} breed {ok(P.NOK_H < P.GLEUF_DIEP and P.NOK_B < P.GLEUF_B)}: in elke stand ligt het deksel "
           f"plat op de rand; tussen twee standen tilt het {fmt(P.NOK_H)} mm op en valt het met een klik terug.",
           f"- LED-theelichtje Ø {fmt(P.LED_D, 0)} in een ringetje van Ø {fmt(P.LED_RING_D)} "
           f"({fmt(pa['led']['speling_per_kant_mm'], 2)} mm per kant); {fmt(pa['led']['vrije_hoogte_mm'], 0)} mm vrij "
           f"boven de vloer {ok(pa['led']['vrije_hoogte_mm'] > LED_H)} (de hoogste LED-lichtjes zijn ~{fmt(LED_H, 0)} mm).",
           "", "Overlap in 3D (mm³, moet 0 zijn):", "",
           "| Stand | Opgetild | Trommel-schil | Deksel-schil | Trommel-deksel | Steeltje-deksel |", "|---|---|---|---|---|---|"]
    for s in pa["standen"]:
        waarden = [s["trommel_schil_mm3"], s["deksel_schil_mm3"], s["trommel_deksel_mm3"], s["steel_deksel_mm3"]]
        L_.append(f"| {fmt(s['stand'], 1)} | {fmt(s['opgetild_mm'])} | " + " | ".join(f"{ok(v < 0.01)} {fmt(v, 2)}" for v in waarden) + " |")
    L_ += ["", "*Stand 0,5*: halverwege een klik, het deksel ligt op de nokjes.", "", "![Doorsnede](doorsnede.png)", "",
           "## 4. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
           "| Onderdeel | Maat | Eén stuk, waterdicht | A1 / A1 mini | Overhang > 48° | Vlakke plafonds (bruggen) | Te smal | Gewicht (schatting) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in res["printbaarheid"].values():
        dun = r["dunne_details"]
        L_.append(f"| {r['naam']} | {' × '.join(fmt(v, 0) for v in r['maat_mm'])} | "
                  f"{ok(r['stukken'] == 1 and r['waterdicht'])} {r['stukken']} stuk | "
                  f"{ok(r['past_op_A1'])} / {ok(r['past_op_A1_mini'])} | {ok(r['overhang_mm2'] < 5)} {fmt(r['overhang_mm2'], 0)} mm² | "
                  f"{fmt(r['vlakke_plafonds_mm2'], 0)} mm² | "
                  f"{'✅ geen' if not dun or dun[0]['grootste_te_smalle_stuk_mm2'] < 1.5 else '⚠️ ' + fmt(dun[0]['grootste_te_smalle_stuk_mm2']) + ' mm²'} | "
                  f"~{fmt(r['gram_geschat'], 0)} g |")
    L_ += ["", f"*Vlakke plafonds*: de bovenkant van de tandkolommen in de schil. Die zijn {fmt(G.MOND_B, 0)} mm breed; de "
           "printer overbrugt ze van de ene tand naar de andere. De platte gaten in de trommel (tandgaten, de ogen van "
           f"'bang', hoogstens {fmt(P.BRUG, 0)} mm breed) zijn even goed bruggen, maar in een wand van "
           f"{fmt(P.TROMMEL_WAND)} mm telt printcheck ze als korte brug. Alle plafonds zijn in 2D gecontroleerd (zie 1). "
           "Korte bruggen (< 3 mm) en de plafonds van de pengaatjes en gleufjes onder het deksel tellen niet als overhang.",
           "", "Het gewicht is geschat uit de doorsneden (de wanden uit het project, boven- en onderlagen, 15 % infill), niet met een slicer.", "",
           "Niet getest: een echte print, het klikken, en hoe fel het licht door de gaten komt met een echt "
           "LED-theelichtje."]
    (OUT / "RAPPORT.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
