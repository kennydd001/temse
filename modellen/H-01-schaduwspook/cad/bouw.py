"""Schrijft de printbestanden van H-01 Schaduwspook.

Gebruik:  python bouw.py                                  alle varianten (params.VARIANTEN)
          python bouw.py eigen VORM VORM [VORM VORM] [hoogte]
                                                          eigen beeldje met 2 tot 4 schaduwen, gelijk verdeeld
                                                          over een halve draai (2: 0° en 90°, 4: om de 45°)

  VORM: 'schedel', 'spook', 'pompoen', 'kat', 'grafsteen', 'tekst:WOORD' of 'smal:WOORD' (versmalde letters),
        met eventueel '*0.85' erachter om hem smaller te maken. B.v.:  python bouw.py eigen smal:EMMA schedel 40

  print/   per variant een STL (rechtop, voetbalk op het bed) en schaduwspook.3mf (Bambu Studio, één plaat per
           variant). Een eigen beeldje komt in print/eigen_<vormen>.stl.
"""
import json
import re
import sys
from pathlib import Path

import trimesh

import _pad  # noqa: F401
import beeldje as BJ
import params as P
import schaduw as S
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import save_stl, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PRINT = ROOT / "print"
INST = {"wall_loops": "3"}     # dunne zuiltjes: drie wanden; voetbalk ligt vlak op het bed, geen brim nodig


def bestandsnaam(sleutel):
    return f"schaduwspook_{sleutel}"


def stl(naam, m, info, extra=None):
    pad = PRINT / f"{naam}.stl"
    save_stl(m, pad)
    tm = trimesh.load(pad)
    assert tm.is_watertight and tm.is_winding_consistent, pad
    info[str(pad.relative_to(ROOT))] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents], **(extra or {})}
    print(f"  {str(pad.relative_to(ROOT)):44s} {[round(float(v), 1) for v in tm.extents]}")


def project(pad):
    codes = list(dict.fromkeys(v[5] for v in P.VARIANTEN.values()))
    slots = {c: i + 1 for i, c in enumerate(codes)}
    fil = [B.Filament(P.FILAMENTEN[c][2], P.FILAMENTEN[c][1], f"{P.FILAMENTEN[c][0]} {c}") for c in codes]
    platen = []
    for i, (sleutel, (nl, _, _, _, _, kleur)) in enumerate(P.VARIANTEN.items(), start=1):
        tm = to_trimesh(BJ.beeldje(sleutel).simplify(1e-3))
        obj = B.Object(f"schaduwspook {nl}", [B.Deel(nl, tm.vertices.copy(), tm.faces.copy(), slots[kleur])],
                       dict(INST), (128.0, 128.0))
        pl = B.Plaat(f"{i} {nl}", [obj])
        platen.append(pl)
    B.plaats_torens(fil, platen)
    for pl in platen:
        pl.miniatuur = miniatuur(pl, fil)
    namen = ", ".join(v[0] for v in P.VARIANTEN.values())
    return B.schrijf(pad, f"H-01 Schaduwspook {P.VERSIE}", fil, platen,
                     beschrijving=f"Schaduwkunst: schijn er met een lamp op en de schaduw toont iets heel anders dan het "
                                  f"beeldje. Draai het en de schaduw verandert: vier vormen (om de 45°) of twee (om de "
                                  f"90°). Varianten: {namen}.")


def main():
    PRINT.mkdir(parents=True, exist_ok=True)
    info = {}
    print("print/")
    for sleutel, (nl, en, vormen, hoeken, hoogte, kleur) in P.VARIANTEN.items():
        m = BJ.beeldje(sleutel)
        stl(bestandsnaam(sleutel), m, info, {"schaduwen": [{"vorm": v, "hoek": h} for v, h in zip(vormen, hoeken)],
                                             "hoogte_mm": hoogte,
                                             "schaduwcontrole": S.verschil(m, BJ.silhouetten_van(sleutel), hoeken)})
    r = project(PRINT / "schaduwspook.3mf")
    print("  print/schaduwspook.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def eigen(vormen, hoogte=55.0):
    """Eigen beeldje: alleen een STL, de varianten en het project blijven zoals ze zijn."""
    hoeken = [k * 180.0 / len(vormen) for k in range(len(vormen))]
    sil = BJ.printklaar(vormen, hoogte)
    m = S.object_(sil, hoeken)
    naam = "eigen_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "-", v).strip("-").lower() for v in vormen)
    PRINT.mkdir(parents=True, exist_ok=True)
    stl(naam, m, {})
    for v, d in zip(vormen, S.verschil(m, sil, hoeken)):
        print(f"  {v:16s} {d['hoek']:5.1f}°  ontbreekt {d['ontbreekt_mm2']} van {d['doel_mm2']} mm²")


def getal(s):
    try:
        return float(s)
    except ValueError:
        return None


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "eigen":
        vormen = [a for a in args[1:] if getal(a) is None]
        hoogtes = [getal(a) for a in args[1:] if getal(a) is not None]
        assert 2 <= len(vormen) <= 4, "geef 2 tot 4 vormen"
        eigen(vormen, hoogtes[0] if hoogtes else 55.0)
    else:
        main()
