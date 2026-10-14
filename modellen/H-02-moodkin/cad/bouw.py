"""Schrijft de printbestanden van H-02 Moodkin.

Gebruik:  python bouw.py
  print/   per onderdeel een STL in printstand en moodkin.3mf (Bambu Studio, één plaat per onderdeel, elke
           plaat in één kleur: geen AMS nodig).
"""
import json
from pathlib import Path

import trimesh

import _pad  # noqa: F401
import lantaarn as L
import params as P
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import save_stl, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PRINT = ROOT / "print"

# (naam NL, kleur, instellingen): de trommelwand (1,6 mm) helemaal uit wanden, de schil stevig rond de kassen
ONDERDELEN = {
    "schil": ("schil", P.ORANJE, {"wall_loops": "3"}),
    "trommel": ("trommel met de vier gezichten", P.GEEL, {"wall_loops": "4"}),
    "deksel": ("deksel", P.ORANJE, {"wall_loops": "3"}),
    "steel": ("steeltje", P.GROEN, {"wall_loops": "3", "brim_type": "outer_only", "brim_width": "4"}),
}


def stl(naam, m, info):
    pad = PRINT / f"moodkin_{naam}.stl"
    save_stl(m, pad)
    tm = trimesh.load(pad)
    assert tm.is_watertight and tm.is_winding_consistent, pad
    info[str(pad.relative_to(ROOT))] = {"onderdeel": ONDERDELEN[naam][0], "filament": P.FILAMENTEN[ONDERDELEN[naam][1]][0],
                                        "volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents]}
    print(f"  {str(pad.relative_to(ROOT)):32s} {[round(float(v), 1) for v in tm.extents]}")


def project(pad, delen):
    codes = list(dict.fromkeys(v[1] for v in ONDERDELEN.values()))
    slots = {c: i + 1 for i, c in enumerate(codes)}
    fil = [B.Filament(P.FILAMENTEN[c][2], P.FILAMENTEN[c][1], f"{P.FILAMENTEN[c][0]} {c}") for c in codes]
    platen = []
    for i, (sleutel, (nl, kleur, inst)) in enumerate(ONDERDELEN.items(), start=1):
        tm = to_trimesh(delen[sleutel].simplify(1e-3))
        obj = B.Object(f"moodkin {sleutel}", [B.Deel(nl, tm.vertices.copy(), tm.faces.copy(), slots[kleur])],
                       dict(inst), (128.0, 128.0))
        platen.append(B.Plaat(f"{i} {nl}", [obj]))
    B.plaats_torens(fil, platen)
    for pl in platen:
        pl.miniatuur = miniatuur(pl, fil)
    return B.schrijf(pad, f"H-02 Moodkin {P.VERSIE}", fil, platen,
                     beschrijving="Pompoenlantaarn met vier gezichten. Draai het deksel aan het steeltje: om de 90° "
                                  "klikt een ander gezicht voor de gaten (blij, boos, bang, knipoog). Alleen voor een "
                                  "LED-theelichtje (Ø 38 mm), nooit een echte kaars.")


def main():
    PRINT.mkdir(parents=True, exist_ok=True)
    delen = L.printstand()
    info = {}
    print("print/")
    for naam, m in delen.items():
        stl(naam, m, info)
    r = project(PRINT / "moodkin.3mf", delen)
    print("  print/moodkin.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
