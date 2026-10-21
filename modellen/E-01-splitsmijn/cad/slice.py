"""Slicet de platen van E-01 met PrusaSlicer (A1-achtig profiel): de pasproef en het eindmodel.

Gebruik:  python slice.py        (vereist `prusa-slicer`)
Schrijft  controle/slicer.json        (pasproef: gewicht, tijd, support per plaat)
          controle/slicer_eind.json   (eindmodel: hetzelfde, met de instellingen per object uit bouw.INST)
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import _pad  # noqa: F401
import platen
from bouw import inst
from gereedschap.geom import to_trimesh, union
from gereedschap.slicer import slice_plaat

ROOT = Path(__file__).resolve().parent.parent

# PrusaSlicer-tegenhanger van de Bambu-instellingen per object (zie bouw.INST)
INSTELLINGEN = {"perimeters": 3}
EXTRA = {"5 slinger, mesblok en plank": {"fill_density": "100%", "fill_pattern": "rectilinear"}}


def prusa(ins):
    """Bambu-instellingen van een object (bouw.INST) als PrusaSlicer-sleutels."""
    c = {"perimeters": int(ins.get("wall_loops", "2"))}
    if ins.get("sparse_infill_density") == "100%":
        c.update(fill_density="100%", fill_pattern="rectilinear")
    return c


def seconden(t):
    return sum(int(n) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[e] for n, e in re.findall(r"(\d+)([dhms])", t))


def tijd(s):
    return f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m {s % 60}s"


def proef(td):
    uit = {}
    for naam, kleur, delen in platen.platen():
        tm = to_trimesh(union([m for _, m in delen]).simplify(1e-3))
        r = slice_plaat([(kleur, tm)], naam, instellingen={**INSTELLINGEN, **EXTRA.get(naam, {})}, tmp=Path(td))
        uit[naam] = r
        print(f"{naam:24s} {r['gram']:6.1f} g  {r['tijd']:>12s}  support {r['support']}", flush=True)
    (ROOT / "controle" / "slicer.json").write_text(json.dumps(uit, indent=2, ensure_ascii=False) + "\n")


def eind(td):
    """Per plaat de objecten met dezelfde instellingen samen geslicet, in één kleur (zonder de spoeling van een
    kleurwissel: die rekent Bambu Studio)."""
    uit = {}
    for plaat, objecten in platen.platen_eind():
        groepen = {}
        for naam, delen in objecten:
            c = prusa(inst(naam))
            groepen.setdefault(json.dumps(c, sort_keys=True), (c, []))[1].extend(m for *_, m in delen)
        r = {"gram": 0.0, "seconden": 0, "support": False, "groepen": []}
        for c, ms in groepen.values():
            tm = to_trimesh(union(ms).simplify(1e-3))
            g = slice_plaat([("PLA", tm)], plaat, instellingen=c, tmp=Path(td))
            r["gram"] = round(r["gram"] + g["gram"], 1)
            r["seconden"] += seconden(g["tijd"])
            r["support"] = r["support"] or g["support"]
            r["groepen"].append({"instellingen": c, "gram": g["gram"], "tijd": g["tijd"],
                                 "support_extrusie_mm": g["support_extrusie_mm"]})
        r["tijd"] = tijd(r["seconden"])
        uit[plaat] = r
        print(f"{plaat:48s} {r['gram']:6.1f} g  {r['tijd']:>8s}  support {r['support']}", flush=True)
    (ROOT / "controle" / "slicer_eind.json").write_text(json.dumps(uit, indent=2, ensure_ascii=False) + "\n")


def main():
    (ROOT / "controle").mkdir(exist_ok=True)
    welke = sys.argv[1:] or ["proef", "eind"]
    with tempfile.TemporaryDirectory() as td:
        for w in welke:
            {"proef": proef, "eind": eind}[w](td)


if __name__ == "__main__":
    main()
