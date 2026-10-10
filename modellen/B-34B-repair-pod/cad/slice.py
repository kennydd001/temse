"""Slicet de drie printplaten en de pasproef van B-34B met PrusaSlicer (A1-achtig profiel).

Gebruik:  python slice.py        (vereist `prusa-slicer`)
Schrijft  controle/slicer.json   (gewicht, tijd, support per plaat)
"""
import json
import tempfile
from pathlib import Path


import _pad  # noqa: F401
import bouw
import platen
from gereedschap.geom import to_trimesh, union
from gereedschap.slicer import slice_plaat

ROOT = Path(__file__).resolve().parent.parent

# PrusaSlicer-tegenhanger van de Bambu-instellingen per object (zie bouw.INST): 3 wanden overal
INSTELLINGEN = {"perimeters": 3}


def main():
    uit = {}
    with tempfile.TemporaryDirectory() as td:
        proef = [(f"pasproef {n}", k, d) for n, k, d in bouw.pasproef_platen()]
        for naam, kleur, delen in platen.platen() + proef:
            tm = to_trimesh(union([m for _, m in delen]).simplify(1e-3))
            r = slice_plaat([(kleur, tm)], naam, instellingen=INSTELLINGEN, tmp=Path(td))
            uit[naam] = r
            print(f"{naam:42s} {r['gram']:6.1f} g  {r['tijd']:>12s}  support {r['support']}")
    (ROOT / "controle").mkdir(exist_ok=True)
    (ROOT / "controle" / "slicer.json").write_text(json.dumps(uit, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
