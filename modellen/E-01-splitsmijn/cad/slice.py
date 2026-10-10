"""Slicet de platen van de pasproef van E-01 met PrusaSlicer (A1-achtig profiel).

Gebruik:  python slice.py        (vereist `prusa-slicer`)
Schrijft  controle/slicer.json   (gewicht, tijd, support per plaat)
"""
import json
import tempfile
from pathlib import Path

import _pad  # noqa: F401
import platen
from gereedschap.geom import to_trimesh, union
from gereedschap.slicer import slice_plaat

ROOT = Path(__file__).resolve().parent.parent

# PrusaSlicer-tegenhanger van de Bambu-instellingen per object (zie bouw.INST)
INSTELLINGEN = {"perimeters": 3}
EXTRA = {"4 slinger, mesblok en plank": {"fill_density": "100%", "fill_pattern": "rectilinear"}}


def main():
    uit = {}
    with tempfile.TemporaryDirectory() as td:
        for naam, kleur, delen in platen.platen():
            tm = to_trimesh(union([m for _, m in delen]).simplify(1e-3))
            r = slice_plaat([(kleur, tm)], naam, instellingen={**INSTELLINGEN, **EXTRA.get(naam, {})}, tmp=Path(td))
            uit[naam] = r
            print(f"{naam:24s} {r['gram']:6.1f} g  {r['tijd']:>12s}  support {r['support']}", flush=True)
    (ROOT / "controle").mkdir(exist_ok=True)
    (ROOT / "controle" / "slicer.json").write_text(json.dumps(uit, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
