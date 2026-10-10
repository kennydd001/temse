"""Slicet elke printplaat met PrusaSlicer en een profiel dat de Bambu Lab A1 benadert.

Gebruik:  python slice.py        (vereist `prusa-slicer` op het pad)
Schrijft  controle/slicer.json   (gewicht, tijd, wissels, support) en controle/a1_benadering.ini.

Bambu Studio zelf draait hier niet. Het profiel neemt de A1-standaard over (0,2 mm laag, 2 wanden,
15 % grid-infill, A1-snelheden), dus het filamentgewicht is betrouwbaar; de printtijd en de
spoelhoeveelheid van de AMS Lite wijken af en zijn alleen een indicatie.
"""
import json
import tempfile
from pathlib import Path

import trimesh

import _pad  # noqa: F401  (maakt gereedschap/ importeerbaar)
from gereedschap.slicer import A1, ini, slice_plaat
from params import FILAMENTEN

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
# korte kleurnaam (NL of EN, zoals in de bestandsnamen) -> schermkleur
HEX = {v[i]: v[2] for v in FILAMENTEN.values() for i in (3, 4)}


def main():
    """Slicet de gedeelde onderdelen en de NL-fronten (EN verschilt alleen in de labels)."""
    OUT.mkdir(exist_ok=True)
    groups = {}
    stls = (list((ROOT / "gedeeld").rglob("*.stl")) + list((ROOT / "varianten").glob("*/nl/*.stl"))
            + list((ROOT / "fanart").glob("*/nl/*.stl")))
    for f in sorted(stls):
        groups.setdefault(str(f.parent.relative_to(ROOT) / f.stem.split("__")[0]), []).append(f)
    result = {}
    with tempfile.TemporaryDirectory() as td:
        for name, files in groups.items():
            delen = [(f.stem.split("__")[1] if "__" in f.stem else "enkel", trimesh.load(f)) for f in sorted(files)]
            r = slice_plaat(delen, Path(name).name, HEX, tmp=Path(td))
            result[name] = r
            print(f"{name:42s} {r['gram']:6.1f} g  {r['tijd']:>12s}  wissels {r['wissels']:3d}  support {r['support']}")
    ini(A1, OUT / "a1_benadering.ini")
    (OUT / "slicer.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
