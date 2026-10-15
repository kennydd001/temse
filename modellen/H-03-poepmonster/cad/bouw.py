"""Schrijft de printbestanden van H-03 Poepmonster.

Gebruik:  python bouw.py                         beide varianten (params.VARIANTEN)
          python bouw.py eigen ACHTER [VOOR]     eigen hoogtes (mm, tanddalen achteraan en vooraan); VOOR is
                                                 standaard ACHTER + 35. B.v.: python bouw.py eigen 50
  print/   per variant een STL (rechtop, zoals op het bed) en poepmonster.3mf (Bambu Studio, één plaat per
           variant). Een eigen hoogte komt in print/poepmonster_eigen_<achter>.stl.
"""
import json
import sys
from pathlib import Path

import trimesh

import _pad  # noqa: F401
import monster as M
import params as P
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import save_stl, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PRINT = ROOT / "print"
INST = {"wall_loops": "3"}     # de wand is 2 mm: met 3 lijnen (1,26 mm) en een beetje infill ertussen


def stl(naam, m, info, extra=None):
    pad = PRINT / f"{naam}.stl"
    save_stl(m, pad)
    tm = trimesh.load(pad)
    assert tm.is_watertight and tm.is_winding_consistent, pad
    info[str(pad.relative_to(ROOT))] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents], **(extra or {})}
    print(f"  {str(pad.relative_to(ROOT)):36s} {[round(float(v), 1) for v in tm.extents]}")


def project(pad):
    fil = [B.Filament(P.FILAMENTEN[P.ORANJE][2], P.FILAMENTEN[P.ORANJE][1], f"{P.FILAMENTEN[P.ORANJE][0]} {P.ORANJE}")]
    platen = []
    for i, (sleutel, (nl, *_)) in enumerate(P.VARIANTEN.items(), start=1):
        tm = to_trimesh(M.monster(sleutel).simplify(1e-3))
        obj = B.Object(f"poepmonster {nl}", [B.Deel(nl, tm.vertices.copy(), tm.faces.copy(), 1)], dict(INST),
                       (128.0, 128.0))
        platen.append(B.Plaat(f"{i} {nl}", [obj]))
    B.plaats_torens(fil, platen)
    for pl in platen:
        pl.miniatuur = miniatuur(pl, fil)
    return B.schrijf(pad, f"H-03 Poepmonster {P.VERSIE}", fil, platen,
                     beschrijving="Pompoenmonster dat het purge-afval van de A1 opvangt: zet hem waar de poep uit de "
                                  "printer valt. Laag achteraan, hoog vooraan, tanden rondom, en door de buik zie je "
                                  "hoeveel hij al gegeten heeft. Varianten: laag en hoog.")


def main():
    PRINT.mkdir(parents=True, exist_ok=True)
    info = {}
    print("print/")
    for sleutel, (nl, en, achter, voor, buik) in P.VARIANTEN.items():
        stl(f"poepmonster_{sleutel}", M.monster(sleutel), info,
            {"rand_achter_mm": achter, "rand_voor_mm": voor, "tanden_mm": P.TAND_H, "inhoud_cm3": M.inhoud_cm3(sleutel)})
    r = project(PRINT / "poepmonster.3mf")
    print("  print/poepmonster.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def eigen(achter, voor=None):
    """Eigen hoogte: alleen een STL; de varianten en het project blijven zoals ze zijn."""
    voor = achter + 35.0 if voor is None else voor
    assert 25.0 <= achter < voor <= P.Z_TOP - P.TAND_H, "achter moet tussen 25 en de voorkant liggen"
    buik0 = 9.0
    buik1 = voor - P.OOG_ONDER_RAND - P.OOG_GROOT[1] - 5.0
    assert buik1 - buik0 >= 14.0, "te laag voor ogen en buik: kies een hogere voorkant"
    P.VARIANTEN["eigen"] = ("eigen", "custom", float(achter), float(voor), (buik0, min(buik1, buik0 + 42.0)))
    PRINT.mkdir(parents=True, exist_ok=True)
    stl(f"poepmonster_eigen_{achter:g}", M.monster("eigen"), {})
    print(f"  inhoud tot de achterrand: {M.inhoud_cm3('eigen'):.0f} cm³")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "eigen":
        eigen(*[float(a) for a in args[1:3]])
    else:
        main()
