"""Schrijft de printbestanden van B-34A Taakvreter.

Gebruik:  python bouw.py

  print/      elk onderdeel als STL in printligging, en taakvreter.3mf (Bambu Studio, 4 platen, 1 kleur per plaat)
  pasproef/   eerst printen, pasproef.3mf met twee platen: (1) het afgeschermde invoerrol-proefstuk (de romp
              rond de rollen, met deksel); (2) beide rollen, kruk, knop, nokas, mondstuk, paspennen en een
              paspenstrip. Daarmee test je papiersoorten, scheef invoeren, slip, terugdraaien en vingers.
"""
import json
import re
from pathlib import Path

import trimesh

import _pad  # noqa: F401
import params as P
import platen
import vreter as V
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import save_stl, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PRINT = ROOT / "print"
PROEF = ROOT / "pasproef"

# instellingen per object (bovenop "0.20mm Standard @BBL A1": 2 wanden, 15 % infill; gyroid zet project.py)
BRIM = {"brim_type": "outer_only", "brim_width": "5"}
INST = {
    "romp": {"wall_loops": "3"},
    "proefframe": {"wall_loops": "3"},
    "lade": {"wall_loops": "3"},
    "mondstuk": {"wall_loops": "3"},
    "rol": {"wall_loops": "4", **BRIM},                       # staand en 105 mm hoog: brim tegen omvallen
    "ogen": {"wall_loops": "3", **BRIM},
    "kruk": {"wall_loops": "4", "sparse_infill_density": "40%"},
    "nokas": {"wall_loops": "4"},
    "drijfstang": {"wall_loops": "4"},
    "tuimelaar": {"wall_loops": "4"},
    "knop": {"wall_loops": "3"},
    "paspen": {"wall_loops": "4", "brim_type": "no_brim"},   # Ø5: met 4 wanden volledig massief
}


def inst(naam):
    for k, v in INST.items():
        if naam.startswith(k):
            return dict(v)
    return {}


def sleutel(naam):
    """'paspen 3' -> 'paspen', 'romp voor' blijft 'romp voor': één STL per uniek onderdeel."""
    return re.sub(r" \d+$", "", naam)


def bestandsnaam(naam):
    return naam.replace(" ", "_")


def filamenten(codes):
    return [B.Filament(P.FILAMENTEN[c][2], P.FILAMENTEN[c][1], f"{P.FILAMENTEN[c][0]} {c}") for c in codes]


def check(path):
    tm = trimesh.load(path)
    assert tm.is_watertight and tm.is_winding_consistent, path
    return tm


def schrijf_stl(naam, m, map_, info):
    pad = map_ / f"{bestandsnaam(naam)}.stl"
    save_stl(m, pad)
    tm = check(pad)
    info[str(pad.relative_to(ROOT))] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents]}
    print(f"  {str(pad.relative_to(ROOT)):42s} {[round(float(v), 1) for v in tm.extents]}")


def project(pad, titel, plaatlijst, codes, beschrijving):
    """plaatlijst: [(plaatnaam, kleur, [(naam, Manifold op het bed)])] -> Bambu-3MF, één filament per kleur."""
    slots = {c: i + 1 for i, c in enumerate(codes)}
    fil = filamenten(codes)
    bp = []
    for naam, kleur, delen in plaatlijst:
        objs = []
        for n, m in delen:
            tm = to_trimesh(m.simplify(1e-3))
            b = m.bounding_box()
            objs.append(B.Object(n, [B.Deel(n, tm.vertices.copy(), tm.faces.copy(), slots.get(kleur, 1))], inst(n),
                                 ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2)))
        pl = B.Plaat(naam, objs)
        pl.miniatuur = miniatuur(pl, fil)
        bp.append(pl)
    return B.schrijf(pad, titel, fil, bp, beschrijving=beschrijving)


# --- pasproef -----------------------------------------------------------------------------------

def pasproef_platen():
    """Twee platen in één kleur naar keuze: het proefframe, en wat erin hoort."""
    voor, achter = V.proefframe()
    frame = [("proefframe voor", platen.op_bed(voor.transform(platen.Y_OP), 128, 70)),
             ("proefframe achter", platen.op_bed(achter.transform(platen.Y_NEER), 128, 165))]
    s = platen.stuk
    delen = [("rol onder", s("rol onder", 40, 70)), ("rol boven", s("rol boven", 70, 70)),
             ("nokas", s("nokas", 100, 70)), ("kruk", s("kruk", 150, 70)), ("knop", s("knop", 195, 70)),
             ("mondstuk", s("mondstuk", 128, 130)), ("paspenstrip", platen.op_bed(V.paspenstrip(), 128, 175))]
    delen += [(f"paspen {k + 1}", s("paspen", 185 + (k % 3) * 10, 100 + (k // 3) * 10)) for k in range(6)]
    return [("1 proefframe", P.ORANJE, frame), ("2 rollen, kruk en paspennen", P.ORANJE, delen)]


def main():
    info = {}
    print("print/")
    PRINT.mkdir(parents=True, exist_ok=True)
    gedaan = set()
    for _, _, delen in platen.platen():
        for n, m in delen:
            k = sleutel(n)
            if k in gedaan:
                continue
            gedaan.add(k)
            schrijf_stl(k, platen.op_bed(m, 0, 0), PRINT, info)
            info[f"print/{bestandsnaam(k)}.stl"]["aantal"] = platen.AANTAL[k]
    r = project(PRINT / "taakvreter.3mf", f"B-34A Taakvreter {P.VERSIE}", platen.platen(), [P.ORANJE, P.WIT, P.ZWART],
                "B-34A Taakvreter: bureaumonster dat met een slinger je afgewerkte taakbriefjes heel opeet. Twee "
                "invoerrollen nemen een memoblaadje mee naar de lade in de buik; de ogen rollen mee. Plaat 1 en 2 "
                "Pumpkin Orange, plaat 3 Ivory White, plaat 4 Charcoal. Nodig: 3 O-ringen 14 x 2 mm (NBR).")
    print("  print/taakvreter.3mf", r)
    print("pasproef/")
    PROEF.mkdir(parents=True, exist_ok=True)
    for _, _, delen in pasproef_platen():
        for n, m in delen:
            if sleutel(n) in gedaan:                  # zelfde onderdelen als in print/
                continue
            schrijf_stl(n, platen.op_bed(m, 0, 0), PROEF, info)
    r = project(PROEF / "pasproef.3mf", f"B-34A Taakvreter pasproef {P.VERSIE}", pasproef_platen(), [P.ORANJE],
                "Pasproef: plaat 1 het afgeschermde invoerrol-proefstuk, plaat 2 de rollen, kruk, nokas, mondstuk, "
                "paspennen en een paspenstrip. Eén kleur, kies zelf. Nodig: 3 O-ringen 14 x 2 mm.")
    print("  pasproef/pasproef.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
