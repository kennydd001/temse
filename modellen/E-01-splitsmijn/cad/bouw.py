"""Schrijft de printbestanden van de pasproef van E-01 Splitsmijn: het valbord op ware grootte.

Gebruik:  python bouw.py

  pasproef/   elk onderdeel als STL in printligging, en pasproef.3mf (Bambu Studio, 5 platen, 1 kleur per plaat).
              voorplaat_glad.stl is dezelfde voorplaat zonder kijkspleten, voor doorzichtig filament.
"""
import json
import re
from pathlib import Path

import trimesh

import _pad  # noqa: F401
import params as P
import platen
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import save_stl, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PROEF = ROOT / "pasproef"

# instellingen per object (bovenop "0.20mm Standard @BBL A1": 2 wanden, 15 % grid)
INST = {
    "achterplaat": {"wall_loops": "3"},
    "voorplaat": {"wall_loops": "2"},
    "voet": {"wall_loops": "3"},
    "slinger": {"wall_loops": "3", "sparse_infill_density": "100%"},   # zwaar gewicht: slingert langer en egaler
    "plank": {"wall_loops": "4"},
    "munt": {"wall_loops": "3", "brim_type": "no_brim"},
}


def inst(naam):
    for k, v in INST.items():
        if naam.startswith(k):
            return dict(v)
    return {}


def sleutel(naam):
    return re.sub(r" \d+$", "", naam)


def filamenten(codes):
    return [B.Filament(P.FILAMENTEN[c][2], P.FILAMENTEN[c][1], f"{P.FILAMENTEN[c][0]} {c}") for c in codes]


def schrijf_stl(naam, m, info):
    pad = PROEF / f"{naam.replace(' ', '_')}.stl"
    save_stl(m, pad)
    tm = trimesh.load(pad)
    assert tm.is_watertight and tm.is_winding_consistent, pad
    info[str(pad.relative_to(ROOT))] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents],
                                        "aantal": platen.AANTAL.get(naam, 1)}
    print(f"  {str(pad.relative_to(ROOT)):34s} {[round(float(v), 1) for v in tm.extents]}")


def project(pad, titel, plaatlijst, codes, beschrijving):
    slots = {c: i + 1 for i, c in enumerate(codes)}
    fil = filamenten(codes)
    bp = []
    for naam, kleur, delen in plaatlijst:
        objs = []
        for n, m in delen:
            tm = to_trimesh(m.simplify(1e-3))
            b = m.bounding_box()
            objs.append(B.Object(n, [B.Deel(n, tm.vertices.copy(), tm.faces.copy(), slots[kleur])], inst(n),
                                 ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2)))
        pl = B.Plaat(naam, objs)
        pl.miniatuur = miniatuur(pl, fil)
        bp.append(pl)
    return B.schrijf(pad, titel, fil, bp, beschrijving=beschrijving)


def main():
    info = {}
    print("pasproef/")
    PROEF.mkdir(parents=True, exist_ok=True)
    gedaan = set()
    for _, _, delen in platen.platen():
        for n, m in delen:
            k = sleutel(n)
            if k not in gedaan:
                gedaan.add(k)
                schrijf_stl(k, platen.op_bed(m, 0, 0), info)
    schrijf_stl("voorplaat glad", platen.stuk("voorplaat glad", 0, 0), info)
    r = project(PROEF / "pasproef.3mf", f"E-01 Splitsmijn pasproef {P.VERSIE}", platen.platen(),
                [P.ZWART, P.BRUIN, P.ROOD, P.BLAUW],
                "Pasproef van de Splitsmijn: het valbord op ware grootte met de slinger, een voet met twee bakjes en "
                "10 muntjes. Test of de muntjes nergens blijven hangen en of de splitsingen echt willekeurig zijn.")
    print("  pasproef/pasproef.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
