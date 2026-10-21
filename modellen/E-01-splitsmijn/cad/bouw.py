"""Schrijft de printbestanden van E-01 Splitsmijn: het eindmodel en de pasproef.

Gebruik:  python bouw.py

  print/      het eindmodel: elk onderdeel als STL in printligging (in één stuk, ook de meerkleurige) en
              splitsmijn.3mf (Bambu Studio, 9 platen; plaat 1 is de passtrip om de gaten te kiezen).
  pasproef/   elk onderdeel als STL in printligging, en pasproef.3mf (Bambu Studio, 6 platen, 1 kleur per plaat;
              plaat 1 is de passtrip om de gaten te kiezen).
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
EIND = ROOT / "print"

# instellingen per object (bovenop "0.20mm Standard @BBL A1": 2 wanden, 15 % infill; gyroid zet project.py)
INST = {
    "achterplaat": {"wall_loops": "3"},
    "voorplaat": {"wall_loops": "2"},
    "voet": {"wall_loops": "3"},
    "slinger": {"wall_loops": "3", "sparse_infill_density": "100%"},   # zwaar gewicht: slingert langer en egaler
    "plank": {"wall_loops": "4"},
    "munt": {"wall_loops": "3", "brim_type": "no_brim"},
    # eindmodel
    "mijnhout": {"wall_loops": "2"},
    "mijnberg": {"wall_loops": "3"},
    "tienveld": {"wall_loops": "2"},
    "portaal": {"wall_loops": "3"},
    "steel": {"wall_loops": "3", "sparse_infill_density": "100%"},   # de hele slinger massief, zoals de pasproef
    "kop": {"wall_loops": "3", "sparse_infill_density": "100%"},
    "mesblok": {"wall_loops": "3", "sparse_infill_density": "100%"},
    "verdelervin": {"wall_loops": "3"},
    "karretje": {"wall_loops": "2"},
    "kokervoet": {"wall_loops": "3"},
    "kokerkap": {"wall_loops": "3"},
    "gatenstrip": {"wall_loops": "2"},          # zoals de voorplaat
    "pennenblok": {"wall_loops": "3"},          # zoals de achterplaat
    "beugelproef": {"wall_loops": "2"},          # zoals de beugel (standaard)
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


def schrijf_stl(naam, m, info, map_=PROEF, aantal=None):
    pad = map_ / f"{naam.replace(' ', '_')}.stl"
    save_stl(m, pad)
    tm = trimesh.load(pad)
    assert tm.is_watertight and tm.is_winding_consistent, pad
    info[str(pad.relative_to(ROOT))] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                        "maat_mm": [round(float(v), 1) for v in tm.extents],
                                        "aantal": aantal if aantal is not None else platen.AANTAL.get(naam, 1)}
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


def project_eind(pad):
    """Bambu-project van het eindmodel: meerkleurige onderdelen zijn één object met een deel per kleur."""
    codes = [P.ZWART, P.BRUIN, P.TERRA, P.GEEL, P.ROOD, P.BLAUW]
    slots = {c: i + 1 for i, c in enumerate(codes)}
    fil = [B.Filament(P.FILAMENTEN_EIND[c][2], P.FILAMENTEN_EIND[c][1], f"{P.FILAMENTEN_EIND[c][0]} {c}") for c in codes]
    bp = []
    for naam, objecten in platen.platen_eind():
        objs = []
        for n, delen in objecten:
            ds = []
            for dn, kleur, m in delen:
                tm = to_trimesh(m.simplify(1e-3))
                ds.append(B.Deel(dn, tm.vertices.copy(), tm.faces.copy(), slots[kleur]))
            b = platen.manifold3d_unie([m for *_, m in delen]).bounding_box()
            objs.append(B.Object(n, ds, inst(n), ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2)))
        bp.append(B.Plaat(naam, objs))
    B.plaats_torens(fil, bp)
    for pl in bp:
        pl.miniatuur = miniatuur(pl, fil)
    return B.schrijf(pad, f"E-01 De Splitsmijn {P.VERSIE_EIND}", fil, bp,
                     beschrijving="De Splitsmijn: leren splitsen tot 10 (1ste leerjaar). Gooi diamantmuntjes in de mijn, "
                                  "geef de houweel een duw, en de muntjes verdwijnen in twee mijnkarretjes. Trek er een uit: "
                                  "hoeveel zitten er in de andere? Print eerst plaat 1 (passtrip) en kies de gaten.")


def main():
    info = {}
    print("print/")
    EIND.mkdir(parents=True, exist_ok=True)
    gedaan = set()
    for _, objecten in platen.platen_eind():
        for n, delen in objecten:
            k = sleutel(n)
            if k not in gedaan and k not in ("gatenstrip", "pennenblok", "beugelproef"):
                gedaan.add(k)
                m = platen.manifold3d_unie([m for *_, m in delen])
                schrijf_stl(k, platen.op_bed(m, 0, 0), info, EIND, platen.AANTAL_EIND.get(k, 1))
    r = project_eind(EIND / "splitsmijn.3mf")
    print("  print/splitsmijn.3mf", r)

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
                "10 muntjes. Print eerst plaat 1 (passtrip) en kies de gaten. Test of de muntjes nergens blijven "
                "hangen en of de splitsingen echt willekeurig zijn.")
    print("  pasproef/pasproef.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
