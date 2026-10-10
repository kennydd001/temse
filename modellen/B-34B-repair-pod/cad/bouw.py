"""Schrijft de printbestanden van B-34B Repair Pod.

Gebruik:  python bouw.py

  print/      elk onderdeel als STL in printligging, en repair-pod.3mf (Bambu Studio, 3 platen, 1 kleur per plaat)
  pasproef/   eerst printen, pasproef.3mf met twee platen: (1) scharnierstrip (gaten in wand en arm in drie
              maten), bitblokje en vier pennen; (2) één zijmechanisme (halve voet, één bakje, tandarm,
              drie armen, tien pennen en de schuif)
"""
import json
from pathlib import Path

import trimesh
from shapely.geometry import Point

import _pad  # noqa: F401
import params as P
import platen
import pod
from gereedschap.bambu import project as B
from gereedschap.bambu.miniatuur import miniatuur
from gereedschap.geom import box3, extrude, rounded_rect, save_stl, text, to_trimesh

ROOT = Path(__file__).resolve().parent.parent
PRINT = ROOT / "print"
PROEF = ROOT / "pasproef"

# instellingen per object (bovenop "0.20mm Standard @BBL A1": 2 wanden, 15 % grid)
INST = {
    "voet": {"wall_loops": "3"},
    "dak": {"wall_loops": "3"},
    "bak": {"wall_loops": "3"},                                   # wanden van 2,4 mm helemaal massief
    "arm": {"wall_loops": "3", "sparse_infill_density": "40%"},
    "schuif": {"wall_loops": "3", "sparse_infill_density": "40%"},
    "pen": {"wall_loops": "4", "brim_type": "no_brim"},            # Ø5: met 4 wanden volledig massief
}


def inst(naam):
    for k, v in INST.items():
        if naam.startswith(k) or k in naam:
            return dict(v)
    return {}


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
            objs.append(B.Object(n, [B.Deel(n, tm.vertices.copy(), tm.faces.copy(), slots[kleur])], inst(n),
                                 ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2)))
        pl = B.Plaat(naam, objs)
        pl.miniatuur = miniatuur(pl, fil)
        bp.append(pl)
    return B.schrijf(pad, titel, fil, bp, beschrijving=beschrijving)


# --- pasproef -----------------------------------------------------------------------------------

def label(s, h, cx, cy, z, diep=0.6):
    return extrude(text(s, h, cx, cy), z - diep, z + 0.01)


def scharnierstrip():
    """Wandstrip rechtop (gaten liggend, zoals in voet en bak) en armstrip plat (gaten staand, zoals in de armen)."""
    maten_wand = (4.95, 5.05, 5.15)
    maten_arm = (5.15, 5.25, 5.35)
    # wand: 3 mm dik, staat op een voetje waar de maten in staan
    wand = box3(0, 0, 0, 54, P.WAND_VOOR, 16) + box3(0, -7, 0, 54, P.WAND_VOOR + 7, 2)
    for i, d in enumerate(maten_wand):
        x = 9 + i * 18
        wand = wand - pod.xz(pod.druppel((x, 10.0), d / 2, omhoog=True), -1, P.WAND_VOOR + 1)   # zoals in voet en bak
        wand = wand - label(f"{d:.2f}".replace(".", ","), 3.0, x, -3.8, 2.0)
    # arm: 4,5 mm dik, gaten boven, maten eronder
    arm = extrude(rounded_rect(0, 0, 54, 18, 3), 0, P.ARM_T)
    for i, d in enumerate(maten_arm):
        x = 9 + i * 18
        arm = arm - extrude(Point(x, 12).buffer(d / 2, quad_segs=24), -1, P.ARM_T + 1)
        arm = arm - label(f"{d:.2f}".replace(".", ","), 3.0, x, 4.5, P.ARM_T, 0.5)
    return wand, arm


def bitblok():
    m = extrude(rounded_rect(0, 0, 44, 18, 3), 0, 14)
    for i, sw in enumerate((6.4, 6.6, 6.8)):
        x = 8 + i * 14
        m = m - extrude(pod.zeskant(x, 10, sw), 14 - P.BIT_DIEP, 15)
    for i, sw in enumerate((6.4, 6.6, 6.8)):
        m = m - label(f"{sw:.1f}".replace(".", ","), 2.2, 8 + i * 14, 2.2, 14, 0.5)   # bovenop, vóór het gat
    return m


def proefvoet():
    """Rechterhelft van de voet met de hele middenrug: genoeg voor één zijmechanisme."""
    v = pod.voet()
    return v ^ box3(-P.RUG_HALF - 3, -20, -1, 80, P.DIEPTE + 20, 120)


def pasproef_platen():
    """Twee platen: eerst de snelle proef (scharnierstrip, bitblok, 4 pennen), dan één zijmechanisme met de schuif."""
    d = pod.onderdelen()
    wand, arm_strip = scharnierstrip()
    arm = platen.plat("binnenarm achter rechts", d["binnenarm achter rechts"][1])
    tandarm = platen.plat("binnenarm voor rechts", d["binnenarm voor rechts"][1])
    pen = platen.pen_plat()
    snel = [("scharnierstrip wand", platen.op_bed(wand, 128, 100)),
            ("scharnierstrip arm", platen.op_bed(arm_strip, 128, 130)),
            ("bitblok", platen.op_bed(bitblok(), 128, 158))]
    snel += [(f"pen {k + 1}", platen.op_bed(pen, 100 + k * 14, 75)) for k in range(4)]
    zij = [("proefvoet", platen.op_bed(proefvoet(), 52, 80)),
           ("bak", platen.op_bed(platen.plat("bak rechts", d["bak rechts"][1]), 52, 190)),
           ("tandarm", platen.op_bed(tandarm, 112, 80))]
    zij += [(f"arm {i + 1}", platen.op_bed(arm, 132 + i * 20, 80)) for i in range(3)]
    zij += [(f"pen {k + 1}", platen.op_bed(pen, 120 + (k % 5) * 14, 150 + (k // 5) * 14)) for k in range(10)]
    zij += [("schuif", platen.op_bed(platen.plat("schuif", d["schuif"][1]), 212, 80))]
    return [("1 scharnierstrip, bitblok en pennen", P.ORANJE, snel), ("2 zijmechanisme", P.ORANJE, zij)]


def main():
    info = {}
    print("print/")
    PRINT.mkdir(parents=True, exist_ok=True)
    gedaan = set()
    for _, _, delen in platen.platen():
        for n, m in delen:
            sleutel = n.split(" ")[0]                    # "arm 3" -> "arm": één STL per uniek onderdeel
            if sleutel in gedaan:
                continue
            gedaan.add(sleutel)
            schrijf_stl(sleutel, platen.op_bed(m, 0, 0), PRINT, info)
            info[f"print/{sleutel}.stl"]["aantal"] = platen.AANTAL[sleutel]
    r = project(PRINT / "repair-pod.3mf", f"B-34B Repair Pod {P.VERSIE}", platen.platen(), [P.ZWART, P.WIT, P.ORANJE],
                "B-34B Repair Pod: capsule die met één schuif opent tot een werkplek met twee schroevenbakjes "
                "die horizontaal blijven. Plaat 1 Charcoal, plaat 2 Ivory White, plaat 3 Pumpkin Orange (18 pennen: 16 + 2 reserve).")
    print("  print/repair-pod.3mf", r)
    print("pasproef/")
    PROEF.mkdir(parents=True, exist_ok=True)
    for _, _, delen in pasproef_platen():
        for n, m in delen:
            if n.split(" ")[0] in gedaan:             # zelfde onderdelen als in print/
                continue
            schrijf_stl(n, platen.op_bed(m, 0, 0), PROEF, info)
    r = project(PROEF / "pasproef.3mf", f"B-34B Repair Pod pasproef {P.VERSIE}", pasproef_platen(), [P.ORANJE],
                "Pasproef: plaat 1 scharnierstrip, bitblokje en pennen (40 min); plaat 2 één zijmechanisme met bakje en schuif. Eén kleur, kies zelf.")
    print("  pasproef/pasproef.3mf", r)
    (ROOT / "onderdelen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
