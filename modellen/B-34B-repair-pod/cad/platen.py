"""Printligging en plaatindeling van B-34B: elk onderdeel in één kleur, één kleur per plaat.

  plaat 1  Charcoal  voet, dak
  plaat 2  Ivory     twee bakjes
  plaat 3  Orange    acht armen, de schuif en 18 klikpennen (16 + 2 reserve)

Printligging:
  voet, dak, bakken   rechtop zoals in de montage (voet- en dakonderkant, bakbodem op het bed)
  armen, schuif       plat: het x-z-vlak van het mechanisme op het bed (tandjes en ogen op hun mooist)
  pennen              staand op de kop, zodat de schacht precies rond is
"""
import numpy as np

import pod
from params import BAK_Z0, DAK_Z0, ORANJE, WIT, ZWART

R_VOOR = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)    # y -> -z: voorarm plat
R_ACHTER = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=float)  # y -> z: achterarm plat


def op_bed(m, x, y):
    """Leg een Manifold met de onderkant op z = 0 en het midden van zijn voetafdruk op (x, y)."""
    b = m.bounding_box()
    return m.translate([x - (b[0] + b[3]) / 2, y - (b[1] + b[4]) / 2, -b[2]])


def plat(naam, m):
    """Printligging van een onderdeel (nog niet op zijn plek op de plaat)."""
    if naam == "dak":
        return m.translate([0, 0, -DAK_Z0])
    if naam.startswith("bak"):
        return m.translate([0, 0, -BAK_Z0])
    if "arm" in naam and "voor" in naam:
        return m.transform(R_VOOR)
    if "arm" in naam or naam == "schuif":       # schuif: voorkant op het bed, zwaluwstaart en knop omhoog
        return m.transform(R_ACHTER)
    return m


def pen_plat():
    """Pen staand op zijn kop (as verticaal, kop onder)."""
    m = pod.pen()                                    # as langs +y, kop bij y = [-KOP_T, 0]
    m = m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))   # y -> z
    return m


def platen():
    """[(plaatnaam, kleur, [(onderdeelnaam, Manifold op het bed)])]

    Gespiegelde platte delen zijn dezelfde delen omgedraaid: alle zes gewone armen zijn gelijk, beide
    tandarmen zijn gelijk (één wordt omgekeerd gemonteerd) en beide bakjes zijn gelijk.
    """
    d = pod.onderdelen()
    zwart = [("voet", op_bed(plat("voet", d["voet"][1]), 66, 128)),
             ("dak", op_bed(plat("dak", d["dak"][1]), 190, 128))]
    bak = plat("bak rechts", d["bak rechts"][1])
    wit = [("bak 1", op_bed(bak, 98, 128)), ("bak 2", op_bed(bak, 158, 128))]
    arm = plat("binnenarm achter rechts", d["binnenarm achter rechts"][1])
    tandarm = plat("binnenarm voor rechts", d["binnenarm voor rechts"][1])
    oranje = [(f"tandarm {i + 1}", op_bed(tandarm, 22 + i * 24, 60)) for i in range(2)]
    oranje += [(f"arm {i + 1}", op_bed(arm, 22 + (i % 4) * 22, 60 if i < 2 else 170)) for i in range(6)]
    oranje[2:4] = [(n, op_bed(arm, 70 + i * 20, 60)) for i, (n, _) in enumerate(oranje[2:4])]
    oranje.append(("schuif", op_bed(plat("schuif", d["schuif"][1]), 128, 128)))
    pen = pen_plat()
    for k in range(18):
        oranje.append((f"pen {k + 1}", op_bed(pen, 158 + (k % 6) * 14, 70 + (k // 6) * 14)))
    return [("1 voet en dak", ZWART, zwart), ("2 bakjes", WIT, wit), ("3 armen, schuif en pennen", ORANJE, oranje)]


# printset: onderdeel -> aantal (voor de README en de STL-bestanden)
AANTAL = {"voet": 1, "dak": 1, "bak": 2, "arm": 6, "tandarm": 2, "schuif": 1, "pen": 18}
