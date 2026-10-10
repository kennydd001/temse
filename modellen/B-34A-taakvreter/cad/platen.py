"""Printligging en plaatindeling van B-34A: elk onderdeel in één kleur, één kleur per plaat.

  plaat 1  Orange    voorhelft, kruk, 10 paspennen (8 + 2 reserve)
  plaat 2  Orange    achterhelft
  plaat 3  Ivory     lade, ogen, tanden, knop, twee hoorntjes
  plaat 4  Charcoal  beide rollen, nokas, drijfstang, tuimelaar, mondstuk, twee pupillen

Printligging (alles zonder support):
  voorhelft     gezicht op het bed: alles binnenin groeit vanaf het gezicht; de halve gaten liggen bovenop
  achterhelft   achterkant op het bed: de glijgoot is een schuin plafond dat naar boven kijkt
  rollen        staand (precies rond): onderste op het tandwiel, bovenste op het linkereind
  kruk, knop    staand op de buitenkant; nokas, tuimelaar, drijfstang plat met de pen omhoog
  ogen          staand op de rechtertrommel
  mondstuk      met de lip op het bed; tanden en pupillen met de voorkant op het bed
"""
import numpy as np

import params as P
import vreter as V

Y_OP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float)     # y wordt de hoogte
Y_NEER = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)   # -y wordt de hoogte
X_OP = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)     # x wordt de hoogte
X_NEER = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=float)   # -x wordt de hoogte

LIGGING = {
    "romp voor": Y_OP, "romp achter": Y_NEER, "mondstuk": Y_OP, "tanden": Y_OP, "pupil": Y_OP,
    "rol onder": X_NEER, "kruk": X_NEER, "knop": X_NEER, "ogen": X_NEER,
    "rol boven": X_OP, "nokas": X_OP, "tuimelaar": X_OP, "drijfstang": X_OP,
}

AANTAL = {"romp voor": 1, "romp achter": 1, "kruk": 1, "paspen": 10, "lade": 1, "ogen": 1, "tanden": 1,
          "knop": 1, "hoorn": 2, "rol onder": 1, "rol boven": 1, "nokas": 1, "drijfstang": 1, "tuimelaar": 1,
          "mondstuk": 1, "pupil": 2}


def op_bed(m, x, y):
    """Leg een Manifold met de onderkant op z = 0 en het midden van zijn voetafdruk op (x, y)."""
    b = m.bounding_box()
    return m.translate([x - (b[0] + b[3]) / 2, y - (b[1] + b[4]) / 2, -b[2]])


def plat(naam, m):
    """Printligging van een onderdeel (gebouwd rond zijn eigen as of in de montage)."""
    sleutel = naam.split(" ")[0] if naam.startswith(("pupil", "hoorn", "paspen")) else naam
    T = LIGGING.get(sleutel)
    return m.transform(T) if T is not None else m


def onderdeel(naam):
    """Het onderdeel zoals het gebouwd is (zonder stand)."""
    if naam == "pupil":
        return V.pupil(P.OOG_X)
    if naam == "hoorn":
        return V.hoorn()
    if naam == "paspen":
        return V.paspen()
    return V.delen()[naam][1]


def stuk(naam, x, y):
    return op_bed(plat(naam, onderdeel(naam)), x, y)


def platen():
    """[(plaatnaam, kleur, [(onderdeelnaam, Manifold op het bed)])]"""
    oranje = [("romp voor", stuk("romp voor", 72, 128)), ("kruk", stuk("kruk", 172, 150))]
    oranje += [(f"paspen {k + 1}", stuk("paspen", 200 + (k % 5) * 10, 64 + (k // 5) * 10)) for k in range(10)]
    achter = [("romp achter", stuk("romp achter", 128, 128))]
    wit = [("lade", stuk("lade", 70, 72)), ("ogen", stuk("ogen", 165, 60)), ("tanden", stuk("tanden", 70, 135)),
           ("knop", stuk("knop", 205, 60)), ("hoorn 1", stuk("hoorn", 165, 100)), ("hoorn 2", stuk("hoorn", 195, 100))]
    zwart = [("rol onder", stuk("rol onder", 40, 60)), ("rol boven", stuk("rol boven", 70, 60)),
             ("nokas", stuk("nokas", 100, 60)), ("drijfstang", stuk("drijfstang", 150, 60)),
             ("tuimelaar", stuk("tuimelaar", 190, 60)), ("mondstuk", stuk("mondstuk", 128, 120)),
             ("pupil 1", stuk("pupil", 215, 60)), ("pupil 2", stuk("pupil", 230, 60))]
    return [("1 voorhelft, kruk en paspennen", P.ORANJE, oranje), ("2 achterhelft", P.ORANJE, achter),
            ("3 lade, ogen en witte delen", P.WIT, wit), ("4 rollen en zwarte delen", P.ZWART, zwart)]
