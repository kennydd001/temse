"""Printligging en plaatindeling van de pasproef van E-01 (het valbord): één kleur per plaat.

  plaat 1  Charcoal     passtrip: eerst printen en de gaten kiezen (gatenstrip, pennenblok, beugelproef)
  plaat 2  Charcoal     achterplaat (pinnen omhoog)
  plaat 3  Charcoal     voorplaat met tralies (binnenkant op het bed, pennen voor de beugel omhoog)
  plaat 4  Latte Brown  voet met de twee bakjes, beugel voor het mes
  plaat 5  Maroon Red   slinger, mesblok en kantelplank
  plaat 6  Marine Blue  10 diamantmuntjes

Alles print zonder support: achter- en voorplaat liggen plat, de slinger ligt met de achterkant van de arm
op het bed (pennetjes en gewicht omhoog), het mesblok op zijn achterkant (het mes staat rechtop), de beugel
staat op zijn poten (de tunnel voor de arm loopt dan verticaal), de plank staat met de naaf omhoog.
"""
import numpy as np

import params as P
import valbord as V

Y_OP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float)     # y wordt de hoogte

LIGGING = {"achterplaat": Y_OP, "voorplaat": Y_OP, "voorplaat glad": Y_OP, "plank": Y_OP, "slinger": Y_OP, "mesblok": Y_OP}
AANTAL = {"gatenstrip": 1, "pennenblok": 1, "beugelproef": 1, "achterplaat": 1, "voorplaat": 1, "beugel": 1, "voet": 1, "slinger": 1, "mesblok": 1, "plank": 1,
          "munt": P.AANTAL_MUNTEN}


def op_bed(m, x, y):
    b = m.bounding_box()
    return m.translate([x - (b[0] + b[3]) / 2, y - (b[1] + b[4]) / 2, -b[2]])


def onderdeel(naam):
    if naam == "munt":
        return V.munt()
    if naam == "voorplaat glad":
        return V.voorplaat(met_tralies=False)
    if naam in ("gatenstrip", "pennenblok", "beugelproef"):
        return getattr(V, naam)()
    return V.delen()[naam][1]


def stuk(naam, x, y):
    m = onderdeel(naam)
    T = LIGGING.get(naam)
    return op_bed(m.transform(T) if T is not None else m, x, y)


def platen():
    munten = [(f"munt {k + 1}", stuk("munt", 68 + (k % 5) * 30, 113 + (k // 5) * 30)) for k in range(P.AANTAL_MUNTEN)]
    return [("1 passtrip", P.ZWART, [("gatenstrip", stuk("gatenstrip", 100, 128)), ("pennenblok", stuk("pennenblok", 165, 140)),
                                     ("beugelproef", stuk("beugelproef", 165, 112))]),
            ("2 achterplaat", P.ZWART, [("achterplaat", stuk("achterplaat", 128, 128))]),
            ("3 voorplaat", P.ZWART, [("voorplaat", stuk("voorplaat", 128, 128))]),
            ("4 voet en beugel", P.BRUIN, [("voet", stuk("voet", 128, 110)), ("beugel", stuk("beugel", 128, 185))]),
            ("5 slinger, mesblok en plank", P.ROOD, [("slinger", stuk("slinger", 100, 128)), ("mesblok", stuk("mesblok", 170, 90)),
                                                     ("plank", stuk("plank", 170, 150))]),
            ("6 munten", P.BLAUW, munten)]
