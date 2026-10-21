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


# === eindmodel ======================================================================================
#   plaat 1  Charcoal          passtrip (zoals bij de pasproef)
#   plaat 2  Charcoal          achterplaat (dezelfde als bij de pasproef)
#   plaat 3  Charcoal          voorplaat (pennen voor de beugel 2 mm langer)
#   plaat 4  Latte Brown       mijnhout, met de voorkant op het bed (de nokjes achterop wijzen omhoog)
#   plaat 5  Terracotta        mijnberg (ondersteboven: de tunnels zijn open naar boven), tienveld
#   plaat 6  Latte Brown + Charcoal   portaal met de naam, steel, beugel, mesblok, kokervoeten (rechtop, prop op het
#                                     bed), kokerkappen (op hun buitenkant)
#   plaat 7  Sunflower Yellow  houweelkop, kantelplank, verdelervin
#   plaat 8  Maroon Red        twee karretjes
#   plaat 9  Marine Blue + Sunflower Yellow   10 munten, onderste helft geel
Y_NEER = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)     # -y wordt de hoogte
OM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], dtype=float)        # ondersteboven
X_NEER = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=float)     # +x op het bed
X_OP = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)       # -x op het bed
LIGGING_EIND = {"achterplaat": Y_OP, "voorplaat": Y_OP, "mijnhout": Y_NEER, "mijnberg": OM,
                "kokerkap L": X_NEER, "kokerkap R": X_OP,
                "portaal": Y_OP, "steel": Y_OP, "kop": Y_OP,
                "mesblok": Y_OP, "plank": Y_OP, "verdelervin": Y_OP}
AANTAL_EIND = {"munt": P.AANTAL_MUNTEN}


def manifold3d_unie(ms):
    from manifold3d import Manifold, OpType
    return Manifold.batch_boolean(list(ms), OpType.Add) if len(ms) > 1 else ms[0]


def onderdeel_eind(naam):
    """[(deelnaam, kleur, Manifold)] van een onderdeel van het eindmodel; meerkleurig: meer dan één deel."""
    import mijn as M
    if naam == "munt":
        geel, blauw = M.munt_delen()
        return [("geel", P.GEEL, geel), ("blauw", P.BLAUW, blauw)]
    if naam == "portaal":
        return [(n, k, m) for n, (k, m) in M.portaal_delen().items()]
    if naam == "tienveld":
        return [(naam, P.TERRA, M.tienveld())]
    if naam in ("gatenstrip", "pennenblok", "beugelproef"):
        return [(naam, P.ZWART, getattr(V, naam)())]
    kleur, m, _ = M.delen()[naam]
    return [(naam, kleur, m)]


def stuk_eind(naam, x, y):
    """Onderdeel in printligging op (x, y): [(deelnaam, kleur, Manifold)], alle delen samen verschoven."""
    delen = onderdeel_eind(naam)
    T = LIGGING_EIND.get(naam)
    if T is not None:
        delen = [(n, k, m.transform(T)) for n, k, m in delen]
    b = manifold3d_unie([m for *_, m in delen]).bounding_box()
    d = [x - (b[0] + b[3]) / 2, y - (b[1] + b[4]) / 2, -b[2]]
    return [(n, k, m.translate(d)) for n, k, m in delen]


def platen_eind():
    """[(plaatnaam, [(objectnaam, [(deelnaam, kleur, Manifold)])])]"""
    munten = [(f"munt {k + 1}", stuk_eind("munt", 68 + (k % 5) * 30, 113 + (k // 5) * 30)) for k in range(P.AANTAL_MUNTEN)]
    return [
        ("1 passtrip", [("gatenstrip", stuk_eind("gatenstrip", 100, 128)), ("pennenblok", stuk_eind("pennenblok", 165, 140)),
                        ("beugelproef", stuk_eind("beugelproef", 165, 112))]),
        ("2 achterplaat", [("achterplaat", stuk_eind("achterplaat", 128, 128))]),
        ("3 voorplaat", [("voorplaat", stuk_eind("voorplaat", 128, 128))]),
        ("4 mijnhout", [("mijnhout", stuk_eind("mijnhout", 128, 128))]),
        ("5 mijnberg en tienveld", [("mijnberg", stuk_eind("mijnberg", 128, 70)),
                                    ("tienveld", stuk_eind("tienveld", 128, 190))]),
        ("6 portaal, steel, beugel, mesblok, kokervoeten en -kappen", [
            ("portaal", stuk_eind("portaal", 128, 222)), ("steel", stuk_eind("steel", 232, 105)),
            ("beugel", stuk_eind("beugel", 150, 60)), ("mesblok", stuk_eind("mesblok", 150, 110)),
            ("kokervoet L", stuk_eind("kokervoet L", 95, 150)), ("kokervoet R", stuk_eind("kokervoet R", 125, 150)),
            ("kokerkap L", stuk_eind("kokerkap L", 95, 180)), ("kokerkap R", stuk_eind("kokerkap R", 125, 180))]),
        ("7 houweelkop, plank, verdelervin", [("kop", stuk_eind("kop", 128, 150)), ("plank", stuk_eind("plank", 100, 95)),
                                              ("verdelervin", stuk_eind("verdelervin", 160, 95))]),
        ("8 karretjes", [("karretje L", stuk_eind("karretje L", 75, 128)), ("karretje R", stuk_eind("karretje R", 181, 128))]),
        ("9 munten", munten),
    ]
