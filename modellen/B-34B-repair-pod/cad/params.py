"""Alle maten van B-34B Repair Pod (mm, graden). Eén plek om aan te passen na de pasproef.

Assen in de montage: x = breedte (0 = midden), y = diepte (voorkant van bakken en voet op y = 0),
z = hoogte (tafel op z = 0). Het mechanisme werkt in het x-z-vlak; de armen zitten vóór (y < 0) en
achter (y > DIEPTE) de bakken.
"""
import math

VERSIE = "v0.1"

# --- filamenten (Bambu PLA, uit de kleurplanning: Charcoal, Pumpkin Orange, Ivory White) ---------
ZWART = "11101"     # voet en dak
ORANJE = "10301"    # armen, schuif, pennen
WIT = "11100"       # bakjes
FILAMENTEN = {
    ZWART: ("Charcoal", "PLA Matte", "#000000", "zwart", "black"),
    ORANJE: ("Pumpkin Orange", "PLA Basic", "#FF9016", "oranje", "orange"),
    WIT: ("Ivory White", "PLA Matte", "#FFFFFF", "wit", "white"),
}

# --- lagen in y ---------------------------------------------------------------------------------
DIEPTE = 86.0            # buitenmaat van bak en voet in y (binnen 80 + 2 x 3)
WAND_VOOR = 3.0          # voor- en achterwand van bakken en voet, waar de pennen in zitten
ARM_T = 4.5              # dikte van een arm (y)
ARM_SPLEET = 0.5         # tussen arm en wand
KOP_T = 1.6              # pennenkop

# --- mechanisme (x-z) ---------------------------------------------------------------------------
M = 1.25                 # module tandsector en heugel
Z_WIEL = 16              # tanden van een volle cirkel: steekstraal 10
R_WIEL = M * Z_WIEL / 2
HEUGEL_HALF = 3.6        # steeklijn van de heugel op x = +-3,6 (rug tussen de tandvoeten 4 mm)
P1 = (HEUGEL_HALF + R_WIEL, 18.0)     # draaipunt binnenarm (met tandsector) op de voet
ARM_D = 26.0                           # afstand binnenarm - buitenarm (horizontaal parallellogram)
P2 = (P1[0] + ARM_D, P1[1])
L = 58.0                 # armlengte (hart-op-hart)
THETA_DICHT = 92.0       # armen 2° voorbij verticaal: het gewicht houdt de pod dicht
THETA_OPEN = 28.0        # armen 28° boven horizontaal; 28° van het dode punt (0°)
SPELING_TAND = 0.1       # per deel, op de steekcirkel

# --- pennen (klikpen met kop, apart geprint) ----------------------------------------------------
PEN_D = 5.0
GAT_ARM = 5.25           # draaigat in de arm (0,125 mm speling rondom; pasproef: 5,15 / 5,25 / 5,35)
GAT_WAND = 5.05          # gat in de wand van voet of bak (pasproef: 4,95 / 5,05 / 5,15)
PEN_SPEL = 0.15          # axiale speling bij de kop en achter de wand
PEN_KOP_D = 9.0
OOG_R = 5.2              # straal van het armeinde rond de pen
ARM_B = 9.0              # breedte van een arm

# --- bakjes -------------------------------------------------------------------------------------
BAK_X0 = 6.5             # binnenkant van een bak in de dichte stand (rug tegen de middenrug)
BAK_B = 46.0             # buitenbreedte in x
BAK_H = 26.0             # wandhoogte
BAK_R = 7.0              # ronde bovenrand aan de buitenkant
BAK_PEN_H = 22.0         # draaipunten op de bak, boven de bakbodem
ZIJWAND = 2.4
BODEM = 2.4
VAKKEN = 2               # vakken per bak (scheiding dwars op y)

# --- voet en dak --------------------------------------------------------------------------------
RUG_HALF = 5.5           # middenrug: x = -5,5 .. 5,5
VOET_HALF = 57.0         # voetplaat x = -57 .. 57
VOET_H = 4.0
SPELING_ZWAAI = 1.5      # vrije ruimte onder de zwaaiende bak
DAK_T = 5.0
DAK_OVERSTEEK = 1.5      # dak steekt zoveel buiten de dichte bakken uit
KAM_HALF = 14.0          # bitkam op het dak, x = -14 .. 14
KAM_H = 9.0
BIT_SW = 6.6             # zeskant voor 1/4"-bits (6,35 mm), sleutelwijdte
BIT_DIEP = 13.0
BITS = [(-6.5, y) for y in (16, 30, 44, 58, 72)] + [(6.5, y) for y in (16, 30, 44, 58, 72)]

# --- schuif (heugel + knop) ---------------------------------------------------------------------
ZWALUW = (4.0, 5.6, 2.0)     # zwaluwstaart: breedte aan de voorkant, achteraan, diepte
ZWALUW_SPEL = 0.3            # speling rondom in de groef (loodrecht op elke flank)
KNOP = (24.0, 8.0, 9.0)      # knop boven het dak: x, y (vanaf de voorkant van de schuif), z


def rad(g):
    return math.radians(g)


def arm_punt(p, theta):
    """Eindpunt op de bak van een arm met voetpunt p bij hoek theta (graden)."""
    return (p[0] + L * math.cos(rad(theta)), p[1] + L * math.sin(rad(theta)))


def verschuiving(theta):
    """Hoe ver een bak (rechts) verschoven is t.o.v. de dichte stand."""
    return (L * (math.cos(rad(theta)) - math.cos(rad(THETA_DICHT))),
            L * (math.sin(rad(theta)) - math.sin(rad(THETA_DICHT))))


def heugel_slag(theta):
    """Hoeveel de heugel omhoog is geschoven t.o.v. dicht (mm)."""
    return R_WIEL * rad(THETA_DICHT - theta)


T1_DICHT = arm_punt(P1, THETA_DICHT)
T2_DICHT = arm_punt(P2, THETA_DICHT)
BAK_Z0 = T1_DICHT[1] - BAK_PEN_H          # bakbodem in de dichte stand
BAK_TOP = BAK_Z0 + BAK_H
DAK_Z0 = round(max(BAK_TOP, T1_DICHT[1] + OOG_R) + 0.9, 2)   # onderkant dak
DAK_HALF = BAK_X0 + BAK_B + DAK_OVERSTEEK
SLAG = heugel_slag(THETA_OPEN)
