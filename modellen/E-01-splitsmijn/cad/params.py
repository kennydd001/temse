"""Alle maten van E-01 Splitsmijn, het valbord (mm, graden). Eén plek om aan te passen na de pasproef.

Assen van het valbord: x = breedte (0 = midden), z = hoogte (onderrand van het bord op z = 0), y = diepte:
achterplaat y = 0 .. ACHTER_T, de spleet waarin de muntjes rechtop vallen y = ACHTER_T .. Y_VOOR, de
voorplaat y = Y_VOOR .. Y_VOOR + VOOR_T. De kijkkant is +y.

Waarom een slinger (zie simulatie.py): een muntje dat groot is naast de pinnen rolt er in een vaste baan
overheen, en dan beslist het mikpunt bij het inwerpen links of rechts. Een kantelplank onder de inworp,
vast aan een zware slinger die het kind een duw geeft, beslist in plaats daarvan: welke kant ligt er op
dat moment lager. Het muntje is te licht om de slinger te storen.
"""
import math

VERSIE = "v0.1"

# --- filamenten (uit het tabblad Filamenten: wat er besteld is) ------------------------------------
ZWART = "11101"     # achterplaat en voorplaat (donker: de muntjes vallen op)
BRUIN = "11800"     # voet met de bakjes, beugel
BLAUW = "11600"     # diamantmuntjes
ROOD = "10205"      # slinger, mesblok en kantelplank: alles wat beweegt
FILAMENTEN = {
    ZWART: ("Charcoal", "PLA Matte", "#000000", "zwart", "black"),
    BRUIN: ("Latte Brown", "PLA Matte", "#D3B7A7", "bruin", "brown"),
    BLAUW: ("Marine Blue", "PLA Matte", "#0078BF", "blauw", "blue"),
    ROOD: ("Maroon Red", "PLA Basic", "#9D2235", "rood", "red"),
}

# --- muntjes ------------------------------------------------------------------------------------
MUNT_D = 25.0             # als een munt van 2 euro: goed te pakken, en klein genoeg naast de pinnen
MUNT_T = 4.5
MUNT_FAS = 0.5
AANTAL_MUNTEN = 10

# --- valbord ------------------------------------------------------------------------------------
ACHTER_T = 2.4
SPLEET = MUNT_T + 1.5     # het muntje kan hooguit ~3° scheef hangen: klemt nergens
Y_VOOR = ACHTER_T + SPLEET
VOOR_T = 2.5
BORD_T = Y_VOOR + VOOR_T

PIN_D = 8.0
STEEK = 44.0              # pinnen in een rij: tussenruimte 36 voor een muntje van 25
RIJ = STEEK * math.sqrt(3) / 2   # driehoeksraster
RIJEN = 4                 # rij 0 (boven) .. 3; de onderste rij heeft een pin in het midden
Z_ONDERSTE_RIJ = 30.0
NOK_D = 3.0               # nokje op elke pin, klemt in de voorplaat
NOK_H = 1.6
NOK_GAT = 3.1

BINNEN_HALF = 1.5 * STEEK + PIN_D / 2 + MUNT_D + 4.0   # 99: de buitenste pinnen houden een muntje + 4 vrij
RAND = 10.0               # zijranden x = +-99 .. +-109
BULT_R = 8.0              # halve pinnen in de wand naast de rijen met een middenpin
INWORP_HALF = MUNT_D / 2 + 1.0    # 1 mm speling per kant: hoe smaller, hoe minder het mikpunt meetelt
Z_PLANK = 184.0           # draaipunt van de kantelplank
PLAFOND_Z = Z_PLANK + 5.0 + MUNT_D + 4.0    # boven de vlakke plank (PLANK_H 10) nog 4 mm vrij voor een muntje
BORD_H = PLAFOND_Z + 12.0
TRECHTER_HALF = 36.0      # V-uitsparing bovenin de voorplaat: hier leg je het muntje tegen de achterplaat

TRALIE_B = 3.5            # kijkspleten in de voorplaat (de rand van een muntje, 4,5 mm, past er niet in)


def rij_z(k):
    return Z_ONDERSTE_RIJ + (RIJEN - 1 - k) * RIJ


def middenrij(k):
    """Rijen met een pin in het midden (0, +-STEEK); de andere hebben +-STEEK/2 en +-1,5 STEEK."""
    return (RIJEN - 1 - k) % 2 == 0


def pinnen():
    """[(x, z)] van alle pinnen."""
    out = []
    for k in range(RIJEN):
        xs = (0.0, -STEEK, STEEK) if middenrij(k) else (-STEEK / 2, STEEK / 2, -1.5 * STEEK, 1.5 * STEEK)
        out += [(x, rij_z(k)) for x in xs]
    return out


def bulten():
    """[(x, z)] van de halve pinnen in de zijwanden."""
    return [(s * BINNEN_HALF, rij_z(k)) for k in range(RIJEN) if middenrij(k) for s in (-1, 1)]


# --- kantelplank en slinger ---------------------------------------------------------------------
# De slinger hangt met een mes (een prisma met de punt omlaag) in een ondiepe V-groef op een beugeltje
# vóór de slingerarm, zoals een ouderwetse weegschaal: bijna geen wrijving, dus hij slingert lang door.
# De naaf met de kantelplank gaat door een gat in de voorplaat zonder het te raken.
PLANK_HALF = 18.0         # plank x = +-18, in de spleet 5 breed (y)
PLANK_H = 10.0            # zo hoog als de naaf: de bovenkant is vlak, er zit geen bult in het midden
PLANK_Y = (ACHTER_T + 0.5, Y_VOOR - 0.5)
NAAF_R = 5.0              # naaf van de plank door de voorplaat (gat NAAF_R + 1: raakt niets)
ZESKANT = 8.0             # zeskant vooraan op de naaf: de slingerarm schuift erop
ZESKANT_SPEL = 0.2
ARM_Y = (BORD_T + 0.5, BORD_T + 5.5)
ARM_B = 8.0
SLINGER_L = 150.0         # van het mes tot het midden van het gewicht: slingertijd ~0,8 s
GEWICHT = (44.0, 36.0, 17.0)   # b x h x d (y): helemaal massief ~27 g, zwaartepunt boven het mes
MES = (ARM_Y[1], ARM_Y[1] + 6.5)   # y van het mesblok vóór de arm (los onderdeel, klikt op twee pennetjes)
MES_HALF = 6.0            # halve breedte van het mes (de V)
MES_BLOK = (10.0, 16.0)   # halve breedte en hoogte (boven de as) van het mesblok
MES_PEN = (7.5, 13.0, 3.0)   # x, z boven de as en Ø van de pennetjes op de arm
MES_HOEK = 60.0           # tophoek van het mes
GROEF_HOEK = 150.0        # V-groef in de beugel (flanken 15° boven horizontaal): 45° draaien voor ze raken
GROEF_DIEP = 1.2
BEUGEL_HALF = 22.0
BEUGEL_Y = MES[0] + 0.5   # achterkant van de voorwand van de beugel (de arm zwaait erachter)
POOT = (14.5, 12.0)       # poten van de beugel achter de voorwand: vanaf |x| = 14,5 (de arm zwaait ertussen), tot 12 mm onder de as
BEUGEL_PEN = (17.5, 6.0, 4.0)   # x, z onder de as en Ø van de pennen op de voorplaat waar de beugel op klikt
MAX_HOEK = 35.0           # zo ver mag de slinger uitslaan zonder iets te raken

# --- verdeler (midden in de voet, steekt met een vin omhoog in de spleet) -----------------------
VERDELER_T = 3.0          # dikte in x
VERDELER_TOP = 20.0       # nok van de verdeler, 6 mm onder de onderste middenpin

# --- voet met bakjes (pasproef) -----------------------------------------------------------------
VOET_HALF = BINNEN_HALF + RAND + 4.0
VOET_Y = (-18.0, 30.0)    # het bord staat op y = 0 .. BORD_T; de kinderen pakken de muntjes aan de voorkant (+y)
VOET_Z = (-32.0, 15.0)    # bakjes onder het bord, zakken voor de zijranden tot z = 15
VOET_WAND = 2.4
VOORRAND_H = 8.0          # lage voorrand: muntjes rollen er niet uit en zijn makkelijk te pakken
SPEL = 0.3


def rad(g):
    return math.radians(g)
