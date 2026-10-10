"""Alle maten van B-34A Taakvreter (mm, graden). Eén plek om aan te passen na de pasproef.

Assen: x = breedte (0 = midden, rechts = de kant van de kruk), y = diepte (voorkant van het gezicht op
y = 0), z = hoogte (tafel op z = 0). De romp is gedeeld in een voor- en een achterhelft op y = Y_DEEL, het
vlak waarin beide rolassen liggen: alle assen vallen bij de montage in halve gaten.
"""
import math

VERSIE = "v0.1"

ORANJE = "10301"    # romp, kruk, paspennen
WIT = "11100"       # lade, ogen, tanden, knop, hoorntjes
ZWART = "11101"     # rollen, nokas, oogmechaniek, mondstuk, pupillen
FILAMENTEN = {
    ZWART: ("Charcoal", "PLA Matte", "#000000", "zwart", "black"),
    ORANJE: ("Pumpkin Orange", "PLA Basic", "#FF9016", "oranje", "orange"),
    WIT: ("Ivory White", "PLA Matte", "#FFFFFF", "wit", "white"),
}

# --- romp ---------------------------------------------------------------------------------------
B2 = 62.0                 # halve buitenbreedte
H = 136.0                 # hoogte
D = 96.0                  # diepte
WAND = 3.0
Y_DEEL = 34.0             # deelvlak voor/achter = vlak van de rolassen
R_TOP = 34.0              # afronding bovenhoeken (vooraanzicht)
R_VOET = 8.0              # afronding onderhoeken
SCHUIN = 2.0              # 45°-afschuining van de voor- en achterrand (printbaar op het bed)
X_BINNEN = B2 - WAND      # binnenkant zijwand
E = 0.01

# --- papier, mond -------------------------------------------------------------------------------
PAPIER = 76.0             # memoblaadjes 76 x 76 mm
SLEUF_B = 82.0            # breedte invoer: 76 mm scheef tot ~4,4° past nog
SLEUF_H = 4.0             # hoogte invoer: daar past geen vinger in
TRECHTER = 2.0            # invoer loopt vooraan 45° open, 2 mm rondom
MOND = (50.0, 19.0)       # halve breedte, diepte van de grijns (superellips onder z = MOND_TOP)
MOND_EXP = 3.0            # 2 = halve ellips; 3 = vlakkere bodem (de sleuf past er onderin breed genoeg in)
MOND_TOP = 67.0
LIP = 2.5                 # lip van het mondstuk rond de mond, vóór het gezicht
LIP_T = 1.2
MOND_SPEL = 0.1           # het mondstuk klemt in de mond
TANDEN_T = 1.2            # dikte van het bovengebit, vóór op het mondstuk
TUNNEL_Y = 23.5           # achterkant van de invoertunnel
TUNNEL_HALF = 44.0        # tunnel achter de gezichtswand: smaller dan de mond (lagerbeugel links)

# --- rollen -------------------------------------------------------------------------------------
M_ROL = 1.0
Z_TAND = 17               # tandwiel aan de rechterkant van beide rollen, 1:1
AS = M_ROL * Z_TAND       # hart-op-hart 17 mm
ROL_R = 8.3               # rolstraal (Ø 16,6)
ROL_X = 42.0              # rollen van x = -42 tot 42 (sleuf -41 .. 41)
TAND_X = (42.3, 46.3)     # tandwielen, aan de rollen vast
TAND_KOP = M_ROL * Z_TAND / 2 + M_ROL          # 9,5
TAND_VOET = M_ROL * Z_TAND / 2 - 1.25 * M_ROL  # 7,25
Z_ONDER = 47.0            # as onderste rol (aangedreven, met O-ringen)
Z_BOVEN = Z_ONDER + AS    # as bovenste rol
GROEF = (-28.0, 0.0, 28.0)    # O-ringgroeven in de onderste rol
GROEF_DIEP = 0.9          # O-ring 14 x 2 mm (NBR): op de groefbodem (Ø 14,8) 6 % gerekt; of een gewikkeld elastiekje
GROEF_B = 2.4
ORING_SNOER = 1.95        # snoerdikte van de licht gerekte O-ring
ORING_R = ROL_R - GROEF_DIEP + ORING_SNOER   # 9,35: drukt 0,65 mm in tegen de bovenste rol (nominaal)
LAGERSPELING = 0.40       # wat de O-ringen de rollen uit elkaar kunnen duwen: per kant twee lagers en een zeskant
NIP_Z = (Z_ONDER + ORING_R + Z_BOVEN - ROL_R) / 2   # papiervlak
STOMP_R = 3.0             # geprinte asstompen aan de rollen
SCHOUDER_R = 4.5          # schouder op de stomp: houdt de rol op zijn plek tussen de wanden
STOMP_EIND = 60.7         # |x| van het eind van een stomp (blind gat tot 61, buitenwand op 62)
GAT_EIND = 61.0
GAT_DRAAI = 6.3           # draaigat (halve gaten in beide rompdelen)
RIB = (53.0, 59.0)        # |x| van de lagerribben in de zijwanden (rechts beide assen, links de onderste)
RIB_LINKS = (-59.0, -55.5)
ZESKANT = 5.2             # zeskant van kruk-as en nokas (omgeschreven Ø 6,0: past in het ronde deel)
ZESKANT_GAT = 5.4         # zeskantkoker in de rollen
KOKER_DIEP = 10.0
BOVEN_X0 = -44.7          # linkereind van de bovenste rol (naaf tegen de lagerbeugel)

# --- kruk ---------------------------------------------------------------------------------------
KRUK_R = 32.0
KRUK_X = (64.5, 68.5)     # krukarm, buiten de rechterwand (x = 62)
KRUK_B = 6.0              # halve breedte van de arm
KRAAG = (46.6, 51.2)      # kraag op de kruk-as binnen de wand, met 45° naar de as tot 52,7
KNOP_R = 6.5
KNOP_L = 16.0
KNOPPEN_R = 2.25          # klikpen van de knop
KNOP_GAT = 2.45           # gat in de krukarm (straal)

# --- ogen (vierstangenmechanisme links) -----------------------------------------------------------
OOG_X = 24.0
OOG_Z = 95.0
OOG_Y = 18.5              # oogas
TROMMEL_R = 13.0
TROMMEL_HALF = 11.0       # trommels x = +-OOG_X +- 11
VENSTER_R = 12.0
PUPIL_R = 4.5
PUPIL_T = 1.0             # dikte van de pupil vóór de trommel (midden)
OOGAS_R = 3.0
U_OPENING = 5.6           # klikopening van de oogaslagers (as Ø 6)
NOK_E = 5.0               # kruk op de bovenste rol
STANG = 34.0              # drijfstang
TUIMEL = 11.0             # tuimelaar op de oogas
TUIMEL_X = (-57.9, -55.2)    # dun genoeg om vrij te blijven van de afgeronde bovenhoek van de zijwand
STANG_X = (-54.9, -52.1)
NOKSCHIJF_X = (-51.5, -48.7)
NOKSCHIJF_R = 8.5
BEUGEL_X = (-48.0, -45.0)     # lagerbeugel: bovenste rol (nokas) en oogas, van voor tot achter
OOGKLEM_X = (3.0, 6.0)        # middelste oogaslager, tussen de trommels
OOGKLEM_LINKS = (-44.0, -41.0)  # linker oogaslager, rechts naast de beugel
PEN_R = 2.0                   # pennen van tuimelaar en drijfstang
PEN_GAT = 2.2

# --- lade (buikbak) -----------------------------------------------------------------------------
LADE_HALF = 55.6
LADE_Z = (3.4, 34.1)
LADE_Y = (-0.6, D - WAND - 0.6)
LADE_WAND = 2.0
LADE_FAS = 3.0            # 45° onderaan de lange kanten (past in de afgeronde onderhoeken van de romp)
BAAI_SPEL = 0.4
SCHAP = (34.8, 36.8, 44.0)    # schap onder de rollen: z-onder, z-boven, tot y (stuk voorbij Y_DEEL aan de voorhelft)
SCHAP_HALF_ACHTER = 52.7      # breedte van het stuk voorbij het deelvlak (vrij van de ribben in de achterhelft)

# --- glijgoot achter de rollen ------------------------------------------------------------------
GOOT_MIDDEN = (44.5, 36.8)    # (y, z) middelpunt van de kwart-ellips
GOOT_R = (46.0, 20.0)         # stralen in y en z
GOOT_TOP = 72.0

# --- paspennen, hoorntjes -----------------------------------------------------------------------
PASPENNEN = [(-40.0, 122.0), (40.0, 122.0), (52.0, 85.0), (-50.0, 118.0), (-57.5, 38.6), (57.5, 38.6)]
KOLOM_R = 4.0
PASPEN_D = 5.0
PASPEN_L = 16.0
GAT_PASPEN_VAST = 4.9     # voorhelft: klemt
GAT_PASPEN_LOS = 5.05     # achterhelft: schuift er met wat wrijving op (open te maken bij een papierprop)
HOORN_X = 20.0
HOORN_Y = 17.0
HOORN_GAT = 5.05          # in de romp (druppelvorm); in het hoorntje zelf 4,9 (klemt)


def rad(g):
    return math.radians(g)
