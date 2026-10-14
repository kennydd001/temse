"""Maten van H-02 Moodkin (mm, graden).

Assen: z = hoogte (onderkant schil op z = 0), het gezicht kijkt naar -y (vooraan). De trommel en het deksel
draaien samen rond de z-as; om de 90° klikt het volgende gezicht voor de gaten van de schil.

Vier onderdelen: de schil (pompoen met vooraan een vast gesneden gezicht: oogkassen, neuskas, tandkolommen),
de trommel (een buis met de vier gezichten, om de 90°, die door die kassen te zien zijn), het deksel en het
steeltje. De trommel klemt met vier pennetjes onder het deksel; het deksel ligt op de rand van de schil en klikt
met vier gleufjes over vier nokjes. Het steeltje steekt met een vierkant pennetje in het deksel.
"""
VERSIE = "v0.1"

# --- schil -----------------------------------------------------------------------------------------
H_SCHIL = 72.0          # bovenrand van de schil
R_BODEM = 36.0          # straal onderaan (standvlak)
R_MAX = 56.0            # straal van de buik (midden van de ribben)
Z_BOL = 34.0            # tot hier loopt de buik uit; daarboven recht omhoog
N_RIBBEN = 8
RIB = 0.04              # ribben: straal ± 4 %
WAND = 2.0              # dunste wand van de schil (in een ribdal)
VLOER = 2.0

# --- trommel ---------------------------------------------------------------------------------------
R_TROMMEL = 50.0        # buitenstraal
TROMMEL_WAND = 1.6
SPELING = 1.2           # tussen trommel en schil
R_BINNEN_SCHIL = R_TROMMEL + SPELING
Z_TROMMEL = 30.0        # onderkant trommel: boven de buik van de schil
Z_GEZICHT = (32.0, 64.0)  # band met de gezichten
GEZICHT_B = 62.0        # breedte van een gezicht: ±38° op de trommel, de buren beginnen pas op 52°
RICHEL_B = 6.0          # richel binnen bovenaan de trommel, voor de pennetjes (45° afgeschuind)
PEN_D = 4.0
PEN_H = 4.0
PEN_GAT = 4.2           # klemt met duimdruk (zoals bij E-01: +0,2)
PEN_R = R_TROMMEL - TROMMEL_WAND - RICHEL_B / 2
RICHEL_DIK = 3.0
PEN_HOEKEN = (45.0, 135.0, 225.0, 315.0)   # tussen de gezichten

# --- deksel ----------------------------------------------------------------------------------------
R_DEKSEL = 58.0
DEKSEL_RAND = 3.0       # rechte rand onderaan
DEKSEL_H = 15.0         # koepel boven de rand
STEEL_R = (6.5, 4.8)    # onder, boven
STEEL_H = 26.0
STEEL_BOCHT = 6.0       # het steeltje buigt opzij
STEEL_GROEVEN = 6
STEEL_DRAAI = 25.0      # graden, over de hele hoogte
STEEL_PEN = 6.0         # vierkant: het steeltje draait het deksel mee
STEEL_PEN_H = 5.0
STEEL_GAT = 6.3

# --- klik om de 90° ----------------------------------------------------------------------------------
NOK_R = 52.6            # op de rand van de schil (tussen trommel en ribdal)
NOK_H = 0.8
NOK_B = 1.6
GLEUF_DIEP = 1.2
GLEUF_B = 3.0
NOK_HOEKEN = (45.0, 135.0, 225.0, 315.0)   # niet in het gezicht vooraan (-90°)

# --- LED-theelichtje -------------------------------------------------------------------------------
LED_D = 38.0            # gewoon LED-theelichtje (Ø 38 mm); NOOIT een echte kaars
LED_RING_D = 39.6
LED_RING_H = 3.0
LED_RING_W = 1.6

# --- gezichten -------------------------------------------------------------------------------------
GEZICHTEN = ("blij", "boos", "bang", "knipoog")   # op de trommel om de 90°, in deze volgorde
RES = 0.1
OVERHANG = 40.0         # graden vanaf verticaal (marge t.o.v. 45°)
BRUG = 12.0             # een gat mag een plat plafond hebben tot zo breed: de printer overbrugt het

# --- filament ----------------------------------------------------------------------------------------
ORANJE = "10301"
GEEL = "10402"          # binnenkant: oplichtend vruchtvlees
GROEN = "10502"
FILAMENTEN = {
    ORANJE: ("Pumpkin Orange", "PLA Basic", "#FF9016", "oranje", "orange"),
    GEEL: ("Sunflower Yellow", "PLA Basic", "#FEC600", "geel", "yellow"),
    GROEN: ("Mistletoe Green", "PLA Basic", "#3F8E43", "groen", "green"),
}
