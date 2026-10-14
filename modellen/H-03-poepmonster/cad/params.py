"""Maten van H-03 Poepmonster (mm, graden).

Een pompoenmonster dat de 'poep' van de A1 opvangt: de stukjes filament die de printer bij elke kleurwissel
uitspuugt. Eén stuk, een bak met ribben. Bovenaan is hij schuin afgesneden: laag achteraan (daar komt de poep
binnen), hoog vooraan (wat te ver vliegt, botst ertegen en valt erin), met tanden rondom. Vooraan twee ogen en
een buik met hechtingen: door de spleten zie je hoeveel hij al gegeten heeft.

Assen: z = hoogte (bodem op z = 0), het gezicht kijkt naar -y (vooraan).
"""
VERSIE = "v0.1"

# --- romp ------------------------------------------------------------------------------------------
R_BODEM = 44.0          # straal onderaan (standvlak)
R_MAX = 56.0            # straal van de buik
Z_BOL = 36.0            # tot hier loopt de buik uit
R_TOP = 51.0            # bovenaan loopt hij weer iets naar binnen (op Z_TOP)
Z_TOP = 120.0
N_RIBBEN = 8
RIB = 0.04              # ribben: straal ± 4 %
WAND = 2.0
VLOER = 2.0

# --- mond: de bovenrand ------------------------------------------------------------------------------
TANDEN = 24             # tanden rondom
TAND_H = 8.0

# variant: (naam NL, naam EN, rand achteraan, rand vooraan, buikvenster van-tot)  -- hoogtes van de tanddalen
VARIANTEN = {
    "laag": ("laag", "low", 40.0, 75.0, (9.0, 35.0)),
    "hoog": ("hoog", "high", 65.0, 100.0, (12.0, 54.0)),
}

# --- gezicht (u = breedte, + = rechts als je ervoor staat; v = hoogte boven het bed) -----------------
OOG_GROOT = (-16.0, 11.0)   # (u, straal), linker oog
OOG_KLEIN = (16.5, 8.0)
OOG_ONDER_RAND = 24.0       # midden van de ogen zoveel onder de rand vooraan
BUIK_B = 66.0               # breedte van de buikspleet
HECHTING_B = 2.4            # breedte van een hechting (een staafje door de spleet)
HECHTING_STAP = 9.0         # afstand tussen twee hechtingen: de spleten ertussen zijn 6,6 mm
RES = 0.1
OVERHANG = 40.0
BRUG = 12.0

# --- filament ----------------------------------------------------------------------------------------
ORANJE = "10301"
FILAMENTEN = {
    ORANJE: ("Pumpkin Orange", "PLA Basic", "#FF9016", "oranje", "orange"),
}
