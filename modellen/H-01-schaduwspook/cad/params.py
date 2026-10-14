"""Maten en varianten van H-01 Schaduwspook (mm, graden).

Een variant is een rij silhouetten met elk een hoek. Het beeldje is de doorsnede van al die silhouetten,
elk uitgetrokken in zijn eigen kijkrichting. Schijn je in de richting van een hoek, dan zie je dat
silhouet; draai je het beeldje verder, dan verandert de schaduw. Alle silhouetten staan op een voetbalk en
worden even hoog geschaald.

Twee silhouetten onder 90° passen altijd precies. Bij vier onder 45° moet elk silhouet passen binnen wat de
schuine richtingen toelaten: de breedste vorm iets smaller maken ('pompoen*0.85') houdt alles heel.
"""
VERSIE = "v0.1"

VOET_H = 2.4          # voetbalk onder elk silhouet: houdt alle losse letters en stukken bij elkaar
VOET_STAP = 0.2       # elk volgend silhouet een laag hogere balk: nooit randen op exact dezelfde hoogte
VOET_MARGE = 2.5      # balk steekt zover uit naast het silhouet
OVERLAP = 0.6         # silhouet zakt zover in de balk: één stuk
RES = 0.1             # rijhoogte voor het zelfdragend maken
OVERHANG = 40.0       # graden vanaf verticaal; 45° mag, 40° houdt marge
MIN_DETAIL = 1.2      # dunste stijl in de doorsnede (3 lijnen van 0,4 mm)

# Bambu-filament (één kleur; de schaduw is altijd zwart, de kleur is voor overdag)
ORANJE = "10301"
ZWART = "11101"
FILAMENTEN = {
    ORANJE: ("Pumpkin Orange", "PLA Basic", "#FF9016", "oranje", "orange"),
    ZWART: ("Charcoal", "PLA Matte", "#000000", "zwart", "black"),
}

# sleutel: (naam NL, naam EN, silhouetten, hoeken, hoogte, filament)
VARIANTEN = {
    "vier-schaduwen": ("vier schaduwen", "four shadows", ("pompoen*0.85", "spook", "schedel", "grafsteen"),
                       (0.0, 45.0, 90.0, 135.0), 60.0, ORANJE),
    "boo-schedel": ("BOO en schedel", "BOO and skull", ("smal:BOO", "schedel"), (0.0, 90.0), 55.0, ZWART),
    "trick-treat": ("TRICK en TREAT", "TRICK and TREAT", ("smal:TRICK", "smal:TREAT"), (0.0, 90.0), 36.0, ORANJE),
}
