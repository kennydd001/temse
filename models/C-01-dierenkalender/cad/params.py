"""Maten voor C-01 Eeuwigdurende dierenkalender (v0.1).

Alle maten in mm. Startwaarden komen uit het tabblad "Dierenkalender" van de
MakerWorld-productielijst; afwijkingen staan erbij met de reden.
"""

# --- Kaartjes -------------------------------------------------------------
CARD_H = 28.0            # hoogte van elk kaartje
CARD_BASE = 1.2          # Ivory White basis
CARD_TEXT = 0.4          # Charcoal tekst, verzonken in de bovenste 0,4 mm
CARD_T = CARD_BASE + CARD_TEXT   # 1,6 mm totaal, overal vlak
CARD_R = 1.5             # hoekradius
CARD_W = {"dag": 56.0, "cijfer": 22.0, "maand": 76.0}

DAGEN = ["MAANDAG", "DINSDAG", "WOENSDAG", "DONDERDAG", "VRIJDAG", "ZATERDAG", "ZONDAG"]
MAANDEN = ["JANUARI", "FEBRUARI", "MAART", "APRIL", "MEI", "JUNI", "JULI",
           "AUGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DECEMBER"]
TIENTALLEN = ["0", "1", "2", "3"]
EENHEDEN = [str(i) for i in range(10)]

# --- Dierenfront ----------------------------------------------------------
PLATE_T = 3.0            # vlakke achterplaat
BOARD_W = 200.0          # breedte bord
BOARD_TOP = 100.0        # bovenrand bord (dier steekt erboven uit)
BOARD_R = 8.0
TOTAL_H = 210.0          # bord + dier + tong
TONGUE_W = 188.0
TONGUE_L = 16.0          # steekt onder het bord uit
HEAD_RELIEF = 1.0        # kop ligt 1 mm hoger dan het bord
PAW_TOP = 5.0            # pootjes: hoogste oranje reliëf (z)
COLOR_LAYER = 0.6        # wit reliëf op de kop
DETAIL_TOP = 5.0         # bovenkant ogen/neus/mond (z)

# Kaartgeleiders: open bovenaan, draagrand onderaan, zijlippen met 45° onderkant
CHANNEL = 2.2            # vrije ruimte tussen plaat en lip (proef: 2,0 / 2,2 / 2,4)
SIDE_CLEAR = 0.4         # speling per zijde naast het kaartje
RAIL_W = 3.0             # breedte van een geleiderrib
LIP = 2.0                # lip steekt 2 mm over de kaartrand
LIP_TIP = 0.8            # minimale dikte van de lippunt
RIM_H = 3.0              # hoogte draagrand onder het kaartje
RIM_GAP = 8.0            # opening midden in de draagrand om het kaartje omhoog te duwen
CARD_EXPOSED = 9.0       # zoveel steekt het kaartje boven de geleiders uit (vastpakken)
WINDOW_Y0 = 40.0         # onderkant draagrand op het bord


def guide_top_z(channel=CHANNEL):
    return PLATE_T + channel + LIP + LIP_TIP


# --- Voet met opbergbakken -------------------------------------------------
FOOT_W = 206.0
FOOT_H = 36.0
FOOT_D = 72.0
FOOT_R = 6.0             # afgeronde staande hoeken
WALL = 2.4
FLOOR = 2.0
BIN_INNER_W = [58.0, 24.0, 24.0, 78.0]   # dag | tiental | eenheid | maand
BIN_INNER_D = 26.0       # 12 maandkaartjes = 19,2 mm + 6,8 mm marge
BIN_FRONT_H = 10.0       # lage voorkant: vingertoegang en tekst van het voorste kaartje zichtbaar
TILT_DEG = 8.0           # front leunt achterover
SLOT_W = 3.8             # tong 3,0 mm (proef: 3,6 / 3,8 / 4,0)
SLOT_L = 188.8
SLOT_DEPTH = 16.5        # 16 mm tong + 0,5 mm, zodat de schouders op de voet rusten
SLOT_FRONT_Y = 36.0      # voorkant sleuf aan de bovenzijde, gemeten vanaf de voorkant voet
SLOT_BLOCK_BACK_Y = 46.0 # achterkant van het massieve sleufblok

# --- Kleuren (Bambu Lab, uit tabblad Filamenten) ---------------------------
FILAMENT = {
    "Pumpkin Orange 10301": "#FF9016",
    "Ivory White 11100": "#FFFFFF",
    "Charcoal 11101": "#000000",
    "Mistletoe Green 10502": "#3F8E43",
    "Latte Brown 11800": "#D3B7A7",
    "Terracotta 11203": "#B15533",
    "Hot Pink 10204": "#F5547C",
    "Sunflower Yellow 10402": "#FEC600",
}
