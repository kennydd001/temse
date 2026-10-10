"""Sonic (fan-art, alleen voor eigen gebruik): overgetrokken van de referentie (klassieke Sonic-kop, driekwart).

Blauw, latte snuit en binnenoor, witte ogen, zwarte lijnen; witte handschoenen over de bordrand.
Sonic is een personage van SEGA. Niet publiceren of verkopen.
"""
from shapely.ops import unary_union

from params import BLAUW, LATTE, WIT, ZWART

from .kit import Dier, Tekening, handschoenen
from .trace import Bron, lagen

BRON = Bron(
    "sonic.png",
    palet=(((3, 4, 8), ZWART), ((254, 254, 254), WIT), ((18, 70, 151), BLAUW), ((243, 215, 141), LATTE)),
    hoogte=103.0, verdik=0.1)

HANDEN = (-44.0, 20.0)


def tekening():
    sil, kop = lagen(BRON, basis=BLAUW, hoogtes={LATTE: 4.6})
    handen, hand_lagen = handschoenen(HANDEN, WIT)
    return Tekening(silhouet=unary_union([sil, handen]), lagen=kop + hand_lagen)


DIER = Dier("sonic", 9, {"nl": "sonic", "en": "sonic"}, bord=BLAUW, kop=ZWART, decor=(WIT, "ring"),
            tekening=tekening, fanart=True)
