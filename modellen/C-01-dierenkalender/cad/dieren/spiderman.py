"""Spider-Man (fan-art, alleen voor eigen gebruik): overgetrokken van de referentie (het masker).

Rood masker, zwart web (verdikt tot het printbaar is) en zwarte oogranden, verzonken witte lenzen;
rode handschoenen over de bordrand, blauw bord.
Spider-Man is een personage van Marvel. Niet publiceren of verkopen.
"""
from shapely.ops import unary_union

from params import BLAUW, ROOD, WIT, ZWART

from .kit import Dier, Tekening, handschoenen
from .trace import Bron, lagen

BRON = Bron(
    "spiderman.png",
    palet=(((12, 3, 3), ZWART), ((254, 254, 254), WIT), ((224, 43, 37), ROOD)),
    hoogte=103.0, verdik=0.22)

HAND_X = 44.0


def tekening():
    sil, kop = lagen(BRON, basis=ROOD, hoogtes={WIT: 4.2})
    handen, hand_lagen = handschoenen((-HAND_X, HAND_X), ROOD)
    return Tekening(silhouet=unary_union([sil, handen]), lagen=kop + hand_lagen)


DIER = Dier("spiderman", 10, {"nl": "spiderman", "en": "spiderman"}, bord=BLAUW, kop=ZWART, decor=(ZWART, "web"),
            tekening=tekening, fanart=True)
