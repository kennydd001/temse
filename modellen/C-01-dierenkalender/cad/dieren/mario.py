"""Mario (fan-art, alleen voor eigen gebruik): overgetrokken van de referentie (het Mario-kussen).

Vier kleuren: rode pet en mond, latte gezicht, wit embleem/ogen/tanden, zwarte lijnen, snor, haar en
irissen (de lichtblauwe iris kan niet als vijfde kleur). De neus staat hoger dan de rest.
Mario is een personage van Nintendo. Niet publiceren of verkopen.
"""
from shapely.ops import unary_union

from params import LATTE, ROOD, WIT, ZWART

from .kit import Dier, Tekening, handschoenen
from .trace import ACHTER, Bron, lagen

BRON = Bron(
    "mario.webp",
    palet=(((28, 26, 29), ZWART), ((107, 39, 26), ZWART), ((80, 158, 215), ZWART, False),
           ((254, 254, 254), WIT), ((218, 212, 209), WIT), ((211, 180, 151), LATTE),
           ((239, 74, 49), ROOD), ((218, 56, 36), ROOD),
           ((35, 108, 179), ACHTER), ((25, 70, 130), ACHTER)),     # blauwe pluche rand van het kussen
    hoogte=103.0, ruis=0.35, glad=1.2)

HAND_X = 50.0
HAND_Y = 100.5


def tekening():
    sil, kop = lagen(BRON, basis=LATTE, verhoog=((LATTE, (0.0, 124.0), 5.4),))
    handen, hand_lagen = handschoenen((-HAND_X, HAND_X), WIT)
    return Tekening(silhouet=unary_union([sil, handen]), lagen=kop + hand_lagen)


DIER = Dier("mario", 8, {"nl": "mario", "en": "mario"}, bord=ROOD, kop=ZWART, decor=(WIT, "ster"),
            tekening=tekening, fanart=True)
