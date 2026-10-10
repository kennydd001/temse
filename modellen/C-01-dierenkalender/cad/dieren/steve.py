"""Steve (fan-art, alleen voor eigen gebruik): blokkige pixelkop en vuisten, blauw bord met witte labels.

Steve is een personage uit Minecraft (Mojang/Microsoft). Niet publiceren of verkopen.
"""
from shapely.ops import unary_union

import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import rounded_rect
from params import BLAUW, LATTE, TERRA, WIT

from .kit import Dier, Tekening, pixels, poot, relief, vlak

MAAT = 11.5   # mm per pixel: 8 pixels = 92 mm kop

# eigen 8x8-weergave: H haar, S huid, W oogwit, B iris, N neus, M mond/baard
GEZICHT = [
    "HHHHHHHH",
    "HHHHHHHH",
    "HSSSSSSH",
    "SSSSSSSS",
    "SWBSSBWS",
    "SSSNNSSS",
    "SSMMMMSS",
    "SSSSSSSS",
]


def tekening():
    x0, top = -4 * MAAT, 100.0 + 8 * MAAT
    p = pixels(GEZICHT, x0, top, MAAT)
    kop = rounded_rect(x0, 100.0, -x0, top, 0.6)
    vuist = lambda cx: rounded_rect(cx - 9, 91.5, cx + 9, 109.5, 1.0)  # noqa: E731
    vuisten = unary_union([vuist(-38), vuist(38)])
    return Tekening(
        silhouet=unary_union([kop, vuisten]),
        lagen=[
            relief(LATTE, kop),
            vlak(TERRA, unary_union([p["H"], p["N"], p["M"]])),
            vlak(WIT, p["W"]),
            vlak(BLAUW, p["B"]),
            poot(LATTE, vuisten),
        ])


DIER = Dier("steve", 11, {"nl": "steve", "en": "steve"}, bord=BLAUW, kop=LATTE, decor=(WIT, None),
            tekening=tekening, label=WIT, fanart=True)
