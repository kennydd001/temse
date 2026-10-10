"""Creeper (fan-art, alleen voor eigen gebruik): groene pixelkop op een Charcoal bord, TNT-blokjes als decor.

De Creeper komt uit Minecraft (Mojang/Microsoft). Niet publiceren of verkopen.
"""
from shapely.ops import unary_union

import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import rounded_rect
from params import GROEN, WIT, ZWART

from .kit import Z_KLEUR, Z_KOP, Dier, Laag, Tekening, detail, pixels, poot, relief

MAAT = 11.5

# eigen 8x8-weergave: F gezicht; L is een iets hoger groen blokje voor de gevlekte textuur
GEZICHT = [
    "L..L...L",
    ".L...L..",
    ".FF..FF.",
    ".FF.LFF.",
    "...FF..L",
    "L.FFFF..",
    "..FFFF.L",
    ".LF..F..",
]


def tekening():
    x0, top = -4 * MAAT, 100.0 + 8 * MAAT
    p = pixels(GEZICHT, x0, top, MAAT)
    kop = rounded_rect(x0, 100.0, -x0, top, 0.6)
    voet = lambda cx: rounded_rect(cx - 10, 91.5, cx + 10, 108, 1.0)  # noqa: E731
    voeten = unary_union([voet(-34), voet(34)])
    tenen = unary_union([rounded_rect(cx + dx - 2.0, 91.5, cx + dx + 2.0, 96.5, 0.3)
                         for cx in (-34, 34) for dx in (-5, 0, 5)])
    return Tekening(
        silhouet=unary_union([kop, voeten]),
        lagen=[
            relief(GROEN, kop),
            Laag(GROEN, p["L"].buffer(-0.6), Z_KOP - 0.01, Z_KLEUR),   # textuur: losse blokjes 0,6 mm hoger
            detail(ZWART, p["F"]),
            poot(GROEN, voeten),
            detail(ZWART, tenen.intersection(voeten.buffer(-0.5))),
        ])


DIER = Dier("creeper", 12, {"nl": "creeper", "en": "creeper"}, bord=ZWART, kop=GROEN, decor=(None, "tnt"),
            tekening=tekening, label=WIT, fanart=True)
