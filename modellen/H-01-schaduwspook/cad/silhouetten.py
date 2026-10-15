"""De silhouetten van H-01, getekend in code. Elk silhouet staat in een eigen vlak: x = breedte, y = hoogte,
met de onderkant op y = 0 en de breedte rond x = 0. De schaal maakt niet uit: variant() schaalt ze op hoogte.

Gaten (ogen, neus, mond, tanden) zijn de lichtvlekken in de schaduw. Ze mogen een vlakke bovenkant hebben:
zelfdragend() (gereedschap/zelfdragend.py) maakt er een spitse boog van, zodat ze zonder support printen.
"""
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import _pad  # noqa: F401
from gereedschap.geom import FONT_SMAL, _FP, ellipse, rounded_rect, smooth, stroke, sym, text


def schedel():
    """Schedel van voren: ronde kop, jukbeenderen, smallere kaak met tanden."""
    kop = unary_union([ellipse(0, 62, 40, 38), rounded_rect(-34, 26, 34, 58, 12), rounded_rect(-25, 0, 25, 34, 9)])
    kop = smooth(kop, close=7)
    oog = ellipse(-16, 50, 11.5, 12.5, rot=-18)
    neus = Polygon([(-5.5, 25), (5.5, 25), (0, 37)])
    tanden = [rounded_rect(x - 1.5, 6, x + 1.5, 21, 1.4) for x in (-15, -7.5, 0, 7.5, 15)]
    return kop.difference(unary_union([sym(oog), neus] + tanden))


def spook():
    """Spookje met de armen schuin omhoog (BOE!), golvende onderrand op de grond, ogen en een O-mond."""
    lijf = unary_union([Point(0, 64).buffer(31, quad_segs=32), box(-31, 0, 31, 64)])
    zoom = unary_union([Polygon([(x - 7, -1), (x + 7, -1), (x, 9)]) for x in (-15.5, 15.5)])   # spitse inkepingen
    arm = stroke([(-25, 44), (-40, 63)], 10)          # schuin omhoog: onderkant ~52°, print vanzelf
    lijf = smooth(unary_union([lijf, sym(arm)]), close=5).difference(zoom).difference(box(-60, -20, 60, 0))
    oog = ellipse(-11, 68, 5.5, 8)
    mond = ellipse(0, 47, 7, 9)
    return lijf.difference(unary_union([sym(oog), mond]))


def pompoen():
    """Uitgesneden pompoen: drie ribben, steeltje, driehoeksogen en een tandenmond."""
    lijf = unary_union([ellipse(0, 40, 30, 40), ellipse(-22, 38, 28, 37), ellipse(22, 38, 28, 37)])
    steel = Polygon([(-7, 72), (7, 72), (9, 88), (13, 92), (9, 97), (1, 92), (-3, 82)])
    lijf = smooth(unary_union([lijf, steel]), close=3)
    oog = Polygon([(-29, 46), (-11, 46), (-20, 61)])
    neus = Polygon([(-5, 36), (5, 36), (0, 45)])
    tand = [(-34, 30), (-24, 22), (-16, 26), (-8, 18), (0, 23), (8, 18), (16, 26), (24, 22), (34, 30)]
    mond = Polygon(tand + [(30, 18), (18, 10), (6, 13), (0, 8), (-6, 13), (-18, 10), (-30, 18)])
    return lijf.difference(unary_union([sym(oog), neus, mond]))


def kat():
    """Zittende zwarte kat van voren: spitse oren, amandelogen, staart die om de poten krult."""
    kop = ellipse(0, 70, 21, 18)
    oor = Polygon([(-20, 72), (-17, 98), (-4, 84)])
    lijf = unary_union([ellipse(0, 34, 25, 34), rounded_rect(-27, 0, 27, 22, 10)])
    staart = stroke([(24, 6), (34, 14), (36, 30), (31, 44)], 7)
    kat = smooth(unary_union([kop, sym(oor), lijf, staart]), close=4)
    oog = affinity.rotate(ellipse(-8.5, 71, 4.5, 6.5), -20, origin=(-8.5, 71))
    neus = Polygon([(-2.5, 63), (2.5, 63), (0, 66)])
    poot = rounded_rect(-1, 0, 1, 12, 1)               # spleet tussen de voorpootjes
    return kat.difference(unary_union([sym(oog), neus, poot]))


def grafsteen():
    """Grafsteen met afgeronde top, een kruis erin en een barst."""
    steen = unary_union([box(-26, 0, 26, 62), ellipse(0, 62, 26, 22)])
    steen = smooth(steen, open_=3)
    kruis = unary_union([rounded_rect(-3, 30, 3, 72, 1.2), rounded_rect(-12, 54, 12, 60, 1.2)])
    barst = stroke([(14, 0), (11, 9), (16, 16), (12, 24)], 2.2)
    return steen.difference(unary_union([kruis, barst]))


def tekst(s, smal=False):
    """Woord in Fredoka SemiBold (smal: de versmalde snit)."""
    return text(s, 100.0, 0, 50.0, font=FONT_SMAL if smal else _FP)


VORMEN = {"schedel": schedel, "spook": spook, "pompoen": pompoen, "kat": kat, "grafsteen": grafsteen}


def vorm(naam):
    """'schedel', 'spook', 'pompoen', 'kat', 'grafsteen', of een woord: 'tekst:BOO' / 'smal:TRICK'.

    Met '*factor' erachter wordt de vorm smaller of breder (b.v. 'pompoen*0.85'): handig als één vorm veel
    breder is dan de andere in een beeldje met vier schaduwen.
    """
    naam, _, factor = naam.partition("*")
    if naam in VORMEN:
        g = VORMEN[naam]()
    else:
        soort, woord = naam.split(":", 1)
        g = tekst(woord, smal=(soort == "smal"))
    return affinity.scale(g, float(factor), 1.0, origin=(0, 0)) if factor else g


def leesbaar(naam):
    """'pompoen*0.85' -> 'pompoen', 'smal:BOO' -> 'BOO'."""
    naam = naam.partition("*")[0]
    return naam.split(":", 1)[1] if ":" in naam else naam
