"""Evolvente tandwielen en heugels als 2D-vormen (shapely), voor FDM-printbare mechanismen.

tandwiel(m, z)  wiel met module m en z tanden, middelpunt (0, 0), een tand recht naar rechts (hoek 0)
heugel(m, n)    heugel langs de y-as, steek op x = 0, tanden naar +x; tand 0 op y = 0

Speling (`speling`, mm) gaat van de tanddikte af, gemeten op de steekcirkel / steeklijn; met twee
delen elk 0,1 heb je samen 0,2 mm, wat voor PLA op een 0,4 mm-nozzle een soepele maar niet
rammelende ingrijping geeft. Drukhoek standaard 20°.
"""
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def _evolvente(rb, r):
    """Hoek (rad) van het evolventepunt op straal r, vanaf het begin op de basiscirkel."""
    if r <= rb:
        return 0.0
    return math.sqrt((r / rb) ** 2 - 1) - math.acos(rb / r)


def tandwiel(m, z, speling=0.1, drukhoek=20.0, kop=1.0, voet=1.25, n=12):
    r = m * z / 2
    rb = r * math.cos(math.radians(drukhoek))
    ra = r + kop * m
    rf = max(r - voet * m, 0.5)
    s = math.pi * m / 2 - speling                  # tanddikte op de steekcirkel
    half = s / (2 * r) + _evolvente(rb, r)          # halve tandhoek op de basiscirkel
    flank = [(rr, half - _evolvente(rb, rr)) for rr in np.linspace(max(rb, rf), ra, n)]
    if rf < rb:                                     # radiaal stukje flank onder de basiscirkel
        flank = [(rf, half)] + flank
    pts = []
    for k in range(z):
        a0 = 2 * math.pi * k / z
        onder = [(rr * math.cos(a0 - h), rr * math.sin(a0 - h)) for rr, h in flank]            # voet -> kop
        boven = [(rr * math.cos(a0 + h), rr * math.sin(a0 + h)) for rr, h in reversed(flank)]  # kop -> voet
        # voetboog naar de volgende tand
        b0, b1 = a0 + half, a0 + 2 * math.pi / z - half
        boog = [(rf * math.cos(t), rf * math.sin(t)) for t in np.linspace(b0, b1, 6)[1:-1]]
        pts += onder + boven + boog
    return Polygon(pts).buffer(0).simplify(0.003)


def heugel(m, n, speling=0.1, drukhoek=20.0, kop=1.0, voet=1.25, dikte=4.0, eerste=0):
    """Heugel met n tanden (nummer eerste .. eerste+n-1), tanden naar +x, rug tot x = -(voet*m + dikte)."""
    p = math.pi * m
    tan = math.tan(math.radians(drukhoek))
    s = p / 2 - speling                             # tanddikte op de steeklijn
    ha, hf = kop * m, voet * m
    pts = []
    for k in range(eerste, eerste + n):
        y0 = k * p
        # trapezium: op de steeklijn breedte s, flanken onder de drukhoek
        pts += [(-hf, y0 - s / 2 - hf * tan), (ha, y0 - s / 2 + ha * tan),
                (ha, y0 + s / 2 - ha * tan), (-hf, y0 + s / 2 + hf * tan)]
    lo, hi = pts[0][1], pts[-1][1]
    rug = [(-hf - dikte, hi), (-hf - dikte, lo)]
    return Polygon(pts + rug).buffer(0)


def sector(wiel, m, z, hoek0, hoek1, naaf=0.0):
    """Alleen de tanden tussen hoek0 en hoek1 (graden), plus een volle naaf met straal `naaf`."""
    ra = m * z / 2 + 1.0 * m + 1
    a = np.radians(np.linspace(hoek0, hoek1, 64))
    taart = Polygon([(0, 0)] + [(ra * math.cos(t), ra * math.sin(t)) for t in a])
    stuk = wiel.intersection(taart)
    return unary_union([stuk, Point(0, 0).buffer(naaf, quad_segs=32)]) if naaf else stuk


def draai(g, graden, om=(0, 0)):
    return affinity.rotate(g, graden, origin=om)
