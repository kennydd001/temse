"""Valsimulatie van het valbord (2D, pymunk): waar komen de muntjes terecht, en blijft er ooit een hangen?

Het valbord is in het vlak (x, z) nagebouwd met de maten uit params.py: zijwanden met halve pinnen,
plafond met inworpgleuf, de pinnen, de verdeler en de kantelplank op de slinger. Een muntje is een schijf
die rechtop in de spleet valt; het wrijven langs voor- en achterplaat zit erin als demping. De slinger is
een echt lichaam (arm, gewicht, plank met naaf) aan een mes; de wrijving van het mes zit erin als een kleine
droge wrijving (gelijk aan een pen van 4 mm met wrijvingsgetal 0,02 .. 0,06).

Het kind kijkt van +y naar het bord: links voor het kind is +x. 'L' en 'links mikken' zijn steeds links voor het kind.

PLA op PLA is niet precies bekend, dus elke vraag draait over een reeks waarden voor stuit (0,3 .. 0,7),
wrijving (0,2 .. 0,5) en demping. Het antwoord is pas betrouwbaar als het over die hele reeks klopt.

Een ronde zoals in de klas: het kind geeft de slinger een duw, en gooit dan 8 muntjes, eentje om de
0,8 .. 2 seconden, en mikt daarbij steeds op dezelfde plek in de trechter (links, midden of rechts).

Gebruik:  python simulatie.py          (drukt de tabellen af)
"""
import itertools
import math
import random

import pymunk

import params as P

DT = 1 / 2000
RHO = 1.24e-3               # g/mm3, PLA
G = 9810.0                  # mm/s2

MATERIAAL = [dict(stuit=s, wrijving=f, demping=d) for s, f, d in
             itertools.product((0.3, 0.5, 0.7), (0.2, 0.5), (0.75, 0.95))]
PEN_WRIJVING = (0.02, 0.06)   # mes in V-groef, uitgedrukt als pen van 4 mm (een geprinte pen zou 0,2 .. 0,4 zijn)


def slinger_massa():
    """(massa g, afstand zwaartepunt onder de as mm, traagheid om de as g mm2)."""
    b, h, d = P.GEWICHT
    m_g = RHO * b * h * d
    m_a = RHO * P.ARM_B * P.SLINGER_L * 5.0 * 0.7          # arm met wat infill
    m_p = RHO * (2 * P.PLANK_HALF * P.PLANK_H * 5.0 + math.pi * P.NAAF_R ** 2 * 14.0)
    m = m_g + m_a + m_p
    d_z = (m_g * P.SLINGER_L + m_a * P.SLINGER_L / 2) / m
    I = m_g * (P.SLINGER_L ** 2 + (b * b + h * h) / 12) + m_a * P.SLINGER_L ** 2 / 3 + m_p * P.PLANK_HALF ** 2 / 3
    return m, d_z, I


def slingertijd():
    m, d, I = slinger_massa()
    return 2 * math.pi * math.sqrt(I / (m * G * d))


def zet_uitslag(sl, graden):
    """Zet de slinger op een uitslag (draaien om de as, niet om het zwaartepunt)."""
    sl.angle = math.radians(graden)
    sl.position = (0, P.Z_PLANK)
    sl.velocity, sl.angular_velocity = (0, 0), 0.0


def ruimte(stuit=0.5, wrijving=0.3, pen=0.3):
    sp = pymunk.Space()
    sp.gravity = (0, -G)
    st = sp.static_body
    W, top = P.BINNEN_HALF, P.PLAFOND_Z
    vorm = [pymunk.Segment(st, (s * W, -80), (s * W, top), 0.5) for s in (-1, 1)]
    vorm += [pymunk.Segment(st, (-W, top), (-P.INWORP_HALF, top), 0.5),
             pymunk.Segment(st, (P.INWORP_HALF, top), (W, top), 0.5)]
    vorm += [pymunk.Circle(st, P.PIN_D / 2, xz) for xz in P.pinnen()]
    vorm += [pymunk.Circle(st, P.BULT_R, xz) for xz in P.bulten()]
    h, t = P.VERDELER_T / 2, P.VERDELER_TOP
    vorm.append(pymunk.Poly(st, [(-h, -80), (h, -80), (h, t - h), (0, t), (-h, t - h)]))
    for v in vorm:
        v.elasticity, v.friction = stuit, wrijving
    sp.add(*vorm)
    # slinger met kantelplank: alleen de plank zit in de spleet bij de muntjes
    m, d, I = slinger_massa()
    sl = pymunk.Body(m, I - m * d * d)
    sl.position = (0, P.Z_PLANK)
    sl.center_of_gravity = (0, -d)
    plank = pymunk.Segment(sl, (-P.PLANK_HALF + P.PLANK_H / 2, 0), (P.PLANK_HALF - P.PLANK_H / 2, 0), P.PLANK_H / 2)
    naaf = pymunk.Circle(sl, P.NAAF_R)
    for v in (plank, naaf):
        v.elasticity, v.friction = stuit, wrijving
    sp.add(sl, plank, naaf)
    sp.add(pymunk.PivotJoint(st, sl, (0, P.Z_PLANK)))
    rem = pymunk.SimpleMotor(st, sl, 0.0)
    rem.max_force = pen * m * G * 2.0    # droge wrijving, als een pen van 4 mm
    sp.add(rem)
    return sp, sl


def nieuwe_munt(sp, rnd, mik, stuit, wrijving, demping=0.85):
    """Muntje in de inworp; demping (alleen op het muntje): het wrijven langs voor- en achterplaat."""
    r = P.MUNT_D / 2
    m = RHO * math.pi * r * r * P.MUNT_T
    b = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r))
    off = P.INWORP_HALF - r - 0.05
    b.position = (max(-off, min(off, mik + rnd.uniform(-1.0, 1.0))), P.PLAFOND_Z)
    b.velocity = (rnd.uniform(-25, 25), -rnd.uniform(0, 100))
    b.angular_velocity = rnd.uniform(-4, 4)
    b.velocity_func = lambda body, g, d, dt: pymunk.Body.update_velocity(body, g, demping ** dt, dt)
    c = pymunk.Circle(b, r)
    c.elasticity, c.friction = stuit, wrijving
    sp.add(b, c)
    return b, c


def ronde(rnd, mik=0.0, n=8, stuit=0.5, wrijving=0.3, demping=0.85, pen=0.3, duw=(25.0, 40.0)):
    """Eén duw, dan n muntjes. Geeft (uitkomsten 'L'/'R'/'vast', uitslag van de slinger bij het laatste muntje)."""
    sp, sl = ruimte(stuit, wrijving, pen)
    zet_uitslag(sl, rnd.uniform(*duw) * rnd.choice((-1, 1)))
    uit, uitslag = [], 0.0
    for _ in range(n):
        hoogste = 0.0
        for _ in range(int(rnd.uniform(0.8, 2.0) / DT)):     # het kind pakt het volgende muntje
            sp.step(DT)
            hoogste = max(hoogste, abs(sl.angle))
        uitslag = math.degrees(hoogste)
        b, c = nieuwe_munt(sp, rnd, mik, stuit, wrijving, demping)
        t, stil = 0.0, 0.0
        while b.position.y > -40 and t < 8.0:
            sp.step(DT)
            t += DT
            stil = stil + DT if b.velocity.length < 2.0 else 0.0
            if stil > 2.0:
                break
        uit.append(("L" if b.position.x > 0 else "R") if b.position.y <= -40 else "vast")   # links voor het kind: +x
        sp.remove(b, c)
    return uit, uitslag


def mikproef(rondes=12, seed=1):
    """p(links) per materiaalset en mikpunt; hoeveel muntjes bleven hangen.

    De drie mikpunten krijgen dezelfde duwen en dezelfde tijden tussen de muntjes: het verschil tussen links
    en rechts mikken is dan echt het effect van het mikken, geen toeval."""
    off = P.INWORP_HALF - P.MUNT_D / 2 - 0.05
    tabel, vast, uitslagen = [], 0, []
    for i, mat in enumerate(MATERIAAL):
        for j, pen in enumerate(PEN_WRIJVING):
            rij = []
            for mik in (off, 0.0, -off):                  # links, midden, rechts voor het kind
                rnd = random.Random(seed * 10000 + i * 10 + j)
                ls = 0
                for _ in range(rondes):
                    u, a = ronde(rnd, mik, pen=pen, **mat)
                    ls += u.count("L")
                    vast += u.count("vast")
                    uitslagen.append(a)
                rij.append(ls / (rondes * 8))
            tabel.append((mat, pen, rij))
    return tabel, vast, uitslagen


def splitsingen(rondes=300, n=8, seed=2):
    """Hoe vaak komt elke splitsing van n voor (gemengde materiaalsets, mikpunt willekeurig)."""
    rnd = random.Random(seed)
    off = P.INWORP_HALF - P.MUNT_D / 2 - 0.05
    telling = [0] * (n + 1)
    for i in range(rondes):
        mat = MATERIAAL[i % len(MATERIAAL)]
        u, _ = ronde(rnd, rnd.uniform(-off, off), n=n, pen=PEN_WRIJVING[i % 2], **mat)
        if "vast" not in u:
            telling[u.count("L")] += 1
    return telling


def binomiaal(n, p=0.5):
    return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


def uitdemping(pen, duw=35.0, tijd=30.0):
    """Uitslag van de slinger (graden) na elke 5 s, zonder muntjes."""
    sp, sl = ruimte(pen=pen)
    zet_uitslag(sl, duw)
    out, t, hoogste = [], 0.0, 0.0
    while t < tijd:
        sp.step(DT)
        t += DT
        hoogste = max(hoogste, abs(sl.angle))
        if abs(t / 5 - round(t / 5)) < DT / 10 and t > 0.1:
            out.append(round(math.degrees(hoogste), 1))
            hoogste = 0.0
    return out


if __name__ == "__main__":
    print(f"slingertijd {slingertijd():.2f} s, massa {slinger_massa()[0]:.0f} g")
    for pen in PEN_WRIJVING:
        print("uitdemping, pen", pen, uitdemping(pen))
    tabel, vast, a = mikproef(rondes=6)
    ps = [p for *_, rij in tabel for p in rij]
    print(f"p(links) over alle sets en mikpunten: {min(ps):.2f} .. {max(ps):.2f}, vast {vast}, uitslag laatste muntje min {min(a):.0f}°")
    t = splitsingen(rondes=120)
    print("splitsingen van 8:", t, [round(v * sum(t), 1) for v in binomiaal(8)])
