"""Renders van H-03 Poepmonster (Blender/Cycles, zonder scherm).

Gebruik:  python render.py [voor|boven|varianten|alle] [--snel]
Schrijft naar renders/:
  voor.png        de hoge variant, schuin van voren, half vol gekleurde poep
  boven.png       van achteren en van boven: de mond met de tanden
  varianten.png   laag en hoog naast elkaar

De 'poep' is nagemaakt: willekeurige platte klodders in filamentkleuren (vaste seed, dus elke keer dezelfde).
"""
import math
import random
import sys
from pathlib import Path

from manifold3d import Manifold

import _pad  # noqa: F401
import monster as M
import params as P
from gereedschap import blender as BL

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "renders"
POEPKLEUREN = ["#C12E1F", "#0A2989", "#00AE42", "#F4EE2A", "#FFFFFF", "#5E43B7", "#F55A74", "#0086D6", "#161616",
               "#FF6A13", "#9D432C", "#A3D8E1"]


def klodder(rng):
    """Eén stukje purge-afval: een platte, wat gedraaide klodder."""
    k = Manifold.sphere(1.0, 24).scale([rng.uniform(6, 10), rng.uniform(5, 8), rng.uniform(1.6, 2.8)])
    for _ in range(rng.randint(1, 3)):                      # opgekrulde randjes en een staartje
        k = k + Manifold.sphere(1.0, 16).scale([rng.uniform(2.5, 5), rng.uniform(1.5, 3), rng.uniform(1.2, 2.2)]).rotate(
            [0, 0, rng.uniform(0, 360)]).translate([rng.uniform(-7, 7), rng.uniform(-5, 5), rng.uniform(0.5, 2.5)])
    return k.rotate([rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(0, 360)])


def poep(sleutel, vulling, seed=7, buiten=0, dx=0.0):
    """Klodders in de bak tot `vulling` (0-1) van de achterrand, en `buiten` op tafel. dx: bak opzij geschoven."""
    rng = random.Random(seed)
    mats = [BL.pla(f"poep{i}", c, 0.35) for i, c in enumerate(POEPKLEUREN)]
    top = P.VLOER + vulling * (P.VARIANTEN[sleutel][2] - P.VLOER)
    z, n = P.VLOER + 3.0, 0
    while z < top:
        r = M.r_romp(z) * (1 - P.RIB) - P.WAND - 7.0
        for _ in range(int(r * r / 55)):
            a, d = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())
            BL.voeg_toe(f"poep{seed}_{n}", klodder(rng).translate([dx + d * math.cos(a), d * math.sin(a), z]),
                        rng.choice(mats))
            n += 1
        z += 3.2
    for i in range(buiten):
        a = rng.uniform(-0.3, 0.9) * math.pi
        d = rng.uniform(70, 95)
        BL.voeg_toe(f"los{seed}_{i}", klodder(rng).translate([dx + d * math.cos(a), d * math.sin(a), 2.5]),
                    rng.choice(mats))


def scene(sleutels, vulling=0.55, buiten=0):
    BL.leeg()
    mats = BL.materialen(P.FILAMENTEN)
    for i, k in enumerate(sleutels):
        dx = (i - (len(sleutels) - 1) / 2) * 135.0
        BL.voeg_toe(k, M.monster(k).translate([dx, 0, 0]), mats[P.ORANJE])
        poep(k, vulling, 7 + i, buiten if i == len(sleutels) - 1 else 0, dx)
    BL.studio(muur_y=0.55)


def voor(snel):
    scene(["hoog"], buiten=3)
    BL.camera((-0.13, -0.40, 0.19), (0.0, 0.0, 0.055), lens=55, diafragma=8.0)
    BL.render(UIT / "voor.png", (1200, 1000), samples=128, snel=snel)


def boven(snel):
    scene(["hoog"], vulling=0.7)
    BL.camera((0.16, 0.24, 0.36), (0.0, 0.0, 0.05), lens=55, diafragma=8.0)
    BL.render(UIT / "boven.png", (1000, 1000), samples=128, snel=snel)


def varianten(snel):
    scene(["laag", "hoog"])
    BL.camera((-0.05, -0.55, 0.22), (0.0, 0.0, 0.05), lens=50, diafragma=8.0)
    BL.render(UIT / "varianten.png", (1400, 900), samples=128, snel=snel)


if __name__ == "__main__":
    argv = sys.argv[1:]
    snel = "--snel" in argv
    wat = [a for a in argv if not a.startswith("--")] or ["alle"]
    UIT.mkdir(exist_ok=True)
    for naam, f in (("voor", voor), ("boven", boven), ("varianten", varianten)):
        if wat[0] in (naam, "alle"):
            f(snel)
