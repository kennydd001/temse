"""45°-zelfdragend maken van een 2D-vorm die rechtop geprint wordt (gedeeld door alle modellen).

Een vorm in het vlak (x = breedte, y = hoogte) die als rechtopstaande wand of als silhouet geprint wordt,
moet zonder support kunnen: geen overhang steiler dan 45°, gaten met een spitse bovenkant, en niets dat in
de lucht begint. zelfdragend() past de vorm daarvoor zo weinig mogelijk aan. Gebruikt door H-01 (de
schaduwsilhouetten) en H-02 (de gezichten in de wanden).

De hulpjes werken met stukken per rij: een gesorteerde lijst intervallen [(a, b), ...].
"""
import math

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


# --- intervallen per rij -----------------------------------------------------------------------

def rij(g, z, x0, x1):
    """Stukken [a, b] van vorm g op hoogte z, gesorteerd."""
    s = g.intersection(LineString([(x0, z), (x1, z)]))
    if s.is_empty:
        return []
    lijnen = getattr(s, "geoms", [s])
    uit = sorted((min(c[0][0], c[-1][0]), max(c[0][0], c[-1][0])) for c in (list(l.coords) for l in lijnen)
                 if len(c) >= 2)
    return samen(uit)


def samen(iv, tol=1e-9):
    uit = []
    for a, b in sorted(iv):
        if b - a <= tol:
            continue
        if uit and a <= uit[-1][1] + tol:
            uit[-1] = (uit[-1][0], max(uit[-1][1], b))
        else:
            uit.append((a, b))
    return uit


def snij(p, q):
    uit, i, j = [], 0, 0
    while i < len(p) and j < len(q):
        a, b = max(p[i][0], q[j][0]), min(p[i][1], q[j][1])
        if b > a:
            uit.append((a, b))
        if p[i][1] < q[j][1]:
            i += 1
        else:
            j += 1
    return uit


def zonder(p, q):
    """p zonder q."""
    uit = []
    for a, b in p:
        stukken = [(a, b)]
        for c, d in q:
            nieuw = []
            for x, y in stukken:
                if d <= x or c >= y:
                    nieuw.append((x, y))
                    continue
                if c > x:
                    nieuw.append((x, c))
                if d < y:
                    nieuw.append((d, y))
            stukken = nieuw
        uit += stukken
    return samen(uit)


def lengte(iv):
    """Totale lengte van de stukken."""
    return sum(b - a for a, b in iv)


# --- 45° zelfdragend --------------------------------------------------------------------------

def zelfdragend(g, res=0.1, hoek=45.0, rust=0.8, fase=0.0, brug=0.0):
    """Vul overhangen op met 45°-wiggen, geef gaten een spitse bovenkant en zet niets in de lucht.

    Van boven naar beneden, per rij van `res` mm, twee regels voor elke rij ten opzichte van de rij erboven:
      - overhang: de lege ruimte mag per rij hooguit dx = res·tan(hoek) per kant breder worden. Een gat dat
        hier pas begint, begint als een punt in het midden van zijn bovenste rij (een spitse boog);
      - rusten: elk stuk materiaal in de rij erboven ligt met minstens `rust` mm (of zijn hele breedte, als
        het smaller is) op materiaal in deze rij. Een smal stuk dat boven een gat hangt (een tand boven de
        mondspleet) krijgt zo een pilaartje tot op de bodem, en een los zwevend stuk (een letter boven de
        voetbalk) wordt eronder bijgevuld tot het ergens op staat.
    Wat volgens die regels niet leeg mag zijn, wordt materiaal. `fase` verschuift het rijenraster (in rijen):
    geef het tweede silhouet een halve rij, dan liggen hun trapjes nooit op dezelfde hoogte.

    `brug` (mm): een gat in een wand mag een plat plafond hebben tot zo breed; de printer overbrugt dat
    tussen de twee zijkanten. Een breder gat begint als een plat stuk van die breedte in het midden en
    groeit vandaar onder 45°. Voor een silhouet dat in de lucht hangt (H-01) blijft het 0.
    """
    dx = res * math.tan(math.radians(hoek))
    x0, z0, x1, z1 = g.bounds
    rand = (x0 - 1, x1 + 1)
    vol_boven = []
    rijen = []                                 # (z, materiaal) van boven naar beneden
    for z in np.arange(z1 - res / 2 - fase * res, z0, -res):
        vol = rij(g, z, *rand)
        leeg = zonder([rand], vol)
        groei = samen([(a - dx, b + dx) for a, b in zonder([rand], vol_boven)])
        nieuw = snij(leeg, groei)
        for a, b in leeg:                      # een gat dat hier begint: punt (of korte brug) in het midden
            if not snij([(a, b)], groei):
                c, half = (a + b) / 2, max(dx, brug / 2)
                nieuw.append((max(a, c - half), min(b, c + half)))
        mat = zonder([rand], samen(nieuw))
        for a, b in vol_boven:                 # elk stuk erboven moet ergens op rusten
            nodig = min(b - a, rust)
            if lengte(snij(mat, [(a, b)])) < nodig - 1e-9:
                c = (a + b) / 2
                mat = samen(mat + [(c - nodig / 2, c + nodig / 2)])
        rijen.append((z, mat))
        vol_boven = mat
    # tussen twee rijen een trapezium per paar overlappende stukken: schuine randen worden rechte lijnen
    stukken = []
    for (za, ma), (zb, mb) in zip(rijen, rijen[1:]):
        for a, b in ma:
            for c, d in mb:
                if max(a, c) < min(b, d):
                    stukken.append(Polygon([(c, zb), (d, zb), (b, za), (a, za)]))
    for z, m in (rijen[0], rijen[-1]):          # halve rij boven en onder
        stukken += [box(a, z - res / 2, b, z + res / 2) for a, b in m]
    uit = unary_union(stukken + [g])
    # haarfijne randjes (< 0,2 mm) in de punt van een boog kan geen printer maken; een gat dat de rand in één
    # punt raakt wordt in 3D een rand die twee delen delen (niet-manifold): openen en sluiten
    uit = uit.buffer(-0.1, join_style="mitre").buffer(0.1, join_style="mitre").union(g)
    uit = uit.buffer(0.03, join_style="mitre").buffer(-0.03, join_style="mitre")
    return uit.simplify(0.01)
