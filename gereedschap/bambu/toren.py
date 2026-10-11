"""Spoeltoren (prime tower) van Bambu Studio 2.x: maat, voetafdruk en een veilige plek op de plaat.

Bambu Studio 2.x bouwt standaard een vierkante toren met ribben (prime_tower_rib_wall = 1). De zijde
volgt uit het primevolume per filamentwissel en het aantal wissels per laag, niet uit
prime_tower_width (35 mm). Bij de hoeken steken de ribben uit en de eerste laag krijgt rondom een brim
van 3 mm. wipe_tower_x/y is de linkeronderhoek van de wand mét ribben; de brim ligt daarbuiten.
Komt één stukje toren of brim buiten 0..256 mm, dan meldt Bambu Studio "Een G-code pad treedt buiten
de grenzen van de printplaat".

Nagerekend uit bambulab/BambuStudio 02.08:
  src/libslic3r/GCode/WipeTower.cpp       plan_tower_new, set_toolchange, rib_section
  src/slic3r/GUI/PartPlate.cpp            estimate_wipe_tower_size: de schatting van de editor
  src/slic3r/GUI/WipeTowerPlacement.cpp   wipe_tower_legal_corner_range: 15 mm van elke rand
  src/libslic3r/Print.cpp                 toren tegen de convex hull van elk deel ("too close to others")
"""
import math

import numpy as np
import shapely
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union

BED = 256.0
MARGE = 15.0           # WIPE_TOWER_MARGIN: Bambu houdt de linkeronderhoek 15 mm van elke rand
TUSSENRUIMTE = 2.0     # vrije ruimte tussen de brim en een deel
MIN_DIEPTE = ((5.0, 5.0), (100.0, 20.0), (250.0, 40.0), (350.0, 60.0))   # WipeTower::min_depth_per_height


def _getal(cfg, sleutel):
    v = cfg[sleutel]
    v = max(float(x) for x in v) if isinstance(v, list) else v
    return float(str(v).rstrip("%"))


def parameters(cfg):
    """De torensleutels uit project_settings (alle waarden als strings, zoals Bambu ze schrijft)."""
    return {"breedte": _getal(cfg, "prime_tower_width"), "tussenruimte": _getal(cfg, "prime_tower_infill_gap") / 100,
            "rib": _getal(cfg, "prime_tower_rib_width"), "rib_extra": _getal(cfg, "prime_tower_extra_rib_length"),
            "brim": _getal(cfg, "prime_tower_brim_width"), "prime": _getal(cfg, "filament_prime_volume"),
            "laag": _getal(cfg, "layer_height"), "nozzle": _getal(cfg, "nozzle_diameter"),
            "ribben": cfg.get("prime_tower_rib_wall", "1") == "1"}


def min_diepte(hoogte):
    return next((d for h, d in MIN_DIEPTE if h >= hoogte), MIN_DIEPTE[-1][1])


def schatting(p, n, hoogte):
    """Zijde van de toren zoals de editor hem schat (estimate_wipe_tower_size), n = filamenten op de plaat."""
    if n < 2:
        return 0.0
    d = math.sqrt(p["prime"] * (n - 1) / p["laag"] * p["tussenruimte"])
    volume = d
    d = max(min_diepte(hoogte), d)
    rib = min(p["rib"], d / 2)
    return rib / math.sqrt(2) + max(d + p["rib_extra"], volume)


def bereik(p, n, hoogte):
    """Toegelaten linkeronderhoek (lo, hi), voor x en y gelijk: 15 mm van de randen voor de geschatte toren."""
    hi = max(MARGE, BED - schatting(p, max(n, 2), hoogte) - MARGE)
    return MARGE, hi


def wand(p, k, hoogte):
    """(breedte, diepte) van de eerste laag van de wand met ribben, bij k filamentwissels per laag."""
    k = max(k, 1)
    lijn = p["nozzle"] * 1.25                           # m_perimeter_width
    tussen = lijn * p["tussenruimte"]                   # lijnafstand in de toren (prime_tower_infill_gap)
    lengte = p["prime"] / (p["laag"] * (lijn - p["laag"] * (1 - math.pi / 4)))   # volume_to_length

    def per_wissel(b):
        return math.ceil(lengte / (b - 2 * lijn)) * tussen

    b = p["breedte"]
    if p["ribben"]:   # vierkant maken: zijde = wortel(diepte x breedte), afgerond op een lijnbreedte
        b = math.ceil(math.sqrt((k * per_wissel(b) + lijn) * b) / lijn) * lijn
    blok = k * per_wissel(b)
    d = blok + lijn
    if not p["ribben"]:
        return b, max(d, min_diepte(hoogte))
    diag = math.hypot(b, d)
    rib_lengte = max(min_diepte(hoogte) * math.sqrt(2) if blok < min_diepte(hoogte) else 0.0, diag) + p["rib_extra"]
    rib_lengte = max(rib_lengte, diag)
    half = min(p["rib"], min(b, d) / 2) / 2
    e = (rib_lengte - diag) / 2                         # verlenging van de ribben in de eerste laag
    ux, uy = b / diag, d / diag
    return b + 2 * (e * ux + half * uy), d + 2 * (e * uy + half * ux)


def voetafdruk(p, x, y, k, hoogte):
    """(x0, y0, x1, y1) van toren plus brim op de plaat, met de linkeronderhoek van de wand op (x, y)."""
    b, d = wand(p, k, hoogte)
    r = p["brim"]
    return x - r, y - r, x + b + r, y + d + r


def filamenten_per_laag(delen, laag=0.2):
    """Grootste aantal filamenten in één laag: een bovengrens voor het aantal wissels per laag.
    delen: [(slot, vertices, faces)] op hun plek op de plaat."""
    z_max = max(float(v[:, 2].max()) for _, v, _ in delen)
    n = max(1, math.ceil(z_max / laag - 1e-6))
    per_slot = {}
    for slot, v, f in delen:
        z = v[f][:, :, 2]
        lo = np.ceil(z.min(axis=1) / laag - 0.5 - 1e-6).astype(int)    # lagen waarvan het midden in de driehoek valt
        hi = np.floor(z.max(axis=1) / laag - 0.5 + 1e-6).astype(int)
        ok = hi >= lo
        verschil = np.zeros(n + 1, int)
        np.add.at(verschil, np.clip(lo[ok], 0, n), 1)
        np.add.at(verschil, np.clip(hi[ok] + 1, 0, n), -1)
        per_slot[slot] = per_slot.get(slot, np.zeros(n, bool)) | (np.cumsum(verschil)[:n] > 0)
    return int(np.sum(list(per_slot.values()), axis=0).max()) if per_slot else 0


def obstakels(delen):
    """Convex hull van elk deel (zoals Bambu de toren controleert), samengevoegd."""
    return unary_union([MultiPoint(v[:, :2]).convex_hull for _, v, _ in delen])


def plaats(p, n, k, hoogte, obstakel=None, voorkeur=(MARGE, BED)):
    """Linkeronderhoek (x, y) voor de toren: binnen het bereik van Bambu, met toren en brim op de plaat
    en vrij van `obstakel` (shapely, plaatcoördinaten). Van de vrije plekken de dichtste bij `voorkeur`
    (standaard linksboven, waar Bambu hem zelf zet). None als er geen plek is."""
    lo, hi = bereik(p, n, hoogte)
    b, d = wand(p, k, hoogte)
    r = p["brim"] + TUSSENRUIMTE
    lo = math.ceil(lo * 2) / 2                           # raster van 1 mm op halve millimeters: exacte getallen
    hi_x = math.floor(min(hi, BED - b - p["brim"] - 1) * 2) / 2
    hi_y = math.floor(min(hi, BED - d - p["brim"] - 1) * 2) / 2
    if hi_x < lo or hi_y < lo:
        return None
    xs = np.unique(np.append(np.arange(lo, hi_x, 1.0), hi_x))
    ys = np.unique(np.append(np.arange(lo, hi_y, 1.0), hi_y))
    X, Y = (a.ravel() for a in np.meshgrid(xs, ys))
    vrij = np.ones(len(X), bool)
    if obstakel is not None and not obstakel.is_empty:
        vrij = ~shapely.intersects(obstakel, shapely.box(X - r, Y - r, X + b + r, Y + d + r))
    if not vrij.any():
        return None
    afstand = np.hypot(X[vrij] - voorkeur[0], Y[vrij] - voorkeur[1])
    i = int(np.argmin(afstand))
    return float(X[vrij][i]), float(Y[vrij][i])


def controleer(p, x, y, n, k, hoogte, obstakel=None):
    """Lijst met problemen voor een toren op (x, y); leeg = in orde."""
    fouten = []
    lo, hi = bereik(p, n, hoogte)
    if not (lo - 0.05 <= x <= hi + 0.05 and lo - 0.05 <= y <= hi + 0.05):
        fouten.append(f"spoeltoren op ({x:g}, {y:g}) ligt buiten het gebied waar Bambu Studio hem houdt "
                      f"({lo:g}..{hi:.1f} mm voor {max(n, 2)} filamenten)")
    x0, y0, x1, y1 = voetafdruk(p, x, y, k, hoogte)
    if x0 < 0 or y0 < 0 or x1 > BED or y1 > BED:
        fouten.append(f"spoeltoren met brim ({x0:.1f}, {y0:.1f})..({x1:.1f}, {y1:.1f}) komt buiten de plaat")
    if obstakel is not None and not obstakel.is_empty and k >= 2:
        r = TUSSENRUIMTE
        if obstakel.intersects(box(x0 - r, y0 - r, x1 + r, y1 + r)):
            fouten.append(f"spoeltoren met brim ({x0:.1f}, {y0:.1f})..({x1:.1f}, {y1:.1f}) raakt een object")
    return fouten
