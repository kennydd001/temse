"""Printbaarheid van één printplaat op de Bambu Lab A1 (gedeeld door alle modellen).

plaat({kleur: Manifold}) meet, op de onderdelen zoals ze op het bed liggen (z = 0 is het bed):
  - de maat en of hij op de A1 past (256 x 256, 3 mm rand)
  - overhang: vlakken die steiler dan ~48° naar beneden kijken en niet op het bed liggen
    (een kleur die op een andere kleur ligt, telt als ondersteund); korte bruggen < 3 mm apart
  - details smaller dan de nozzle: per laag de doorsnede openen met een halve nozzle
  - lagen met meer dan één kleur (AMS-wissels)
"""
import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geom import cs_to_shapely, to_trimesh, union

BED = (256.0, 256.0)
NOZZLE = 0.4
LAYER = 0.2


def plaat(delen, nozzle=NOZZLE, laag=LAYER):
    info = {"kleuren": {}, "overhang": [], "dunne_details": []}
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    sections, solids = {}, []
    for color, m in delen.items():
        tm = to_trimesh(m)
        solids.append(m)
        lo, hi = np.minimum(lo, tm.bounds[0]), np.maximum(hi, tm.bounds[1])
        info["kleuren"][color] = {"volume_cm3": round(float(tm.volume) / 1000, 2),
                                  "z_mm": [round(float(tm.bounds[0][2]), 2), round(float(tm.bounds[1][2]), 2)]}
        # dunne details: per laag doorsnede, openen met halve nozzle; wat verdwijnt is te smal
        z0, z1 = tm.bounds[0][2], tm.bounds[1][2]
        lost_max, where = 0.0, None
        for z in (np.arange(math.ceil(z1 / laag)) + 0.5) * laag:  # vast lagenraster voor alle kleuren
            if not z0 < z < z1:
                continue
            s = cs_to_shapely(m.slice(z))
            if s.is_empty:
                continue
            sections.setdefault(round(z, 2), set()).add(color)
            opened = s.buffer(-nozzle / 2 + 0.01).buffer(nozzle / 2 - 0.01)
            lost = s.difference(opened.buffer(0.02))
            for p in getattr(lost, "geoms", [lost]):
                if p.area > lost_max:
                    lost_max, where = p.area, (round(p.centroid.x, 1), round(p.centroid.y, 1), round(z, 2))
        if lost_max > 0.2:  # kleiner dan 0,2 mm²: slicer laat het weg, niet zichtbaar
            info["dunne_details"].append({"kleur": color, "grootste_te_smalle_stuk_mm2": round(lost_max, 2),
                                          "plek_mm": [float(v) for v in where]})
    tm = to_trimesh(union(solids).simplify(1e-3))
    n, zc = tm.face_normals, tm.triangles_center[:, 2]
    edge = (n[:, 2] < -0.64) & (n[:, 2] >= -0.74) & (zc > 0.3)   # 40-48°: bewust 45°-vlakken
    info["vlakken_rond_45_graden_mm2"] = round(float(tm.area_faces[edge].sum()), 0)
    mask = (n[:, 2] < -0.74) & (zc > 0.3)
    info["korte_bruggen_mm2"] = 0.0
    if mask.any():
        sub = tm.submesh([np.where(mask)[0]], append=True)
        bridges = 0.0
        for comp in sub.split(only_watertight=False):
            if comp.area < 1.0:
                continue
            # helling t.o.v. horizontaal: 0° = plafond, 45° = schuine lip, 90° = wand
            ang = math.degrees(math.acos(min(1, -float(np.mean(comp.face_normals[:, 2])))))
            b = comp.bounds
            # korte brug: (bijna) horizontaal plafond dat smaller is dan 3 mm, b.v. verzonken letters achterop
            foot = unary_union([Polygon(t[:, :2]).buffer(0) for t in comp.triangles])
            span = 2 * foot.area / max(foot.length, 1e-6)
            if ang < 15 and span <= 3.0:
                bridges += float(comp.area)
                continue
            info["overhang"].append({"oppervlak_mm2": round(float(comp.area), 1),
                                     "hoek_tov_horizontaal_graden": round(ang, 1),
                                     "plek_mm": [round(float(v), 1) for v in (b[0] + b[1]) / 2]})
        info["korte_bruggen_mm2"] = round(bridges, 0)
    size = hi - lo
    info["maat_mm"] = [round(float(v), 1) for v in size]
    info["past_op_A1"] = bool(size[0] <= BED[0] - 6 and size[1] <= BED[1] - 6)
    info["vrije_strook_mm"] = [round(BED[0] - float(size[0]), 1), round(BED[1] - float(size[1]), 1)]
    multi = sorted(z for z, cs in sections.items() if len(cs) > 1)
    info["lagen_met_kleurwissel"] = len(multi)
    info["geschatte_wissels"] = int(sum(len(sections[z]) - 1 for z in multi))
    if multi:
        info["kleurwissel_tussen_z_mm"] = [multi[0], multi[-1]]
    return info
