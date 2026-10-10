"""Kleine geometrie-bibliotheek: 2D met shapely, 3D met trimesh + manifold3d."""
import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, OpType
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

FONTS = Path(__file__).resolve().parent / "fonts"
_FP = FontProperties(fname=str(FONTS / "Fredoka-SemiBold.ttf"))
FONT_SMAL = FontProperties(fname=str(FONTS / "Fredoka-SemiBold-Condensed.ttf"))  # wdth 75

# --- 2D -------------------------------------------------------------------

def ellipse(cx, cy, rx, ry, rot=0.0, n=128):
    e = affinity.scale(Point(0, 0).buffer(1.0, quad_segs=n // 4), rx, ry)
    if rot:
        e = affinity.rotate(e, rot, origin=(0, 0))
    return affinity.translate(e, cx, cy)


def rounded_rect(x0, y0, x1, y1, r):
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def smooth(g, close=0.0, open_=0.0):
    """Afronden: close vult binnenhoeken, open_ rondt buitenhoeken af."""
    if close:
        g = g.buffer(close, quad_segs=16).buffer(-close, quad_segs=16)
    if open_:
        g = g.buffer(-open_, quad_segs=16).buffer(open_, quad_segs=16)
    return g


def stroke(points, width):
    return LineString(points).buffer(width / 2, quad_segs=12, cap_style="round", join_style="round")


def arc_points(cx, cy, r, a0, a1, n=40, ry=None):
    ry = r if ry is None else ry
    t = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + r * np.cos(t), cy + ry * np.sin(t)))


def mirror_x(g):
    return affinity.scale(g, -1, 1, origin=(0, 0))


def sym(g):
    return unary_union([g, mirror_x(g)])


def text(s, cap_h, cx=0.0, cy=0.0, max_w=None, font=_FP):
    """Tekst als (Multi)Polygon, gecentreerd op (cx, cy) met hoofdletterhoogte cap_h."""
    ref = TextPath((0, 0), "H", size=100, prop=font).get_extents()
    size = 100 * cap_h / ref.height
    path = TextPath((0, 0), s, size=size, prop=font)
    raw = [p for p in path.to_polygons() if len(p) >= 3]
    signed = [0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) for p in raw]
    outer_sign = np.sign(signed[int(np.argmax(np.abs(signed)))])
    rings = [Polygon(p).buffer(0) for p in raw]
    # niet-nul-regel: contouren in de richting van de grootste zijn vulling, de rest gaten
    outers = [r for r, a in zip(rings, signed) if np.sign(a) == outer_sign]
    holes = [r for r, a in zip(rings, signed) if np.sign(a) != outer_sign]
    g = unary_union(outers)
    if holes:
        g = g.difference(unary_union(holes))
    x0, y0, x1, y1 = g.bounds
    if max_w and (x1 - x0) > max_w:
        g = affinity.scale(g, max_w / (x1 - x0), 1, origin=(x0, y0))
        x0, y0, x1, y1 = g.bounds
    # verticaal centreren op de hoofdletterhoogte (niet op de bbox), horizontaal op de bbox
    return affinity.translate(g, cx - (x0 + x1) / 2, cy - cap_h / 2)


def cs_to_shapely(cs):
    """manifold3d CrossSection -> shapely (positieve lussen zijn vulling, negatieve gaten)."""
    fills, holes = [], []
    for loop in cs.to_polygons():
        if len(loop) < 3:
            continue
        p = Polygon(loop)
        (fills if p.exterior.is_ccw else holes).append(p.buffer(0))
    out = unary_union(fills)
    return out.difference(unary_union(holes)) if holes else out


def polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


# --- 3D -------------------------------------------------------------------

def to_manifold(tm):
    return Manifold(Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                         tri_verts=np.asarray(tm.faces, dtype=np.uint32)))


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=False)


def extrude(g, z0, z1):
    """2D-vorm extruderen van z0 tot z1 -> Manifold."""
    parts = []
    for p in polys(g.buffer(0)):
        if p.area < 1e-6:
            continue
        tm = trimesh.creation.extrude_polygon(p, z1 - z0)
        tm.apply_translation([0, 0, z0])
        parts.append(to_manifold(tm))
    return union(parts)


def vlak_xz(g, y0, y1):
    """2D-vorm in het x-z-vlak (vooraanzicht: eerste coördinaat x, tweede z) extruderen van y0 tot y1."""
    m = extrude(g, -y1, -y0)
    return m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float))


def vlak_yz(g, x0, x1):
    """2D-vorm in het y-z-vlak (zijaanzicht: eerste coördinaat y, tweede z) extruderen van x0 tot x1."""
    m = extrude(g, x0, x1)
    return m.transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))


def druppel(c, r, omhoog=False, n=24):
    """Cirkel met een 45°-punt, voor liggende gaten en pennen die zonder support moeten printen.

    Een uitsteeksel krijgt de punt onder (omhoog=False); een gat de punt boven (dan is het plafond schuin).
    """
    cx, cz = c
    s = 1 if omhoog else -1
    q = r / math.sqrt(2)
    return unary_union([Point(c).buffer(r, quad_segs=n),
                        Polygon([(cx - q, cz + s * q), (cx, cz + s * r * math.sqrt(2)), (cx + q, cz + s * q), (cx, cz)])])


def extrude_profile(profile, length, matrix):
    """Profiel (2D in het lokale XY-vlak) over `length` in lokale Z, daarna 4x4-transformatie."""
    m = extrude(profile, 0.0, length)
    return m.transform(np.asarray(matrix, dtype=float)[:3, :])


def union(items):
    items = [i for i in items if i is not None and not i.is_empty()]
    if not items:
        return Manifold()
    return Manifold.batch_boolean(items, OpType.Add) if len(items) > 1 else items[0]


def afgeronde_plaat(x0, y0, x1, y1, z0, z1, r_hoek, r_rand, n=48):
    """Plaat met afgeronde hoeken (r_hoek, in het grondvlak) en een afgeronde bovenrand (r_rand); onderkant vlak.

    Gebouwd als bolle omhullende van vier gedraaide 'puckjes' op de hoekmiddelpunten: geen splinters
    waar de afrondingen elkaar treffen (zoals bij het doorsnijden van twee afgeronde extrusies).
    """
    from manifold3d import CrossSection
    r_rand = min(r_rand, z1 - z0, r_hoek)
    prof = [(0.0, z0), (r_hoek, z0), (r_hoek, z1 - r_rand)]
    for a in np.linspace(0, math.pi / 2, 12)[1:]:
        prof.append((r_hoek - r_rand + r_rand * math.cos(a), z1 - r_rand + r_rand * math.sin(a)))
    prof.append((0.0, z1))
    puck = Manifold.revolve(CrossSection([prof]), n)
    hoeken = [(x0 + r_hoek, y0 + r_hoek), (x1 - r_hoek, y0 + r_hoek), (x1 - r_hoek, y1 - r_hoek), (x0 + r_hoek, y1 - r_hoek)]
    return Manifold.batch_hull([puck.translate([x, y, 0]) for x, y in hoeken])


def box3(x0, y0, z0, x1, y1, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def translate(dx, dy, dz):
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m


def save_stl(m, path):
    # 1 µm opschonen: haalt splinterdriehoekjes weg die in een STL (float32) zouden samenvallen
    tm = to_trimesh(m.simplify(1e-3))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(path)
    return tm
