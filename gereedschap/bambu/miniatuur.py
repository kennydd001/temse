"""Miniatuur (bovenaanzicht, 512 x 512 PNG) van een printplaat voor in het Bambu Studio-project."""
import io

import matplotlib
import numpy as np
from manifold3d import Manifold, Mesh
from shapely import affinity

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

from ..geom import cs_to_shapely  # noqa: E402


def miniatuur(plaat, filamenten):
    """512 x 512 bovenaanzicht van een plaat, met de echte filamentkleuren.

    Doorsneden van laag naar hoog over elkaar geschilderd (met echte gaten): wat bovenop ligt, zie je.
    """
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_patch(plt.Rectangle((0, 0), 256, 256, color="#2E2E33"))
    kleur = {i + 1: f.kleur for i, f in enumerate(filamenten)}
    delen_ = []
    for obj in plaat.objecten:
        a = np.vstack([d.vertices for d in obj.delen])
        dx = obj.positie[0] - (a[:, 0].min() + a[:, 0].max()) / 2
        dy = obj.positie[1] - (a[:, 1].min() + a[:, 1].max()) / 2
        for d in obj.delen:
            m = Manifold(Mesh(vert_properties=d.vertices.astype(np.float32), tri_verts=d.faces.astype(np.uint32)))
            delen_.append((m, dx, dy, kleur[d.slot]))
    zs = np.unique(np.round(np.concatenate([d.vertices[:, 2] for o in plaat.objecten for d in o.delen]), 2))
    niveaus = (zs[:-1] + zs[1:]) / 2 if len(zs) <= 41 else np.linspace(zs[0], zs[-1], 42)[1:-1]
    for z in niveaus:
        for m, dx, dy, kl in delen_:
            vorm = affinity.translate(cs_to_shapely(m.slice(float(z))), dx, dy)
            for poly in getattr(vorm, "geoms", [vorm]):
                if poly.is_empty or not hasattr(poly, "exterior"):
                    continue
                poly = orient(poly, 1.0)
                pad = MPath.make_compound_path(*[MPath(np.asarray(r.coords), closed=True)
                                                 for r in [poly.exterior, *poly.interiors]])
                ax.add_patch(PathPatch(pad, facecolor=kl, edgecolor="none"))
    if len({d.slot for o in plaat.objecten for d in o.delen}) > 1:
        ax.add_patch(plt.Rectangle(plaat.toren, 35, 35, fill=False, ec="#BBBBBB", ls="--", lw=1.2))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#2E2E33")
    plt.close(fig)
    return buf.getvalue()
