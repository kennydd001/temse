"""Snelle 2D-schets van de dierenfronten (zonder 3D), om tekeningen bij te schaven.

Gebruik:  python schets.py [dier ...] [--uit pad.png]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import kalender as K  # noqa: E402
from dieren import ALLE, PER_SLEUTEL  # noqa: E402
from params import CARD_W, FILAMENTEN, RIM_H, CARD_H, WINDOW_Y0  # noqa: E402
from talen import TALEN  # noqa: E402


def teken(ax, dier, taal="nl"):
    T = K.tekening(dier)
    board = K.bord2d()
    plate = unary_union([board, K.tongue(188), T.silhouet])

    def fill(g, code, z, ec="none"):
        """Vlak met echte gaten (PathPatch), zodat b.v. de mazen van een web open blijven."""
        col = FILAMENTEN[code][2]
        for p in getattr(g, "geoms", [g]):
            if p.is_empty or not hasattr(p, "exterior"):
                continue
            p = orient(p, 1.0)   # buitenrand linksom, gaten rechtsom: dan blijven gaten open
            ringen = [np.asarray(p.exterior.coords)] + [np.asarray(r.coords) for r in p.interiors]
            pad = MPath.make_compound_path(*[MPath(r, closed=True) for r in ringen])
            ax.add_patch(PathPatch(pad, facecolor=col, edgecolor=ec, lw=0.5, zorder=z))

    fill(plate, dier.bord, 0, ec="#555")
    if dier.kop != dier.bord:
        fill(T.silhouet.difference(board), dier.kop, 0.5, ec="#555")
    channels, _ = K.layout([CARD_W[k] for k in K.KINDS])
    centers = [(a + b) / 2 for a, b in channels]
    for i, laag in enumerate(K.decor(dier, centers) + T.lagen):
        fill(laag.vorm, laag.filament, 1 + i * 0.1 + laag.z1 * 0.01, ec="#00000033")
    for (a, b) in channels:
        y0 = WINDOW_Y0 + RIM_H
        ax.fill([a, b, b, a], [y0, y0, y0 + CARD_H, y0 + CARD_H], color="white", ec="#777", lw=0.5, zorder=3)
    ax.set_title(f"{dier.nr:02d} {dier.naam['nl']} / {dier.naam['en']}  "
                 f"({', '.join(FILAMENTEN[c][0] for c in dict.fromkeys([dier.bord, dier.kop] + [l.filament for l in T.lagen]))})",
                 fontsize=8)
    ax.set_xlim(-105, 105)
    ax.set_ylim(-18, 200)
    ax.set_aspect("equal")
    ax.axis("off")


def main(argv):
    out = Path(argv[argv.index("--uit") + 1]) if "--uit" in argv else Path("/tmp/schets.png")
    namen = [a for a in argv if not a.startswith("--") and a != str(out)]
    dieren = [PER_SLEUTEL[n] for n in namen] if namen else ALLE
    n = len(dieren)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.2 * rows), squeeze=False)
    for ax, d in zip(axs.flat, dieren):
        teken(ax, d)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
