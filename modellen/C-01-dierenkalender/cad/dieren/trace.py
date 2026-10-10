"""Een referentieafbeelding overtrekken tot gekleurde lagen (voor de fan-art-varianten).

Werkwijze (alles in mm, de afbeelding wordt eerst herschaald naar PX mm per pixel):
  1. ruis weg (mediaanfilter) en elke pixel naar de dichtstbijzijnde paletkleur (in Lab)
  2. achtergrond = alles wat vanaf de beeldrand bereikbaar is zonder een zwarte lijn te kruisen;
     wat overblijft (grootste stuk, gaten gevuld) is het silhouet
  3. opschonen: zachte meerderheid per kleur (anti-aliasranden weg), zwarte lijnen verdikt tot ze
     printbaar zijn, te smalle stroken en te kleine eilandjes gaan naar de kleur ernaast
  4. per filament de randen volgen (contourpy, subpixel) en naar mm omzetten

De bronafbeeldingen staan in fanart/bronnen/ (zie de LEESMIJ daar); ze horen niet in een publieke repo.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from params import ZWART

from .kit import E, Z_DETAIL, Z_PLAAT, Laag

BRONNEN = Path(__file__).resolve().parents[2] / "fanart" / "bronnen"
PX = 0.06            # mm per pixel tijdens het overtrekken
ACHTER = "achter"


@dataclass(frozen=True)
class Bron:
    bestand: str                     # bestandsnaam in fanart/bronnen/
    palet: tuple                     # ((r, g, b), filament | ACHTER[, muur]), ...: muur = telt als buitenlijn
    hoogte: float                    # mm: hoogte van het silhouet
    onder: float = 91.0              # mm: y van de onderkant van het silhouet
    x: float = 0.0                   # mm: x van het midden van het silhouet
    wis: tuple = ()                  # pixelrechthoeken (x0, y0, x1, y1) in de bron die achtergrond worden
    ruis: float = 0.25               # mm: mediaanfilter tegen ruis en stoftextuur (foto van een kussen)
    verdik: float = 0.0              # mm: zwarte lijnen aan elke kant zoveel dikker
    min_breedte: float = 0.5         # mm: smallere stroken van een vlakkleur verdwijnen
    min_vlak: float = 0.6            # mm²: kleinere eilandjes verdwijnen
    lijn: str = ZWART                # de lijnkleur (grens tussen binnen en buiten)
    glad: float = 0.0                # mm: buitenrand zoveel afronden (b.v. tegen pluche of stof)


def _lab(rgb):
    """sRGB (0..255) naar CIE-Lab, voor kleurafstanden die op het oog lijken."""
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _schijf_dilate(mask, r):
    return ndi.distance_transform_edt(~mask) <= r if r > 0 else mask


def _schijf_open(mask, r):
    if r <= 0:
        return mask
    kern = ndi.distance_transform_edt(mask) > r
    return ndi.distance_transform_edt(~kern) <= r + 0.5


def _vul_dichtstbij(labels, onbekend):
    """Onbekende pixels krijgen het label van de dichtstbijzijnde bekende pixel."""
    if not onbekend.any():
        return labels
    _, (iy, ix) = ndi.distance_transform_edt(onbekend, return_indices=True)
    return labels[iy, ix]


def _classificeer(rgb, palet):
    ref = _lab([p[0] for p in palet])
    lab = _lab(rgb)
    d = ((lab[..., None, :] - ref[None, None]) ** 2).sum(-1)
    return d.argmin(-1)


def _achtergrond(klas, lijn_idx, r):
    """Alles vanaf de rand bereikbaar zonder de (iets verdikte) lijnkleur te kruisen."""
    muur = _schijf_dilate(np.isin(klas, lijn_idx), r)
    vrij = ~muur
    lab, _ = ndi.label(vrij)
    rand = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, rand[rand > 0])
    bg = _schijf_dilate(bg, r)   # de verdikte muur weer terugschuiven naar de echte lijn
    fg = ~bg
    lab, n = ndi.label(fg)
    if n > 1:
        fg = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndi.binary_fill_holes(fg)


def _naar_vormen(mask, naar_mm):
    """Binair masker -> shapely (Multi)Polygon in mm, met gaten, randen subpixel gevolgd."""
    z = ndi.gaussian_filter(np.pad(mask, 2).astype(np.float32), 0.8)
    gen = contourpy.contour_generator(z=z, fill_type=contourpy.FillType.OuterOffset)
    polys, offsets = gen.filled(0.5, 2.0)
    out = []
    for pts, offs in zip(polys, offsets):
        ringen = [naar_mm(pts[a:b] - 2) for a, b in zip(offs[:-1], offs[1:])]
        p = Polygon(ringen[0], ringen[1:]).buffer(0)
        if not p.is_empty:
            out.append(p)
    return unary_union(out).simplify(0.025)


@lru_cache(maxsize=None)
def overtrek(bron):
    """Geeft (silhouet, {filament: vorm}) in mm, in dezelfde assen als de dierentekeningen."""
    pad = BRONNEN / bron.bestand
    if not pad.exists():
        raise FileNotFoundError(f"{pad} ontbreekt: zet de referentieafbeelding daar neer (zie fanart/bronnen/LEESMIJ.md)")
    img = Image.open(pad).convert("RGB")
    for x0, y0, x1, y1 in bron.wis:
        img.paste((255, 255, 255), (x0, y0, x1, y1))
    palet = list(bron.palet)
    namen = [p[1] for p in palet]
    # de buitenlijn: standaard elke paletkleur van de lijnkleur; (rgb, ZWART, False) telt niet mee
    # (b.v. Mario's lichtblauwe iris wordt zwart, maar lijkt te veel op de rand van het kussen)
    lijn_idx = [i for i, p in enumerate(palet) if (p[2] if len(p) > 2 else p[1] == bron.lijn)]

    # 1. schaal bepalen op de bron zelf, dan herschalen naar PX mm per pixel
    klas0 = _classificeer(np.asarray(img, np.float32), palet)
    fg0 = _achtergrond(klas0, lijn_idx, 2)
    ys, xs = np.nonzero(fg0)
    mm_per_px = bron.hoogte / (ys.max() - ys.min() + 1)
    k = mm_per_px / PX
    img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    rgb = np.asarray(img, np.float32)
    if bron.ruis > 0:
        n = max(3, int(round(bron.ruis / PX)) | 1)
        rgb = ndi.median_filter(rgb, size=(n, n, 1))
    klas = _classificeer(rgb, palet)
    fg = _achtergrond(klas, lijn_idx, 0.35 / PX)

    # 2. per filament een label; achtergrondkleur binnen het silhouet en onbekend -> dichtstbij
    filamenten = sorted({n for n in namen if n != ACHTER})
    idx = {f: i for i, f in enumerate(filamenten)}
    labels = np.array([idx.get(n, -1) for n in namen])[klas]
    labels = _vul_dichtstbij(labels, (labels < 0) & fg)

    # 3a. zachte meerderheid: haalt anti-aliasranden (grijs tussen zwart en wit) weg
    zacht = np.stack([ndi.gaussian_filter((labels == i).astype(np.float32), 0.12 / PX) for i in range(len(filamenten))])
    labels = zacht.argmax(0)
    # 3b. lijnen verdikken
    if bron.lijn in idx and bron.verdik > 0:
        labels[_schijf_dilate(labels == idx[bron.lijn], bron.verdik / PX) & fg] = idx[bron.lijn]
    # 3c. te smalle stroken en kleine eilandjes -> kleur ernaast
    onbekend = np.zeros_like(fg)
    for f, i in idx.items():
        m = (labels == i) & fg
        if f != bron.lijn:
            onbekend |= m & ~_schijf_open(m, bron.min_breedte / 2 / PX)
        lab, n = ndi.label(m)
        if n:
            opp = np.bincount(lab.ravel()) * PX * PX
            klein = np.nonzero(opp < bron.min_vlak)[0]
            onbekend |= np.isin(lab, klein[klein > 0])
    labels = _vul_dichtstbij(labels, onbekend & fg)

    # 4. vectoriseren; y omkeren en in mm plaatsen
    ys, xs = np.nonzero(fg)
    cx, onder = (xs.min() + xs.max()) / 2, ys.max()

    def naar_mm(p):
        return np.column_stack([(p[:, 0] - cx) * PX + bron.x, (onder - p[:, 1]) * PX + bron.onder])

    silhouet = _naar_vormen(fg, naar_mm)
    if silhouet.geom_type == "MultiPolygon":
        silhouet = max(silhouet.geoms, key=lambda g: g.area)
    silhouet = Polygon(silhouet.exterior)
    if bron.glad:
        g = bron.glad
        silhouet = silhouet.buffer(-g).buffer(2 * g).buffer(-g)
    vormen = {}
    for f, i in idx.items():
        m = (labels == i) & fg
        if m.any():
            vormen[f] = _naar_vormen(m, naar_mm).intersection(silhouet)
    return silhouet, vormen


def deel(vorm, punt):
    """Het stuk van een (Multi)Polygon dat punt (mm) bevat, b.v. de neus uit alle latte vlakken."""
    for g in getattr(vorm, "geoms", [vorm]):
        if g.contains(Point(punt)):
            return g
    raise ValueError(f"geen vlak op {punt}")


Z_VLAK = 4.4   # vlakken van een overtrokken tekening; de lijnen staan daar 0,6 mm boven (tot Z_DETAIL)


def lagen(bron, basis, hoogtes=None, verhoog=()):
    """Reliëflagen van een overtrokken bron. Geeft (silhouet, lagen).

    De lagen sluiten precies op elkaar aan en overlappen niet (één vorm per kleur en hoogte):
    lijnen eerst, dan de vlakken, en wat er tussen de vlakken nog openligt gaat naar `basis`.
    hoogtes:  {filament: z1} om een vlakkleur hoger of lager te zetten (standaard Z_VLAK)
    verhoog:  ((filament, (x, y), z1), ...): het losse vlak van die kleur op dat punt extra hoog,
              b.v. Mario's neus
    """
    sil, vormen = overtrek(bron)
    h = {f: (Z_DETAIL if f == bron.lijn else Z_VLAK) for f in vormen} | (hoogtes or {})
    rest = sil
    gebied = {}
    if bron.lijn in vormen:
        gebied[bron.lijn] = vormen[bron.lijn].buffer(0.05).intersection(sil)
        rest = rest.difference(gebied[bron.lijn])
    for f, v in vormen.items():
        if f != bron.lijn:
            gebied[f] = v.buffer(0.05).intersection(rest)
            rest = rest.difference(gebied[f])
    gebied[basis] = unary_union([gebied.get(basis, Polygon()), rest])
    out = []
    for f, g in gebied.items():
        hoog = [deel(g, punt) for f2, punt, _ in verhoog if f2 == f]
        g = g.difference(unary_union(hoog)) if hoog else g
        out.append(Laag(f, g, Z_PLAAT - E, h[f]))
        out += [Laag(f, d, Z_PLAAT - E, z1) for d, (f2, _, z1) in zip(hoog, [v for v in verhoog if v[0] == f])]
    return sil, out

