#!/usr/bin/env python3
"""Port of Bambu Studio's automatic flushing-volume calculation.

Sources (bambulab/BambuStudio, master, fetched 2026-10-10; copies in ./flushvol/):
  src/libslic3r/FlushVolCalc.cpp        FlushVolCalculator::calc_flush_vol / calc_flush_vol_rgb
  src/libslic3r/FlushVolPredictor.cpp   table lookup (resources/flush/flush_data_*.txt, DeltaE2000 <= 5)
                                        + measured filament-id pairs (resources/flush/flush_data_material_pair.json)
  src/slic3r/Utils/ColorSpaceConvert.cpp RGB2HSV
  src/slic3r/GUI/Plater.cpp             get_min_flush_volumes(), Sidebar::auto_calc_flushing_volumes_internal()
  src/slic3r/GUI/WipeTowerDialog.cpp    WipingDialog::CalcFlushingVolumes()

Defaults for "Bambu Lab A1 0.4 nozzle" + Bambu PLA (Basic/Matte) @BBL A1:
  nozzle_flush_dataset = 0                          -> resources/flush/flush_data_standard.txt
  max flush           = g_max_flush_volume = 900
  min flush           = get_min_flush_volumes():
                        nozzle_volume(92) - PI*1.75^2/4 * retract_len, retract_len = 18
                        (enable_long_retraction_when_cut=2 [EnableFilament], filament_long_retractions_when_cut=1,
                         filament_retraction_distances_when_cut=18)  -> int(48.70) = 48
  flush_multiplier    = 1.0 (project key, list of floats). NOT applied to the stored matrix; it is
                        applied at slicing time (Print.cpp: volume * flush_multiplier[nozzle]).
  to support filament  = 230 (g_flush_volume_to_support), from support >= 700.

Floating point: the C++ uses 32-bit floats; this port rounds to float32 at the same points,
so results should match exactly except for pathological ties.
"""
import math
import struct

G_MAX_FLUSH_VOLUME = 900
G_FLUSH_VOLUME_TO_SUPPORT = 230
G_MIN_FLUSH_VOLUME_FROM_SUPPORT = 700
A1_MIN_FLUSH = 48  # see get_min_flush_volumes() derivation above
DEFAULT_FLUSH_MULTIPLIER = 1.0

# resources/flush/flush_data_standard.txt (nozzle_flush_dataset 0), verbatim.
FLUSH_DATA_STANDARD = """colors
#000000 #C12E1F #00AE42 #545454 #D1D3D5 #5B6579 #F4EE2A #9D432C #5E43B7 #0A2989 #FF6A13 #8E9089
src dst flush
#000000 #F4EE2A 450
#000000 #5E43B7 330
#C12E1F #F4EE2A 420
#C12E1F #FF6A13 210
#00AE42 #D1D3D5 330
#00AE42 #F4EE2A 240
#00AE42 #FF6A13 270
#545454 #00AE42 180
#545454 #D1D3D5 240
#545454 #F4EE2A 270
#545454 #5E43B7 120
#545454 #FF6A13 300
#545454 #8E9089 120
#D1D3D5 #F4EE2A 120
#D1D3D5 #FF6A13 150
#5B6579 #C12E1F 120
#5B6579 #00AE42 90
#5B6579 #D1D3D5 120
#5B6579 #F4EE2A 180
#5B6579 #9D432C 120
#5B6579 #0A2989 90
#5B6579 #FF6A13 180
#5B6579 #8E9089 90
#F4EE2A #000000 120
#F4EE2A #C12E1F 90
#F4EE2A #00AE42 150
#F4EE2A #9D432C 150
#F4EE2A #FF6A13 90
#9D432C #00AE42 240
#9D432C #D1D3D5 300
#9D432C #F4EE2A 270
#9D432C #FF6A13 180
#9D432C #8E9089 210
#5E43B7 #00AE42 180
#5E43B7 #D1D3D5 270
#5E43B7 #F4EE2A 270
#5E43B7 #9D432C 150
#5E43B7 #FF6A13 270
#5E43B7 #8E9089 210
#0A2989 #C12E1F 330
#0A2989 #00AE42 210
#0A2989 #545454 150
#0A2989 #D1D3D5 450
#0A2989 #5B6579 240
#0A2989 #9D432C 270
#0A2989 #5E43B7 180
#0A2989 #FF6A13 390
#0A2989 #8E9089 270
#FF6A13 #C12E1F 90
#FF6A13 #D1D3D5 210
#FF6A13 #F4EE2A 210
#FF6A13 #9D432C 120
#FF6A13 #8E9089 180
#8E9089 #C12E1F 150
#8E9089 #00AE42 120
#8E9089 #D1D3D5 150
#8E9089 #F4EE2A 270
#8E9089 #FF6A13 150
"""

# resources/flush/flush_data_material_pair.json "pairs" (directional filament_id pairs).
MEASURED_PURGE = {
    ("GFG00", "GFA00"): 650.0,
    ("GFA00", "GFG00"): 650.0,
    ("GFB00", "GFN08"): 400.0,
    ("GFB01", "GFN04"): 350.0,
}


def _f(x):
    """Round a Python float to IEEE float32 (C++ `float`)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def parse_color(c):
    """'#RRGGBB' / '#RRGGBBAA' / (r,g,b) / (r,g,b,a) -> (a, r, g, b) ints."""
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) == 6:
            return 255, int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        if len(s) == 8:
            return int(s[6:8], 16), int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        raise ValueError("bad colour %r" % c)
    c = tuple(int(v) for v in c)
    if len(c) == 3:
        return (255,) + c
    if len(c) == 4:
        return (c[3],) + c[:3]
    raise ValueError("bad colour %r" % (c,))


# ---------------------------------------------------------------- FlushVolPredictor (table lookup)
def _rgb2lab(r, g, b):
    def gamma(x):
        return ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92

    R = gamma(r / 255.0) * 100
    G = gamma(g / 255.0) * 100
    B = gamma(b / 255.0) * 100
    x = 0.412453 * R + 0.357580 * G + 0.180423 * B
    y = 0.212671 * R + 0.715160 * G + 0.072169 * B
    z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    XN, YN, ZN = 95.0489, 100.0, 108.8840
    thr = _f(0.008856)

    def f(t):
        return t ** (1.0 / 3.0) if t > thr else 7.787 * t + 0.137931

    xn, yn, zn = f(x / XN), f(y / YN), f(z / ZN)
    return 116.0 * yn - 16.0, 500.0 * (xn - yn), 200.0 * (yn - zn)


def _delta_e2000(lab1, lab2):
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    p25 = 25.0 ** 7
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    Cm7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1 - math.sqrt(Cm7 / (Cm7 + p25)))
    pa1, pa2 = (1.0 + G) * a1, (1.0 + G) * a2
    pc1, pc2 = math.sqrt(pa1 * pa1 + b1 * b1), math.sqrt(pa2 * pa2 + b2 * b2)

    def hue(pa, pb):
        if pa == 0 and pb == 0:
            return 0.0
        h = math.atan2(pb, pa)
        return h + 2 * math.pi if h < 0 else h

    ph1, ph2 = hue(pa1, b1), hue(pa2, b2)
    dL = l2 - l1
    dC = pc2 - pc1
    pcm = pc1 * pc2
    if pcm == 0:
        dH = 0.0
    else:
        dH = ph2 - ph1
        if dH < -math.pi:
            dH += 2 * math.pi
        elif dH > math.pi:
            dH -= 2 * math.pi
        dH = 2 * math.sqrt(pcm) * math.sin(dH / 2.0)
    Lm = (l1 + l2) / 2.0
    Cm = (pc1 + pc2) / 2.0
    Hs = ph1 + ph2
    if pc1 * pc2 == 0:
        Hm = Hs
    elif abs(ph1 - ph2) <= math.pi:
        Hm = Hs / 2
    else:
        Hm = (Hs + 2 * math.pi) / 2.0 if Hs < 2 * math.pi else (Hs - 2 * math.pi) / 2.0
    d = math.radians
    T = (1 - 0.17 * math.cos(Hm - d(30)) + 0.24 * math.cos(2 * Hm)
         + 0.32 * math.cos(3 * Hm + d(6)) - 0.2 * math.cos(4 * Hm - d(63)))
    dtheta = d(30) * math.exp(-(((Hm - d(275)) / d(25)) ** 2))
    Cm7 = Cm ** 7
    RC = 2 * math.sqrt(Cm7 / (Cm7 + p25))
    Lm50 = (Lm - 50) ** 2
    SL = 1 + (0.015 * Lm50) / math.sqrt(20 + Lm50)
    SC = 1 + 0.045 * Cm
    SH = 1 + 0.015 * Cm * T
    RT = -math.sin(2 * dtheta) * RC
    return _f(math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH)))


class FlushTable:
    def __init__(self, text):
        lines = text.splitlines()
        self.colors = [parse_color(c)[1:] for c in lines[1].split()]
        self.map = {}
        for ln in lines[3:]:
            p = ln.split()
            if len(p) == 3:
                self.map[(parse_color(p[0])[1:], parse_color(p[1])[1:])] = float(p[2])

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(fh.read())

    def _similar(self, rgb):
        lab = _rgb2lab(*rgb)
        for c in self.colors:  # first match in list order, like the C++
            if _delta_e2000(_rgb2lab(*c), lab) <= 5.0:
                return c
        return None

    def predict(self, rgb_from, rgb_to):
        a, b = self._similar(rgb_from), self._similar(rgb_to)
        if a is None or b is None:
            return None
        return self.map.get((a, b))

    def min_flush(self):
        return int(min(self.map.values())) if self.map else None


_TABLES = {0: FlushTable(FLUSH_DATA_STANDARD)}


def register_dataset(dataset, path):
    """Load flush_data_dual_standard.txt (1) / flush_data_dual_highflow.txt (2) for H2D-type printers."""
    _TABLES[dataset] = FlushTable.from_file(path)


# ---------------------------------------------------------------- FlushVolCalc.cpp
def _rgb2hsv(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = _f(cmax - cmin)
    if abs(delta) < 0.001:
        h = 0.0
    elif cmax == r:
        h = _f(60.0 * math.fmod(_f((g - b) / delta), 6.0))
    elif cmax == g:
        h = _f(60.0 * _f(_f((b - r) / delta) + 2))
    else:
        h = _f(60.0 * _f(_f((r - g) / delta) + 4))
    s = 0.0 if abs(cmax) < 0.001 else _f(delta / cmax)
    return h, s, cmax


def _lumi(r, g, b):
    return _f(r * 0.3 + g * 0.59 + b * 0.11)


def _rad(deg):
    return _f(_f(deg / 180.0) * math.pi)


def _delta_hs_bbs(h1, s1, v1, h2, s2, v2):
    h1r, h2r = _rad(h1), _rad(h2)
    dx = _f(_f(math.cos(h1r)) * s1 * v1 - math.cos(h2r) * s2 * v2)
    dy = _f(_f(math.sin(h1r)) * s1 * v1 - math.sin(h2r) * s2 * v2)
    dxy = _f(math.sqrt(_f(dx * dx + dy * dy)))
    return min(_f(1.2), dxy)


def calc_flush_vol_rgb(rgb_from, rgb_to, dataset=0):
    """FlushVolCalculator::calc_flush_vol_rgb -> int (without min_flush offset)."""
    if dataset == 0 and dataset in _TABLES:
        v = _TABLES[0].predict(rgb_from, rgb_to)
        if v is not None:
            return int(v)
    sr, sg, sb = (_f(c / 255.0) for c in rgb_from)
    dr, dg, db = (_f(c / 255.0) for c in rgb_to)
    fh, fs, fv = _rgb2hsv(sr, sg, sb)
    th, ts, tv = _rgb2hsv(dr, dg, db)
    hs_dist = _delta_hs_bbs(fh, fs, fv, th, ts, tv)
    from_l = _lumi(sr, sg, sb)
    to_l = _lumi(dr, dg, db)
    if to_l >= from_l:
        lumi_flush = _f(_f(_f(to_l - from_l) ** _f(0.7)) * 560.0)
    else:
        lumi_flush = _f(_f(from_l - to_l) * 80.0)
        inter_v = _f(0.67 * tv + 0.33 * fv)
        hs_dist = min(inter_v, hs_dist)
    hs_flush = _f(230.0 * hs_dist)
    # calc_triangle_3rd_edge(hs_flush, lumi_flush, 120deg)
    vol = _f(math.sqrt(_f(hs_flush * hs_flush + lumi_flush * lumi_flush
                          - 2 * hs_flush * lumi_flush * _f(math.cos(_rad(120.0))))))
    vol = max(vol, 60.0)
    return int(vol)


def calc_flush_volume(rgb_from, rgb_to, min_flush=A1_MIN_FLUSH, max_flush=G_MAX_FLUSH_VOLUME,
                      dataset=0, from_filament_id="", to_filament_id="",
                      from_is_support=False, to_is_support=False):
    """Bambu Studio auto flushing volume (mm^3) for one colour change, as stored in flush_volumes_matrix.

    rgb_from/rgb_to: '#RRGGBB', '#RRGGBBAA' (alpha 0 = transparent -> treated as white) or tuples.
    Defaults reproduce A1 0.4 nozzle + Bambu PLA: min 48, max 900, dataset 0.
    The flush_multiplier (default 1.0) is NOT applied here (Bambu applies it when slicing).
    Pass filament ids (e.g. 'GFA00') to honour resources/flush/flush_data_material_pair.json.
    """
    if to_is_support:
        return G_FLUSH_VOLUME_TO_SUPPORT
    sa, sr, sg, sb = parse_color(rgb_from)
    da, dr, dg, db = parse_color(rgb_to)
    if sa == 0:
        sr = sg = sb = 255
    if da == 0:
        dr = dg = db = 255
    vol = None
    if dataset != 0 and dataset in _TABLES:
        vol = _TABLES[dataset].predict((sr, sg, sb), (dr, dg, db))
    if vol is None:
        vol = float(calc_flush_vol_rgb((sr, sg, sb), (dr, dg, db), dataset))
        # C++ compares 0..255 luminance against 180/255 / 75/255 here (only matters for dataset != 0)
        is_from_dark = _lumi(sr, sg, sb) > 180.0 / 255.0
        is_to_light = _lumi(dr, dg, db) < 75.0 / 255.0
        if dataset != 0 and is_from_dark and is_to_light:
            vol *= 1.3
        vol += min_flush
    predicted = min(int(vol), max_flush)
    measured = MEASURED_PURGE.get((from_filament_id, to_filament_id))
    if measured is not None:
        predicted = min(max(predicted, int(measured)), max_flush)
    if from_is_support:
        predicted = max(predicted, G_MIN_FLUSH_VOLUME_FROM_SUPPORT)
    return predicted


def min_flush_volume(nozzle_volume=92.0, enable_long_retraction_when_cut=2, long_retractions_when_cut=0,
                     retraction_distances_when_cut=18.0, filament_long_retractions_when_cut=1,
                     filament_retraction_distances_when_cut=18.0):
    """Plater.cpp get_min_flush_volumes() for one filament (defaults = A1 0.4 + Bambu PLA)."""
    extra = int(nozzle_volume)
    retract = retraction_distances_when_cut if (enable_long_retraction_when_cut and long_retractions_when_cut == 1) else 0
    retract = int(retract)
    if filament_long_retractions_when_cut == 0:
        retract = 0
    elif filament_long_retractions_when_cut == 1 and enable_long_retraction_when_cut == 2:
        if filament_retraction_distances_when_cut is not None and not math.isnan(filament_retraction_distances_when_cut):
            retract = int(filament_retraction_distances_when_cut)
        else:
            retract = int(retraction_distances_when_cut)
    return int(extra - math.pi * 1.75 * 1.75 / 4 * retract)


def flush_volumes_matrix(colours, filament_ids=None, min_flush=None, max_flush=G_MAX_FLUSH_VOLUME, dataset=0):
    """Full n*n row-major matrix (row = from, col = to, diagonal 0) as Bambu stores in
    project_settings 'flush_volumes_matrix' for a single-nozzle printer (A1)."""
    n = len(colours)
    ids = filament_ids or [""] * n
    mins = min_flush if isinstance(min_flush, (list, tuple)) else [A1_MIN_FLUSH if min_flush is None else min_flush] * n
    out = []
    for i in range(n):
        for j in range(n):
            out.append(0 if i == j else calc_flush_volume(colours[i], colours[j], mins[i], max_flush, dataset, ids[i], ids[j]))
    return out


if __name__ == "__main__":
    print("min_flush (A1 0.4 + PLA) =", min_flush_volume())
    tests = [("#FFFFFF", "#000000"), ("#000000", "#FFFFFF"), ("#FF9016", "#FFFFFF"), ("#FFFFFF", "#FF9016")]
    for a, b in tests:
        raw = calc_flush_vol_rgb(parse_color(a)[1:], parse_color(b)[1:])
        print(f"{a} -> {b}: {calc_flush_volume(a, b)}  (colour part {raw} + min 48, max 900)")
    print("matrix [white, black, orange]:", flush_volumes_matrix(["#FFFFFF", "#000000", "#FF9016"]))
    print("table hit #000000 -> #F4EE2A:", calc_flush_volume("#000000", "#F4EE2A"))
    print("measured pair GFA00->GFG00:", calc_flush_volume("#FFFFFF", "#FFFFFE", from_filament_id="GFA00", to_filament_id="GFG00"))
