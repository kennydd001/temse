"""Montage- en printcontrole van E-01 Splitsmijn: het eindmodel en de pasproef.

Gebruik:  python controle.py [--snel]
Schrijft  controle/RAPPORT.md, controle/controle.json, controle/valbord.png en controle/mijnberg.png

  1. Muntdoorgang: past een muntje overal tussen de pinnen door, en is er nergens een plek waar het tussen
     twee pinnen (of een pin en de wand) kan blijven liggen? Eindmodel: van de spleet door de trechters in het
     dek tot in de karretjes.
  2. Valsimulatie (simulatie.py): links of rechts, over een reeks materiaalwaarden en mikpunten; hoe vaak elke
     splitsing van 8 voorkomt; blijft er ooit een muntje hangen. Voor het eindmodel (met de houweel en de
     mijnberg) en voor de pasproef.
  3. Slinger: massa uit de geometrie, slingertijd, hoe lang hij doorslingert, zwaartepunt boven het mes.
  4. Spelingen en botsingen over de hele uitslag van de slinger.
  5. Verbindingen: speling per kant van elke klem- of schuifverbinding.
  6. Mijnberg en karretjes: trechters, inhoud, stabiliteit van het geheel.
  7. Printbaarheid per plaat (gereedschap/printcheck.py), gewicht. Met de slicercijfers uit controle/slicer.json en
     controle/slicer_eind.json als die er zijn (python slice.py, na bouw.py).
"""
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

import _pad  # noqa: F401,E402
import params as P  # noqa: E402
import platen  # noqa: E402
import simulatie as S  # noqa: E402
import bouw  # noqa: E402
import mijn as M  # noqa: E402
import valbord as V  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely  # noqa: E402
from gereedschap.printcheck import gram  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "controle"
KLEUR = {P.ZWART: "#3A3A3A", P.BRUIN: "#C9A98F", P.ROOD: "#9D2235", P.BLAUW: "#0078BF", P.TERRA: "#B15533",
         P.GEEL: "#FEC600"}
SAMEN = {frozenset(("portaal", "letters"))}     # delen van hetzelfde object: raken elkaar bewust
R = P.MUNT_D / 2


# --- 1. muntdoorgang en ligplekken --------------------------------------------------------------

def obstakels():
    """[(soort, data)]: cirkels (x, z, r) en verticale wanden (x, kant) in het valbord."""
    out = [("cirkel", (x, z, P.PIN_D / 2)) for x, z in P.pinnen()]
    out += [("cirkel", (x, z, P.BULT_R)) for x, z in P.bulten()]
    out += [("wand", (-P.BINNEN_HALF, 1)), ("wand", (P.BINNEN_HALF, -1))]
    h = P.VERDELER_T / 2
    out += [("cirkel", (0.0, P.VERDELER_TOP - 0.4, 0.4)), ("wand", (h, 1)), ("wand", (-h, -1))]
    return out


def _verdeler(ob):
    """De verdeler staat recht onder de middelste pin: daartussen gaat geen muntje door, het valt ernaast."""
    return abs(ob[1][0]) < P.VERDELER_T and (ob[0] == "wand" or ob[1][1] < P.VERDELER_TOP)


def _raakt(c, ob):
    """Afstand van muntmidden c tot obstakel min R (0 = raakt), en de normaal van het obstakel naar het muntje."""
    soort, d = ob
    if soort == "cirkel":
        x, z, r = d
        v = np.array(c) - (x, z)
        n = float(np.linalg.norm(v))
        return n - R - r, v / n
    x, kant = d
    return kant * (c[0] - x) - R, np.array([kant, 0.0])


def ligplekken(mu=0.3):
    """Plekken waar een muntje op twee steunpunten kan blijven liggen (wrijving tot mu meegerekend)."""
    obs = obstakels()
    plekken, doorgang = [], []
    for a, b in itertools.combinations(obs, 2):
        kand = []
        if a[0] == "cirkel" and b[0] == "cirkel":
            (x1, z1, r1), (x2, z2, r2) = a[1], b[1]
            d = math.hypot(x2 - x1, z2 - z1)
            rest = d - r1 - r2
            if rest < 3 * R and not (_verdeler(a) or _verdeler(b)):
                doorgang.append(rest)
            ra, rb = R + r1, R + r2
            if d > ra + rb or d < abs(ra - rb) or d == 0:
                continue
            t = (ra * ra - rb * rb + d * d) / (2 * d)
            h = math.sqrt(max(ra * ra - t * t, 0.0))
            ux, uz = (x2 - x1) / d, (z2 - z1) / d
            for s in (-1, 1):
                kand.append((x1 + t * ux - s * h * uz, z1 + t * uz + s * h * ux))
        elif a[0] == "cirkel" or b[0] == "cirkel":
            (x1, z1, r1) = (a if a[0] == "cirkel" else b)[1]
            xw, kant = (b if a[0] == "cirkel" else a)[1]
            cx = xw + kant * R
            dx = cx - x1
            if abs(dx) > R + r1:
                if abs(x1 - xw) < 3 * R and kant * (x1 - xw) > 0 and not (_verdeler(a) or _verdeler(b)):
                    doorgang.append(abs(x1 - xw) - r1)
                continue
            dz = math.sqrt((R + r1) ** 2 - dx * dx)
            kand += [(cx, z1 + dz), (cx, z1 - dz)]
        for c in kand:
            if c[1] < 0 or c[1] > P.PLAFOND_Z or abs(c[0]) > P.BINNEN_HALF:
                continue
            if any(_raakt(c, o)[0] < -0.05 for o in obs):
                continue
            n1, n2 = _raakt(c, a)[1], _raakt(c, b)[1]
            # draagt het tweetal het muntje? omhoog (0, 1) moet tussen de normalen liggen, met de wrijvingshoek erbij
            hoek = lambda v: math.degrees(math.atan2(v[1], v[0]))  # noqa: E731
            h1, h2 = sorted((hoek(n1), hoek(n2)))
            if h2 - h1 > 180:
                h1, h2 = h2, h1 + 360
            marge = min(90 - h1, h2 - 90) if h1 <= 90 <= h2 else -min(abs(90 - h1), abs(h2 - 90))
            if marge > -math.degrees(math.atan(mu)):
                plekken.append({"plek": [round(c[0], 1), round(c[1], 1)], "marge_graden": round(marge, 1)})
    return {"plekken": plekken, "kleinste_doorgang": round(min(doorgang), 1) if doorgang else None,
            "boven_plank": P.PLAFOND_Z - (P.Z_PLANK + P.PLANK_H / 2) - P.MUNT_D}


# --- de twee modellen ------------------------------------------------------------------------------

def modellen():
    """De twee modellen die gecontroleerd worden, met wat de controles van elk nodig hebben."""
    return {
        "eindmodel": {"versie": P.VERSIE_EIND, "stand": M.stand, "delen": M.delen, "massa": M.slingermassa(),
                      "dy": M.DY, "eind": True},
        "pasproef": {"versie": P.VERSIE, "stand": V.stand, "delen": V.delen, "massa": S.slinger_massa(), "dy": 0.0,
                     "eind": False},
    }


# --- 2. valsimulatie --------------------------------------------------------------------------------

def simulatie(md, rondes, rondes_splits):
    kw = {"eind": md["eind"], "massa": md["massa"]}
    tabel, vast, uitslagen = S.mikproef(rondes=rondes, **kw)
    telling = S.splitsingen(rondes=rondes_splits, **kw)
    return {"tabel": [(m, p, [round(x, 3) for x in rij]) for m, p, rij in tabel], "vast": vast,
            "totaal": len(tabel) * 3 * rondes * 8, "muntjes_per_cel": rondes * 8,
            "kleinste_uitslag": min(uitslagen), "mediaan_uitslag": statistics.median(uitslagen),
            "per_mik": [statistics.mean(rij[i] for *_, rij in tabel) for i in range(3)],
            "mikeffect_per_stuit": {str(st): round(statistics.mean(rij[0] - rij[2] for m, _, rij in tabel
                                                                   if m["stuit"] == st), 3)
                                    for st in sorted({m["stuit"] for m, _, _ in tabel})},
            "splitsingen": telling}


# --- 3. slinger ---------------------------------------------------------------------------------------

def _zwaartepunt(m):
    """(volume, zwaartepunt (x, y, z)) van een mesh, met tetraëders vanuit de oorsprong."""
    tm = m.to_mesh()
    v = np.asarray(tm.vert_properties[:, :3], dtype=float)
    f = np.asarray(tm.tri_verts)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    sv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    return float(sv.sum()), ((a + b + c) / 4 * sv[:, None]).sum(axis=0) / sv.sum()


def _zwaartepunt_y(ms):
    eig = [_zwaartepunt(m) for m in ms]
    return sum(v * c[1] for v, c in eig) / sum(v for v, _ in eig)


def slinger(md):
    m, d, _ = md["massa"]
    draait = [x for _, x, beweegt in md["delen"]().values() if beweegt]
    mes = (P.BEUGEL_Y + md["dy"], P.MES[1] + md["dy"])
    flank_mes = P.MES_HOEK / 2
    flank_groef = 90 - (180 - P.GROEF_HOEK) / 2
    return {"slingertijd_s": round(S.slingertijd(md["massa"]), 2), "massa_g": round(m, 0), "zwaartepunt_onder_as": round(d, 0),
            "zwaartepunt_y": round(_zwaartepunt_y(draait), 1), "mes_draagt_y": [round(mes[0], 1), round(mes[1], 1)],
            "draaien_voor_de_flanken_raken": round(flank_groef - flank_mes, 0),
            "uitdemping": {str(p): S.uitdemping(p, massa=md["massa"]) for p in S.PEN_WRIJVING}}


# --- 4. spelingen ---------------------------------------------------------------------------------------

def spelingen(md):
    """Kleinste speling tussen elk bewegend en elk vast deel over de hele uitslag; botsingen tussen alle delen
    (behalve delen van hetzelfde object, die bewust op elkaar liggen)."""
    delen = md["delen"]()
    beweegt = [n for n, (_, _, b) in delen.items() if b]
    vast = [n for n, (_, _, b) in delen.items() if not b]
    hoeken = np.linspace(-P.MAX_HOEK, P.MAX_HOEK, 15)
    best, botsing = {}, []
    for h in hoeken:
        st = md["stand"](float(h))
        for a in beweegt:
            for b in vast:
                g = float(st[a][1].min_gap(st[b][1], 6.0))
                if g < best.get((a, b), (99.0, 0.0))[0]:
                    best[(a, b)] = (g, float(h))
        for a, b in itertools.combinations(st, 2):
            if frozenset((a, b)) in SAMEN:
                continue
            v = (st[a][1] ^ st[b][1]).volume()
            if v > 0.01:
                botsing.append((round(float(h)), a, b, round(v, 2)))
    st = md["stand"](0.0)
    paren = [{"paar": f"{a} - {b}", "speling": round(g, 2), "hoek": round(h, 0)}
             for (a, b), (g, h) in sorted(best.items(), key=lambda kv: kv[1][0]) if g < 6 and (a, b) != ("mesblok", "beugel")]
    return {"paren": paren, "botsingen": botsing, "mes_op_groef": round(float(st["mesblok"][1].min_gap(st["beugel"][1], 2.0)), 2)}


# --- 5. verbindingen ------------------------------------------------------------------------------------

def _doos(x0, y0, z0, x1, y1, z1):
    return V.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def passingen(md):
    """Speling per kant in elke verbinding, gemeten tussen de meshes in de montagestand (alleen rond die
    verbinding, en zonder de vlakken waar de delen bewust op elkaar rusten)."""
    st = {k: m for k, (_, m) in md["stand"](0.0).items()}
    zp, W, dy = P.Z_PLANK, P.BINNEN_HALF, md["dy"]
    arm = "steel" if md["eind"] else "slinger"
    arm_y = (P.ARM_Y[0] + dy, P.ARM_Y[1] + dy)
    mes = (P.MES[0] + dy, P.MES[1] + dy)

    def speling(a, b, d):
        return round(float((st[a] ^ d).min_gap(st[b] ^ d, 3.0)), 3)

    x, z = V.nokken()[0]
    px, pz, _ = P.BEUGEL_PEN
    mx, mz, _ = P.MES_PEN
    uit = [
        {"wat": f"nokje Ø {P.NOK_D:.1f} in de voorplaat ({len(V.nokken())} stuks tegelijk)", "param": "NOK_GAT",
         "speling": speling("achterplaat", "voorplaat", _doos(x - 3, P.Y_VOOR + 0.05, z - 3, x + 3, P.Y_VOOR + 3, z + 3)),
         "soort": "licht klemmen"},
        {"wat": f"pennetjes Ø 3 van de {'steel' if md['eind'] else 'arm'} in het mesblok", "param": "MES_GAT",
         "speling": speling(arm, "mesblok", _doos(mx - 3, mes[0] + 0.01, zp + mz - 3, mx + 3, mes[1], zp + mz + 3)),
         "soort": "klemmen"},
        {"wat": "pennen Ø 4 van de voorplaat in de beugel (liggend druppelgat)", "param": "BEUGEL_GAT",
         "speling": speling("voorplaat", "beugel", _doos(px - 4, P.BORD_T + 0.5 + dy, zp - pz - 4, px + 4, 30, zp - pz + 4)),
         "soort": "klemmen"},
        {"wat": f"zeskant van de plank in de {'steel' if md['eind'] else 'arm'}", "param": "ZESKANT_SPEL",
         "speling": speling("plank", arm, _doos(-7, arm_y[0] + 0.2, zp - 7, 7, arm_y[1], zp + 7)),
         "soort": "schuiven zonder wiebelen"},
        {"wat": "naaf van de plank in het gat van de voorplaat", "param": "NAAF_R",
         "speling": speling("plank", "voorplaat", _doos(-8, P.Y_VOOR, zp - 8, 8, P.BORD_T, zp + 8)), "soort": "vrij"},
    ]
    if not md["eind"]:
        uit += [
            {"wat": "zijranden in de kokers van de voet", "param": "SPEL",
             "speling": min(speling(a, "voet", _doos(W - 1, -5, 0.5, W + P.RAND + 5, 20, 15))
                            for a in ("achterplaat", "voorplaat")),
             "soort": "schuiven"},
            {"wat": "vin van de verdeler in de spleet", "param": "",
             "speling": min(speling("voet", a, _doos(-3, 0, 0.5, 3, 14, P.VERDELER_TOP)) for a in ("achterplaat", "voorplaat")),
             "soort": "vrij"}]
        return uit
    zh = M.ZH
    uit += [
        {"wat": "pennetjes Ø 3 van de steel in de houweelkop", "param": "MES_GAT",
         "speling": speling("steel", "kop", _doos(-3, arm_y[1] + 0.05, zh - 9, 3, arm_y[1] + 4, zh + 5)), "soort": "klemmen"},
        {"wat": "steel in de gleuf van de houweelkop", "param": "",
         "speling": speling("steel", "kop", _doos(-6, arm_y[0] + 0.5, zh - 6, 6, arm_y[1] - 0.5, zh + 10)), "soort": "schuiven"},
        {"wat": "zijranden en mijnhout in de kokervoeten", "param": "SPEL",
         "speling": min(speling(a, "kokervoet L", _doos(W - 1, -5, 0.5, W + P.RAND + 5, 20, P.KOKER_H - 0.5))
                        for a in ("achterplaat", "voorplaat", "mijnhout")), "soort": "schuiven"},
        {"wat": "prop van de kokervoet in het dek", "param": "KOKER_SPEL",
         "speling": speling("kokervoet L", "mijnberg", _doos(W - 2, -3, -P.KOKER_D + 1, W + P.RAND + 2, 16, -0.5)),
         "soort": "licht klemmen"},
        {"wat": "nokjes van het mijnhout in de voorplaat", "param": "",
         "speling": speling("mijnhout", "voorplaat", _doos(x - 3, P.Y_VOOR + 1.0, z - 3, x + 3, P.BORD_T - 0.02, z + 3)),
         "soort": "vrij (houden het raamwerk op zijn plaats)"},
        {"wat": "tong van de verdelervin in de wig", "param": "VIN_SLEUF",
         "speling": speling("verdelervin", "mijnberg",
                            _doos(-3, P.ACHTER_T + 0.7, -P.VIN_SLEUF[1] + 0.5, 3, P.Y_VOOR - 0.7, -0.2)),
         "soort": "klemmen"},
        {"wat": "verdelervin in de spleet", "param": "",
         "speling": min(speling("verdelervin", a, _doos(-3, 0, 0.5, 3, 14, P.VERDELER_TOP))
                        for a in ("achterplaat", "voorplaat")),
         "soort": "vrij"},
        {"wat": "bovenrand van het bord en het mijnhout in de kokerkappen", "param": "SPEL",
         "speling": min(speling(a, "kokerkap L", _doos(W - 1, -6, P.BORD_H - P.KAP_ONDER + 0.5, W + P.RAND + 5, 20,
                                                       P.BORD_H - 0.5))
                        for a in ("achterplaat", "voorplaat", "mijnhout")), "soort": "schuiven"},
        {"wat": "tong van het portaal in de kokerkappen", "param": "SPEL",
         "speling": speling("portaal", "kokerkap L", _doos(W - 1, -8, P.BORD_H + P.KAP_LAT + 0.5, W + P.RAND + 5, 14,
                                                           P.PORTAAL_Z[0] - 0.5)),
         "soort": "schuiven"},
        {"wat": "karretjes in de tunnels", "param": "KAR_SPEL",
         "speling": min(speling(k, "mijnberg", _doos(-120, P.BERG_Y[0] + 3, -P.BERG_H + 0.5, 120, P.BERG_Y[1] - 3, -P.DEK - 0.5))
                        for k in ("karretje L", "karretje R")),
         "soort": "schuiven"},
        {"wat": "bovenrand van de karretjes onder het dek", "param": "KAR_RAND_ONDER_DEK",
         "speling": min(speling(k, "mijnberg", _doos(min(s * 20, s * 80), M.kar_maat(1)[1] + 0.3, -P.DEK - 3,
                                                     max(s * 20, s * 80), P.BERG_Y[1] - 0.5, -P.DEK + 0.5))
                        for k, s in (("karretje L", 1), ("karretje R", -1))),
         "soort": "vrij (kleiner dan een muntje)"},
    ]
    return uit


# --- 6. mijnberg, karretjes, stabiliteit ----------------------------------------------------------------

def mijnberg():
    """Trechters, inhoud en stabiliteit van het eindmodel."""
    kar_binnen = P.TUNNEL_X[1] - P.KAR_SPEL - P.KAR_WAND - 1.0
    buiten = math.degrees(math.atan2(P.DEK, P.TRECHTER_BOVEN - kar_binnen))
    wig = math.degrees(math.atan2(P.DEK, P.WIG[1] - P.WIG[0]))
    x0, y0, z0, x1, y1, z1 = M.kar_maat(1)
    w, c = P.KAR_WAND, math.cos(math.radians(P.KAR_SCHUIN))
    inset = (z1 - z0) * math.tan(math.radians(P.KAR_SCHUIN))
    bx = (x0 + w / c, x1 - w / c)                                   # binnenkant aan de bovenrand
    kar = M.karretje(1)
    holte = (_doos(x0, y0, z0, x1, y1, z1) - kar).volume()
    bodem_b = (x1 - x0) - 2 * inset - 2 * w / c
    bodem_d = (y1 - P.KAR_FRONT) - (y0 + w)
    munt_v = math.pi * (P.MUNT_D / 2) ** 2 * P.MUNT_T
    trechter_onder = (P.WIG[1], kar_binnen)
    # stabiliteit: gewicht per onderdeel (zoals geprint), zwaartepunt van het geheel tegen de voetafdruk
    st = M.stand(0.0)
    massa, mom = 0.0, np.zeros(3)
    for naam, (_, m) in st.items():
        ins = bouw.inst(naam)
        infill = 1.0 if ins.get("sparse_infill_density") == "100%" else 0.15
        g = gram(m.translate([0, 0, P.BERG_H]), wand=0.42 * int(ins.get("wall_loops", "2")), infill=infill)
        massa += g
        mom += g * _zwaartepunt(m)[1]
    massa += P.AANTAL_MUNTEN * munt_v / 1000 * 1.24 * 0.9
    zw = mom / massa
    y0b, y1b = P.BERG_Y
    kop_z = M.ZH + P.BERG_H                    # hoogte van de houweelkop boven de tafel: hier duwt het kind
    gewicht_n = massa / 1000 * 9.81
    return {"trechter_buiten_graden": round(buiten, 0), "trechter_wig_graden": round(wig, 0),
            "trechter_onderaan_x": [round(v, 1) for v in trechter_onder], "karretje_binnen_x": [round(bx[0], 1), round(bx[1], 1)],
            "sleuf_y": list(P.SLEUF_Y), "spleet_y": [P.ACHTER_T, P.Y_VOOR],
            "karretje_inhoud_cm3": round(holte / 1000, 0), "tien_munten_cm3": round(10 * munt_v / 1000, 0),
            "munten_plat_op_de_bodem": int(bodem_b // P.MUNT_D) * int(bodem_d // P.MUNT_D),
            "massa_g": round(massa, 0), "zwaartepunt_y": round(float(zw[1]), 1),
            "zwaartepunt_z_boven_tafel": round(float(zw[2]) + P.BERG_H, 1),
            "voet_y": [y0b, y1b], "voet_x": P.BERG_X,
            "kantelkracht_achteruit_N": round(gewicht_n * (zw[1] - y0b) / kop_z, 1),
            "kantelkracht_vooruit_N": round(gewicht_n * (y1b - zw[1]) / kop_z, 1),
            "kantelkracht_opzij_N": round(gewicht_n * P.BERG_X / kop_z, 1),
            # het bord staat met SPEL aan elke kant in de kokers: zo ver kan de bovenkant naar voren of achteren wijken
            "kantelspel_graden": round(math.degrees(math.atan2(2 * P.SPEL, P.KOKER_H)), 1),
            "kantelspel_boven_mm": round(P.PORTAAL_Z[1] * 2 * P.SPEL / P.KOKER_H, 1)}


# --- 7. printbaarheid ---------------------------------------------------------------------------------

def wissels(per_kleur, laag=0.2):
    """Aantal filamentwissels zoals de printer ze doet: laag per laag, en elke laag begint met de kleur waarmee de
    vorige eindigde (als die in de laag zit)."""
    top = max(m.bounding_box()[5] for m in per_kleur.values())
    huidig, n = None, 0
    for z in np.arange(laag / 2, top, laag):
        kleuren = [k for k, m in per_kleur.items() if not m.slice(float(z)).is_empty()]
        if not kleuren:
            continue
        if huidig in kleuren:
            volgorde = [huidig] + [k for k in kleuren if k != huidig]
        else:
            volgorde = kleuren
            n += 0 if huidig is None else 1
        n += len(volgorde) - 1
        huidig = volgorde[-1]
    return n


def printbaar_eind():
    out = {}
    for naam, objecten in platen.platen_eind():
        per_kleur, g = {}, 0.0
        for n, delen in objecten:
            ins = bouw.inst(n)
            infill = 1.0 if ins.get("sparse_infill_density") == "100%" else 0.15
            heel = platen.manifold3d_unie([m for *_, m in delen])
            g += gram(heel, wand=0.42 * int(ins.get("wall_loops", "2")), infill=infill)
            for _, kleur, m in delen:
                per_kleur.setdefault(kleur, []).append(m)
        kleuren = {k: platen.manifold3d_unie(v) for k, v in per_kleur.items()}
        i = printcheck.plaat(kleuren)
        i["kleuren_codes"] = list(per_kleur)
        i["wissels"] = wissels(kleuren) if len(kleuren) > 1 else 0
        i["gram_geschat"] = round(g, 0)
        out[naam] = i
    return out


def printbaar_proef():
    out = {}
    for naam, kleur, delen in platen.platen():
        i = printcheck.plaat({kleur: V.Manifold.batch_boolean([m for _, m in delen], V.OpType.Add)})
        i["kleuren_codes"] = [kleur]
        out[naam] = i
    return out


# --- tekening -----------------------------------------------------------------------------------

def _vul(ax, g, kleur, ec="#222"):
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or not hasattr(p, "exterior"):
            continue
        p = orient(p, 1.0)
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            cs = list(ring.coords)
            verts += cs
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(cs) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=kleur, ec=ec, lw=0.3))


def snede_y(m, y):
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)
    return cs_to_shapely(m.transform(T).slice(-y))


def snede_x(m, x):
    T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)
    return cs_to_shapely(m.transform(T).slice(x))


def banen(md, n=10, seed=4):
    import random
    rnd = random.Random(seed)
    out = []
    sp, sl = S.ruimte(0.5, 0.3, 0.04, eind=md["eind"], massa=md["massa"])
    S.zet_uitslag(sl, 30)
    for _ in range(n):
        for _ in range(int(rnd.uniform(0.6, 1.5) / S.DT)):
            sp.step(S.DT)
        b, c = S.nieuwe_munt(sp, rnd, 0.0, 0.5, 0.3)
        xs, zs, t = [], [], 0.0
        while b.position.y > (-15 if md["eind"] else -30) and t < 5:
            sp.step(S.DT)
            t += S.DT
            xs.append(b.position.x)
            zs.append(b.position.y)
        sp.remove(b, c)
        out.append((xs, zs))
    return out


def tekening(pad, md, banen_, telling):
    fig = plt.figure(figsize=(19, 9.5))
    ax = fig.add_subplot(1, 3, 1)
    st = md["stand"](0.0)
    y_mid = (P.ACHTER_T + P.Y_VOOR) / 2
    namen = (("achterplaat", "mijnberg", "verdelervin", "plank", "karretje L", "karretje R") if md["eind"]
             else ("achterplaat", "voet", "plank"))
    for naam in namen:
        _vul(ax, snede_y(st[naam][1], y_mid), KLEUR[st[naam][0]])
    arm = "steel" if md["eind"] else "slinger"
    for h, kl in ((-P.MAX_HOEK, "#E8D9A0"), (P.MAX_HOEK, "#E8D9A0"), (0.0, None)):
        s_ = md["stand"](h)
        y_arm = P.ARM_Y[0] + md["dy"] + 2.5
        for naam in ((arm, "kop") if md["eind"] else (arm,)):
            _vul(ax, snede_y(s_[naam][1], y_arm), kl or KLEUR[s_[naam][0]], ec="none" if h else "#222")
    for xs, zs in banen_:
        ax.plot(xs, zs, lw=0.7, alpha=0.8)
    ax.set_title("vooraanzicht zoals het kind het ziet: pinnen, kantelplank, houweel (±35°), mijnberg met karretjes, "
                 "gesimuleerde banen", fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_xaxis()          # het kind kijkt van +y: +x is links
    ax.set_ylim(-P.BERG_H - 5 if md["eind"] else -35, (P.PORTAAL_Z[1] if md["eind"] else P.BORD_H) + 8)
    ax2 = fig.add_subplot(1, 3, 2)
    for naam, (kl, m) in st.items():
        _vul(ax2, snede_x(m, 40.0 if md["eind"] else 0.0), KLEUR.get(kl, "#888"))
    ax2.set_title("doorsnede op x = 40 mm: bord, mijnhout, houweel, mijnberg met trechter en karretje" if md["eind"]
                  else "doorsnede midden: spleet, voorplaat, beugel met mes, slinger", fontsize=9)
    ax2.set_aspect("equal")
    ax2.autoscale_view()
    ax3 = fig.add_subplot(1, 3, 3)
    n = len(telling) - 1
    ks = list(range(n + 1))
    tot = sum(telling)
    ax3.bar(ks, [t / tot * 100 for t in telling], color="#0078BF", label="simulatie")
    ax3.plot(ks, [b * 100 for b in S.binomiaal(n)], "k.--", label="eerlijk muntstuk (binomiaal)")
    ax3.set_xticks(ks)
    ax3.set_xticklabels([f"{k}+{n - k}" for k in ks])
    ax3.set_ylabel("% van de rondes")
    ax3.set_title(f"splitsingen van {n} ({tot} rondes): links + rechts", fontsize=9)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pad, dpi=80)
    plt.close(fig)


def tekening_berg(pad):
    """Mijnberg van voren (doorsnede door de spleet) en van opzij (door een karretje), met een karretje half uit."""
    st = M.stand(0.0, (45.0, 0.0))
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [2.4, 1]})
    for naam in ("achterplaat", "mijnberg", "kokervoet L", "kokervoet R", "verdelervin", "karretje L", "karretje R"):
        _vul(axs[0], snede_y(st[naam][1], (P.SLEUF_Y[0] + P.SLEUF_Y[1]) / 2), KLEUR[st[naam][0]])
    axs[0].set_title("doorsnede door de spleet: trechters in het dek, kokervoeten, middenwig met vin, karretjes "
                     "(van voren gezien)", fontsize=9)
    axs[0].set_aspect("equal")
    axs[0].autoscale_view()
    axs[0].invert_xaxis()
    axs[0].set_ylim(-P.BERG_H - 3, 45)
    for naam, (kl, m) in st.items():
        _vul(axs[1], snede_x(m, 50.0), KLEUR.get(kl, "#888"))
    axs[1].set_title("doorsnede op x = 50: het linker karretje 45 mm uitgetrokken", fontsize=9)
    axs[1].set_aspect("equal")
    axs[1].autoscale_view()
    axs[1].set_xlim(-35, 125)
    axs[1].set_ylim(-P.BERG_H - 3, 60)
    fig.tight_layout()
    fig.savefig(pad, dpi=80)
    plt.close(fig)


# --- rapport ------------------------------------------------------------------------------------

def ok(v):
    return "✅" if v else "⚠️"


def f(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


def rapport(r):
    e, pp = r["eindmodel"], r["pasproef"]
    L = ["# Controle E-01 Splitsmijn", "",
         f"Automatisch gegenereerd door `cad/controle.py`: het eindmodel ({P.VERSIE_EIND}) en de pasproef ({P.VERSIE}). "
         "Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "![Valbord](valbord.png)", "", "## 1. Muntdoorgang", "", "Het valbord (in beide modellen hetzelfde):", ""]
    d = r["doorgang"]
    L += [f"- {ok(d['kleinste_doorgang'] >= P.MUNT_D + 3)} Kleinste opening tussen twee pinnen of een pin en de "
          f"wand: {f(d['kleinste_doorgang'], 1)} mm, voor een muntje van {P.MUNT_D:.0f} mm.",
          f"- {ok(not d['plekken'])} Plekken waar een muntje op twee steunpunten kan blijven liggen (ook met "
          "wrijving 0,3 meegerekend): " + ("geen." if not d["plekken"] else f"{d['plekken']}"),
          f"- {ok(d['boven_plank'] >= 3)} Boven de vlakke kantelplank is er nog {d['boven_plank']:.0f} mm tot het "
          "plafond: een muntje rolt er vrij over. De verdeler staat recht onder de middelste pin van de onderste rij "
          "(tussenruimte kleiner dan een muntje): een muntje valt er altijd naast."]
    b = r["mijnberg"]
    wr = math.degrees(math.atan(0.5))
    tro, kbi = b["trechter_onderaan_x"], b["karretje_binnen_x"]
    L += ["", "Eindmodel, van de spleet tot in de karretjes:", "",
          f"- {ok(b['sleuf_y'][0] < b['spleet_y'][0] and b['sleuf_y'][1] > b['spleet_y'][1])} De sleuf in het dek "
          f"(y {f(b['sleuf_y'][0], 1)} tot {f(b['sleuf_y'][1], 1)}) is ruimer dan de spleet ({f(b['spleet_y'][0], 1)} tot "
          f"{f(b['spleet_y'][1], 1)}).",
          f"- {ok(min(b['trechter_buiten_graden'], b['trechter_wig_graden']) > wr + 10)} De trechters staan "
          f"{b['trechter_buiten_graden']:.0f}° (buitenkant) en {b['trechter_wig_graden']:.0f}° (middenwig) steil: een "
          f"muntje glijdt eraf, ook bij wrijving 0,5 (dat is {wr:.0f}°).",
          f"- {ok(tro[0] >= kbi[0] and tro[1] <= kbi[1])} Onderaan lopen de trechters van x = {f(tro[0], 1)} tot "
          f"{f(tro[1], 1)}, binnen de bovenrand van een karretje ({f(kbi[0], 1)} tot {f(kbi[1], 1)}).",
          f"- {ok(P.KAR_RAND_ONDER_DEK < P.MUNT_T)} Tussen de bovenrand van een karretje en het dek is "
          f"{f(P.KAR_RAND_ONDER_DEK, 1)} mm: daar past geen muntje ({f(P.MUNT_T, 1)} mm) tussen.",
          f"- {ok(b['karretje_inhoud_cm3'] > 5 * b['tien_munten_cm3'])} Een karretje houdt {b['karretje_inhoud_cm3']:.0f} cm³; "
          f"10 muntjes zijn {b['tien_munten_cm3']:.0f} cm³. Er liggen er {b['munten_plat_op_de_bodem']} plat naast elkaar op "
          "de bodem.",
          "", "![Mijnberg](mijnberg.png)"]
    L += ["", "## 2. Valsimulatie", "",
          "Het valbord nagebouwd in een 2D-fysicamodel (pymunk, zie `cad/simulatie.py`). Een ronde zoals in de klas: "
          "één duw tegen de slinger, dan 8 muntjes met 0,8 tot 2 seconden ertussen, steeds gemikt op dezelfde plek. "
          f"Doorgerekend voor {len(S.MATERIAAL)} combinaties van stuit, wrijving en demping, en twee waarden voor de "
          "wrijving van het mes. Het eindmodel met de houweel (massa uit de geometrie) en de trechters van de mijnberg, "
          "tot het muntje onder de bovenrand van een karretje is.", "",
          "| | Eindmodel | Pasproef |", "|---|---|---|"]

    def rij_sim(s):
        ps = [p for *_, rij in s["tabel"] for p in rij]
        return {"kans": f"{ok(max(abs(p - 0.5) for p in ps) <= 0.2)} {f(min(ps))} tot {f(max(ps))}",
                "mik": f"{ok(all(abs(p - 0.5) <= 0.08 for p in s['per_mik']))} " + " / ".join(f(p) for p in s["per_mik"]),
                "effect": f"{ok(max(abs(v) for v in s['mikeffect_per_stuit'].values()) <= 0.08)} "
                          + ", ".join(f"{k.replace('.', ',')}: {v:+.2f}".replace(".", ",")
                                      for k, v in s["mikeffect_per_stuit"].items()),
                "vast": f"{ok(s['vast'] <= max(1, s['totaal'] // 500))} {s['vast']} van {s['totaal']}",
                "uitslag": f"{s['mediaan_uitslag']:.0f}° (minstens {s['kleinste_uitslag']:.0f}°)"}
    se, sp_ = rij_sim(e["simulatie"]), rij_sim(pp["simulatie"])
    for k, titel in (("kans", "Kans op links, elke combinatie en elk mikpunt"),
                     ("mik", "Gemiddeld, mikken links / midden / rechts"),
                     ("effect", "Mikeffect (links min rechts mikken) per stuit"),
                     ("vast", "Muntjes die bleven hangen"),
                     ("uitslag", "Uitslag van de slinger bij het laatste muntje (mediaan)")):
        L.append(f"| {titel} | {se[k]} | {sp_[k]} |")
    L += ["", f"Met {e['simulatie']['muntjes_per_cel']} muntjes per vakje is ±0,15 op de kans gewoon toeval. Zonder "
          "slinger is het mikeffect 0,3 tot 0,9: het muntje rolt groot naast de pinnen in een vaste baan, en dan "
          "beslist het mikpunt. De kantelplank op de slinger maakt het eerlijk.", "",
          "Hoe vaak elke splitsing van 8 voorkomt (links + rechts), naast een eerlijk muntstuk:", "",
          "| Splitsing | " + " | ".join(f"{k}+{8 - k}" for k in range(9)) + " |", "|---|" + "---|" * 9]
    for naam, s in (("Eindmodel", e["simulatie"]), ("Pasproef", pp["simulatie"])):
        L.append(f"| {naam} | " + " | ".join(f"{t / sum(s['splitsingen']) * 100:.0f} %" for t in s["splitsingen"]) + " |")
    L.append("| Binomiaal | " + " | ".join(f"{b_ * 100:.0f} %" for b_ in S.binomiaal(8)) + " |")

    def spreiding(t):
        n = sum(t)
        g = sum(k * v for k, v in enumerate(t)) / n
        return g, sum((k - g) ** 2 * v for k, v in enumerate(t)) / n
    L.append("")
    for naam, s in (("Eindmodel", e["simulatie"]), ("Pasproef", pp["simulatie"])):
        g, var = spreiding(s["splitsingen"])
        L.append(f"- {ok(abs(g - 4) < 0.3 and 1.6 < var < 2.4)} {naam}: gemiddeld {f(g)} muntjes links, spreiding "
                 f"(variantie) {f(var)}; een eerlijk muntstuk geeft 4 en 2. Eén vakje wijkt bij {sum(s['splitsingen'])} rondes "
                 "gemakkelijk een paar procent af; de spreiding zegt of de splitsingen echt willekeurig zijn (te klein: "
                 "de muntjes wisselen elkaar af, te groot: ze volgen elkaar).")
    L += ["", "## 3. Slinger", "", "| | Eindmodel (houweel) | Pasproef |", "|---|---|---|"]
    ge, gp = e["slinger"], pp["slinger"]

    def mes_ok(g):
        return f"{ok(g['mes_draagt_y'][0] < g['zwaartepunt_y'] < g['mes_draagt_y'][1])} y = {f(g['zwaartepunt_y'], 1)} " \
               f"(mes {f(g['mes_draagt_y'][0], 1)} tot {f(g['mes_draagt_y'][1], 1)})"
    L += [f"| Massa, zwaartepunt onder het mes | {ge['massa_g']:.0f} g, {ge['zwaartepunt_onder_as']:.0f} mm | "
          f"{gp['massa_g']:.0f} g, {gp['zwaartepunt_onder_as']:.0f} mm |",
          f"| Slingertijd | {f(ge['slingertijd_s'])} s | {f(gp['slingertijd_s'])} s |",
          f"| Zwaartepunt van wat draait, boven het mes | {mes_ok(ge)} | {mes_ok(gp)} |",
          f"| Draaien voor de flanken van het mes de groef raken | {ok(ge['draaien_voor_de_flanken_raken'] > P.MAX_HOEK)} "
          f"{ge['draaien_voor_de_flanken_raken']:.0f}° | {ok(gp['draaien_voor_de_flanken_raken'] > P.MAX_HOEK)} "
          f"{gp['draaien_voor_de_flanken_raken']:.0f}° |"]
    for p in S.PEN_WRIJVING:
        L.append(f"| Uitslag om de 5 s na een duw van 35°, mes als pen met wrijving {f(p)} | "
                 + ", ".join(f"{a:.0f}°" for a in ge["uitdemping"][str(p)]) + " | "
                 + ", ".join(f"{a:.0f}°" for a in gp["uitdemping"][str(p)]) + " |")
    L += ["", "De slinger slaat hooguit 35° uit. Een gewone geprinte pen (wrijvingsgetal 0,2 tot 0,4) valt na 10 tot "
          "20 seconden stil: daarom het mes.", "", "## 4. Spelingen en botsingen over de hele uitslag (±35°)"]
    for naam, x in (("Eindmodel", e), ("Pasproef", pp)):
        sp = x["spelingen"]
        L += ["", f"**{naam}**: {ok(not sp['botsingen'])} botsingen: "
              + ("geen" if not sp["botsingen"] else f"{sp['botsingen'][:6]}")
              + f"; {ok(sp['mes_op_groef'] < 0.05)} het mes rust op de bodem van de groef ({f(sp['mes_op_groef'])} mm).", "",
              "| Bewegend - vast | Kleinste speling | Bij hoek |", "|---|---|---|"]
        for p in sp["paren"]:
            L.append(f"| {p['paar']} | {ok(p['speling'] >= 0.3)} {f(p['speling'])} | {p['hoek']:.0f}° |")
    L += ["", "## 5. Verbindingen", "",
          "Er zijn geen klikhaakjes: alles klemt of schuift. Speling per kant in het CAD, gemeten tussen de onderdelen "
          "in de montagestand. Hoe strak het geprint wordt, hangt af van de printer: een staand gat komt meestal "
          "0,1 tot 0,2 mm kleiner in diameter uit. Print daarom eerst plaat 1 (de passtrip) en zet de gekozen maten in "
          "`cad/params.py`."]
    for naam, x in (("Eindmodel", e), ("Pasproef", pp)):
        L += ["", f"**{naam}**", "", "| Verbinding | Speling per kant | Bedoeld | Kies met |", "|---|---|---|---|"]
        for v in x["passingen"]:
            L.append(f"| {v['wat']} | {f(v['speling'])} mm | {v['soort']} | {('`' + v['param'] + '`') if v['param'] else ''} |")
    achteruit = b["kantelkracht_achteruit_N"]
    L += ["", "## 6. Stabiliteit (eindmodel)", "",
          f"- Alles samen ongeveer {b['massa_g']:.0f} g (geschat, met 10 muntjes). Het zwaartepunt ligt "
          f"{f(b['zwaartepunt_z_boven_tafel'], 0)} mm boven de tafel, op y = {f(b['zwaartepunt_y'], 1)}; de voet loopt van "
          f"y = {b['voet_y'][0]:.0f} tot {b['voet_y'][1]:.0f}.",
          f"- {ok(achteruit >= 3.5)} Wie de houweelkop naar achteren duwt, moet {f(achteruit, 1)} N (~{achteruit * 102:.0f} g) "
          f"duwen voor het bord kantelt; naar voren {f(b['kantelkracht_vooruit_N'], 1)} N, opzij "
          f"{f(b['kantelkracht_opzij_N'], 1)} N. Een duw om te slingeren gaat opzij, en de slinger zwaait dan weg.",
          f"- {ok(b['kantelspel_boven_mm'] <= 6)} Het bord staat met {f(P.SPEL)} mm speling per kant {f(P.KOKER_H, 0)} mm diep "
          f"in de kokervoeten: het kan {f(b['kantelspel_graden'], 1)}° naar voren of achteren wiebelen, bovenaan "
          f"{f(b['kantelspel_boven_mm'], 1)} mm. Recht laten staan hoeft niet: het valt vanzelf terug."]
    L += ["", "## 7. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "", "**Eindmodel** (`print/splitsmijn.3mf`)", "",
          "| Plaat | Kleuren | Maat | Past | Overhang > 48° | Platte plafonds (bruggen) | Te smal | Kleurwissels "
          "| Gewicht (schatting) | Slicer: gewicht | Slicer: tijd | Slicer: support |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot, tot_s, sec = 0.0, 0.0, 0
    se = r.get("slicer_eind", {})
    for naam, i in r["print_eind"].items():
        kl = " + ".join(P.FILAMENTEN_EIND[c][0] for c in i["kleuren_codes"])
        oh = sum(o["oppervlak_mm2"] for o in i["overhang"] if o["hoek_tov_horizontaal_graden"] >= 20)
        brug = sum(o["oppervlak_mm2"] for o in i["overhang"] if o["hoek_tov_horizontaal_graden"] < 20)
        dun = max((t["grootste_te_smalle_stuk_mm2"] for t in i["dunne_details"]), default=0.0)
        tot += i["gram_geschat"]
        rij = (f"| {naam} | {kl} | {' × '.join(f'{v:.0f}' for v in i['maat_mm'])} | {ok(i['past_op_A1'])} | "
               f"{ok(oh < 5)} {oh:.0f} mm² | {brug:.0f} mm² | {'✅ geen' if dun < 1.5 else '⚠️ ' + f(dun, 1) + ' mm²'} | "
               f"{i['wissels']} | ~{i['gram_geschat']:.0f} g |")
        if naam in se:
            s_ = se[naam]
            tot_s += s_["gram"]
            sec += s_["seconden"]
            sup = sum(g["support_extrusie_mm"] for g in s_["groepen"]) * math.pi * 0.875 ** 2 * 1.24 / 1000
            rij += f" {f(s_['gram'], 1)} g | {s_['tijd']} | {ok(sup < 1)} {f(sup, 1)} g |"
        else:
            rij += " | | |"
        L.append(rij)
    L += ["", f"Samen ~{tot:.0f} g geschat uit de doorsneden (met de wanden en de infill uit het project)."
          + (f" De slicer komt op {f(tot_s, 0)} g en {sec // 3600} u {sec % 3600 // 60:02d} printtijd." if se else ""), "",
          "- *Slicer*: PrusaSlicer met een profiel dat de A1 benadert (`cad/slice.py`), elk object met zijn eigen wanden en "
          "infill uit het project, in één kleur: de spoeling van de AMS bij een kleurwissel (plaat 6 en 9) komt erbij. "
          "Het gewicht is betrouwbaar, de tijd een indicatie. *Support* is wat PrusaSlicer met automatische support "
          "(drempel 30°) zou bijzetten; het project zelf print zonder support.",
          "- Plaat 7: PrusaSlicer zet een paar steuntjes onder het plafond van de gleuf in de houweelkop, rond de twee "
          "gaatjes. Zonder support print het dat als brug (5,6 mm breed), en de gaatjes worden in twee stappen "
          "overbrugd: eerst een strook, dan een vierkant.",
          "- *Platte plafonds*: de bodem van de uitsparingen voor de kokervoeten en de sleuf voor de vin in de mijnberg "
          "(die ondersteboven ligt), de gleuf voor de steel in de houweelkop. Ze zijn hoogstens 13 mm breed en aan twee "
          "kanten vast: de printer overbrugt ze. Korte bruggen (< 3 mm) tellen niet mee.",
          "- *Te smal*: kleiner dan 1,5 mm² per laag telt niet (de onderste laag van een schuine rand); de slicer laat dat weg.",
          "- *Kleurwissels*: geteld laag per laag. Op plaat 6 liggen de letters in de bovenste lagen van het portaal, en "
          "de andere delen op die plaat zijn hoger: een paar lagen met twee kleuren. De munten wisselen één keer, halverwege.",
          "", "**Pasproef** (`pasproef/pasproef.3mf`)", "",
          "| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |", "|---|---|---|---|---|---|"]
    for naam, i in r["print_proef"].items():
        kl = P.FILAMENTEN[i["kleuren_codes"][0]][0]
        oh = sum(o["oppervlak_mm2"] for o in i["overhang"])
        dun = "geen" if not i["dunne_details"] else f"{len(i['dunne_details'])} plekken"
        L.append(f"| {naam} | {kl} | {' × '.join(f'{v:.0f}' for v in i['maat_mm'])} | {ok(i['past_op_A1'])} | "
                 f"{ok(oh < 5)} {oh:.0f} | {ok(not i['dunne_details'])} {dun} |")
    if r.get("slicer"):
        L += ["", "Slicer van de pasproef (PrusaSlicer, A1-achtig profiel):", "",
              "| Plaat | Filament | Tijd (indicatie) | Support nodig |", "|---|---|---|---|"]
        for naam, s_ in r["slicer"].items():
            L.append(f"| {naam} | {s_['gram']:.1f} g | {s_['tijd']} | {ok(not s_['support'])} "
                     f"{'ja' if s_['support'] else 'nee'} |")
    return "\n".join(L) + "\n"


def main(argv):
    UIT.mkdir(exist_ok=True)
    snel = "--snel" in argv
    rondes, splits = (3, 60) if snel else (8, 600)
    r = {"doorgang": ligplekken(), "mijnberg": mijnberg()}
    for naam, md in modellen().items():
        print("simulatie", naam, flush=True)
        r[naam] = {"simulatie": simulatie(md, rondes, splits), "slinger": slinger(md), "spelingen": spelingen(md),
                   "passingen": passingen(md)}
    r["print_eind"], r["print_proef"] = printbaar_eind(), printbaar_proef()
    for sleutel, bestand in (("slicer", "slicer.json"), ("slicer_eind", "slicer_eind.json")):
        if (UIT / bestand).exists():
            r[sleutel] = json.loads((UIT / bestand).read_text())
    md = modellen()["eindmodel"]
    tekening(UIT / "valbord.png", md, banen(md), r["eindmodel"]["simulatie"]["splitsingen"])
    tekening_berg(UIT / "mijnberg.png")
    (UIT / "controle.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str) + "\n")
    tekst = rapport(r)
    (UIT / "RAPPORT.md").write_text(tekst)
    print(tekst)


if __name__ == "__main__":
    main(sys.argv[1:])
