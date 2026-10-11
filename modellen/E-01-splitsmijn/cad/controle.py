"""Montage- en printcontrole van de pasproef van E-01 Splitsmijn (het valbord).

Gebruik:  python controle.py        (python slice.py eerst, voor gewicht en tijd in het rapport)
Schrijft  controle/RAPPORT.md, controle/controle.json en controle/valbord.png

  1. Muntdoorgang: past een muntje overal tussen de pinnen door, en is er nergens een plek waar het tussen
     twee pinnen (of een pin en de wand) kan blijven liggen?
  2. Valsimulatie (simulatie.py): links of rechts, over een reeks materiaalwaarden en mikpunten;
     hoe vaak elke splitsing van 8 voorkomt; blijft er ooit een muntje hangen.
  3. Slinger: slingertijd, hoe lang hij doorslingert, vrije uitslag, zwaartepunt boven het mes.
  4. Spelingen en botsingen over de hele uitslag van de slinger.
  5. Printbaarheid per plaat (gereedschap/printcheck.py) en de slicer.
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
import valbord as V  # noqa: E402
from gereedschap import printcheck  # noqa: E402
from gereedschap.geom import cs_to_shapely  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UIT = ROOT / "controle"
KLEUR = {P.ZWART: "#3A3A3A", P.BRUIN: "#C9A98F", P.ROOD: "#9D2235", P.BLAUW: "#0078BF"}
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


# --- 3/4. slinger, spelingen ----------------------------------------------------------------------

def slinger():
    m, d, I = S.slinger_massa()
    # zwaartepunt in y van het draaiende deel (slinger, mesblok, plank)
    delen = [V.slinger(), V.mesblok(), V.plank()]
    vol = [x.volume() for x in delen]
    cy = []
    for x in delen:
        tm = x.to_mesh()
        v = np.asarray(tm.vert_properties[:, :3])
        f = np.asarray(tm.tri_verts)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        sv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        cy.append(float(((a + b + c)[:, 1] / 4 * sv).sum() / sv.sum()))
    cy_tot = sum(c * w * (3.0 if i == 0 else 1.0) for i, (c, w) in enumerate(zip(cy, vol))) / \
        sum(w * (3.0 if i == 0 else 1.0) for i, w in enumerate(vol))
    knife = (P.BEUGEL_Y, P.MES[1])
    flank_mes = P.MES_HOEK / 2
    flank_groef = 90 - (180 - P.GROEF_HOEK) / 2
    return {"slingertijd_s": round(S.slingertijd(), 2), "massa_g": round(m, 0),
            "zwaartepunt_y": round(cy_tot, 1), "mes_draagt_y": [round(knife[0], 1), round(knife[1], 1)],
            "draaien_voor_de_flanken_raken": round(flank_groef - flank_mes, 0),
            "uitdemping": {str(p): S.uitdemping(p) for p in S.PEN_WRIJVING}}


def spelingen():
    """Kleinste speling tussen de bewegende en de vaste delen over de hele uitslag; botsingen."""
    paren = [("plank", "achterplaat", "plank - achterplaat"), ("plank", "voorplaat", "plank en naaf - voorplaat"),
             ("slinger", "voorplaat", "slingerarm - voorplaat"), ("slinger", "beugel", "slingerarm - beugel"),
             ("slinger", "voet", "gewicht - voet"), ("mesblok", "voorplaat", "mesblok - voorplaat")]
    hoeken = np.linspace(-P.MAX_HOEK, P.MAX_HOEK, 15)
    uit, botsing = [], []
    standen = [V.stand(float(h)) for h in hoeken]
    for a, b, titel in paren:
        best, waar = 99.0, 0.0
        for h, st in zip(hoeken, standen):
            g = float(st[a][1].min_gap(st[b][1], 6.0))
            if g < best:
                best, waar = g, float(h)
        uit.append({"paar": titel, "speling": round(best, 2), "hoek": round(waar, 0)})
    for h, st in zip(hoeken, standen):
        for a, b in itertools.combinations(st, 2):
            v = (st[a][1] ^ st[b][1]).volume()
            if v > 0.01:
                botsing.append((round(float(h)), a, b, round(v, 2)))
    # het mes moet alleen met zijn punt op de groef rusten
    st = V.stand(0.0)
    mes_raakt = float(st["mesblok"][1].min_gap(st["beugel"][1], 2.0))
    return {"paren": uit, "botsingen": botsing, "mes_op_groef": round(mes_raakt, 2)}


# --- 5. verbindingen ----------------------------------------------------------------------------

def passingen():
    """Speling per kant in elke verbinding, gemeten tussen de meshes in de montagestand (alleen rond die
    verbinding, en zonder de vlakken waar de delen bewust op elkaar rusten)."""
    st = {k: m for k, (_, m) in V.stand(0.0).items()}
    zp, W = P.Z_PLANK, P.BINNEN_HALF

    def doos(x0, y0, z0, x1, y1, z1):
        return V.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

    def speling(a, b, d):
        return round(float((st[a] ^ d).min_gap(st[b] ^ d, 3.0)), 3)

    x, z = V.nokken()[0]
    px, pz, _ = P.BEUGEL_PEN
    mx, mz, _ = P.MES_PEN
    return [
        {"wat": f"nokje Ø {P.NOK_D:.1f} in de voorplaat ({len(V.nokken())} stuks tegelijk)", "param": "NOK_GAT",
         "speling": speling("achterplaat", "voorplaat", doos(x - 3, P.Y_VOOR + 0.05, z - 3, x + 3, P.Y_VOOR + 3, z + 3)),
         "soort": "licht klemmen"},
        {"wat": "pennetjes Ø 3 van de arm in het mesblok", "param": "MES_GAT",
         "speling": speling("slinger", "mesblok", doos(mx - 3, P.MES[0] + 0.01, zp + mz - 3, mx + 3, P.MES[1], zp + mz + 3)),
         "soort": "klemmen"},
        {"wat": "pennen Ø 4 van de voorplaat in de beugel (liggend druppelgat)", "param": "BEUGEL_GAT",
         "speling": speling("voorplaat", "beugel", doos(px - 4, P.BORD_T + 0.5, zp - pz - 4, px + 4, 25, zp - pz + 4)),
         "soort": "klemmen"},
        {"wat": "zeskant van de plank in de arm", "param": "ZESKANT_SPEL",
         "speling": speling("plank", "slinger", doos(-7, P.ARM_Y[0] + 0.2, zp - 7, 7, P.ARM_Y[1], zp + 7)),
         "soort": "schuiven zonder wiebelen"},
        {"wat": "zijranden in de kokers van de voet", "param": "SPEL",
         "speling": min(speling(a, "voet", doos(W - 1, -5, 0.5, W + P.RAND + 5, 20, 15)) for a in ("achterplaat", "voorplaat")),
         "soort": "schuiven"},
        {"wat": "naaf van de plank in het gat van de voorplaat", "param": "NAAF_R",
         "speling": speling("plank", "voorplaat", doos(-8, P.Y_VOOR, zp - 8, 8, P.BORD_T, zp + 8)), "soort": "vrij"},
        {"wat": "vin van de verdeler in de spleet", "param": "",
         "speling": min(speling("voet", a, doos(-3, 0, 0.5, 3, 14, P.VERDELER_TOP)) for a in ("achterplaat", "voorplaat")),
         "soort": "vrij"},
    ]


# --- 6. printbaarheid ---------------------------------------------------------------------------

def printbaar():
    out = {}
    for naam, kleur, delen in platen.platen():
        i = printcheck.plaat({kleur: V.Manifold.batch_boolean([m for _, m in delen], V.OpType.Add)})
        i["kleur"] = kleur
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


def tekening(pad, banen, telling):
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 3, 1)
    st = V.stand(0.0)
    y_mid = (P.ACHTER_T + P.Y_VOOR) / 2
    for naam in ("achterplaat", "voet", "plank"):
        _vul(ax, snede_y(st[naam][1], y_mid), KLEUR[st[naam][0]])
    for h, kl in ((-P.MAX_HOEK, "#E8A0AA"), (P.MAX_HOEK, "#E8A0AA"), (0.0, KLEUR[P.ROOD])):
        sl = V.stand(h)["slinger"][1]
        _vul(ax, snede_y(sl, (P.ARM_Y[0] + P.ARM_Y[1]) / 2), kl, ec="none" if h else "#222")
    for xs, zs in banen:
        ax.plot(xs, zs, lw=0.7, alpha=0.8)
    ax.set_title("vooraanzicht zoals het kind het ziet: pinnen, kantelplank, slinger (±35°), gesimuleerde banen",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_xaxis()          # het kind kijkt van +y: +x is links
    ax.set_ylim(-35, P.BORD_H + 8)
    ax2 = fig.add_subplot(1, 3, 2)
    for naam, (kl, m) in st.items():
        _vul(ax2, snede_x(m, 0.0), KLEUR.get(kl, "#888"))
    ax2.set_title("doorsnede midden: spleet, voorplaat, beugel met mes, slinger", fontsize=9)
    ax2.set_aspect("equal")
    ax2.autoscale_view()
    ax2.set_xlim(-25, 45)
    ax2.set_ylim(150, 205)
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
    fig.savefig(pad, dpi=85)
    plt.close(fig)


def banen(n=10, seed=4):
    import random
    rnd = random.Random(seed)
    out = []
    sp, sl = S.ruimte(0.5, 0.3, 0.04)
    S.zet_uitslag(sl, 30)
    for _ in range(n):
        for _ in range(int(rnd.uniform(0.6, 1.5) / S.DT)):
            sp.step(S.DT)
        b, c = S.nieuwe_munt(sp, rnd, 0.0, 0.5, 0.3)
        xs, zs, t = [], [], 0.0
        while b.position.y > -30 and t < 5:
            sp.step(S.DT)
            t += S.DT
            xs.append(b.position.x)
            zs.append(b.position.y)
        sp.remove(b, c)
        out.append((xs, zs))
    return out


# --- rapport ------------------------------------------------------------------------------------

def ok(v):
    return "✅" if v else "⚠️"


def rapport(r):
    L = ["# Controle E-01 Splitsmijn: pasproef van het valbord", "",
         f"Automatisch gegenereerd door `cad/controle.py` ({P.VERSIE}). Maten in mm. ✅ = in orde, ⚠️ = bekijken.", "",
         "![Valbord](valbord.png)", "", "## 1. Muntdoorgang", ""]
    d = r["doorgang"]
    L += [f"- {ok(d['kleinste_doorgang'] >= P.MUNT_D + 3)} Kleinste opening tussen twee pinnen of een pin en de "
          f"wand: {d['kleinste_doorgang']} mm, voor een muntje van {P.MUNT_D:.0f} mm.",
          f"- {ok(not d['plekken'])} Plekken waar een muntje op twee steunpunten kan blijven liggen (ook met "
          "wrijving 0,3 meegerekend): " + ("geen." if not d["plekken"] else f"{d['plekken']}"),
          f"- {ok(d['boven_plank'] >= 3)} Boven de vlakke kantelplank is er nog {d['boven_plank']:.0f} mm tot het "
          "plafond: een muntje rolt er vrij over. De verdeler staat recht onder de middelste pin van de onderste rij "
          "(tussenruimte kleiner dan een muntje): een muntje valt er altijd naast."]
    s = r["simulatie"]
    ps = [p for *_, rij in s["tabel"] for p in rij]
    L += ["", "## 2. Valsimulatie", "",
          "Het valbord nagebouwd in een 2D-fysicamodel (pymunk, zie `cad/simulatie.py`). Een ronde zoals in de klas: "
          "één duw tegen de slinger, dan 8 muntjes met 0,8 tot 2 seconden ertussen, steeds gemikt op dezelfde plek. "
          f"Doorgerekend voor {len(S.MATERIAAL)} combinaties van stuit, wrijving en demping, en twee waarden voor de "
          "wrijving van het mes.", "",
          f"- {ok(max(abs(p - 0.5) for p in ps) <= 0.2)} Kans op links, voor elke combinatie en elk mikpunt (links, "
          f"midden, rechts in de trechter): {min(ps):.2f} tot {max(ps):.2f}. Met {s['muntjes_per_cel']} muntjes per "
          "vakje is ±0,15 gewoon toeval.",
          f"- {ok(all(abs(p - 0.5) <= 0.08 for p in s['per_mik']))} Gemiddeld over alle combinaties: mikken links "
          f"{s['per_mik'][0]:.2f}, midden {s['per_mik'][1]:.2f}, rechts {s['per_mik'][2]:.2f} naar links (de drie "
          "mikpunten met dezelfde duwen en tijden). Het mikpunt beslist niet meer.",
          f"- {ok(max(abs(v) for v in s['mikeffect_per_stuit'].values()) <= 0.08)} Mikeffect (links mikken min rechts "
          "mikken) per stuit: " + ", ".join(f"{k}: {v:+.2f}" for k, v in s["mikeffect_per_stuit"].items())
          + ". Zonder slinger is dat 0,3 tot 0,9.",
          f"- {ok(s['vast'] <= max(1, s['totaal'] // 500))} Muntjes die bleven hangen: {s['vast']} van {s['totaal']}.",
          f"- De slinger had bij het laatste muntje van een ronde nog {s['mediaan_uitslag']:.0f}° uitslag (mediaan), "
          f"minstens {s['kleinste_uitslag']:.0f}°: één duw is genoeg voor een hele ronde.",
          "", "Hoe vaak elke splitsing van 8 voorkomt (links + rechts), naast een eerlijk muntstuk:", "",
          "| Splitsing | " + " | ".join(f"{k}+{8 - k}" for k in range(9)) + " |", "|---|" + "---|" * 9,
          "| Simulatie | " + " | ".join(f"{t / sum(s['splitsingen']) * 100:.0f} %" for t in s["splitsingen"]) + " |",
          "| Binomiaal | " + " | ".join(f"{b * 100:.0f} %" for b in S.binomiaal(8)) + " |", "",
          "Het muntje rolt groot naast de pinnen in een vaste baan over de pinnen: zonder slinger beslist het "
          "mikpunt (in de simulatie 65 tot 95 % naar de kant waar het kind mikt). De kantelplank op de slinger "
          "maakt het eerlijk: welke kant op dat moment lager ligt."]
    g = r["slinger"]
    L += ["", "## 3. Slinger", "",
          f"- Slingertijd {g['slingertijd_s']} s, massa ~{g['massa_g']:.0f} g (gewicht helemaal massief).",
          f"- {ok(g['mes_draagt_y'][0] < g['zwaartepunt_y'] < g['mes_draagt_y'][1])} Het zwaartepunt van wat draait "
          f"ligt op y = {g['zwaartepunt_y']}, boven het stuk waar het mes op de groef rust ({g['mes_draagt_y'][0]} tot "
          f"{g['mes_draagt_y'][1]}): de slinger kantelt niet naar voren of naar achteren.",
          f"- {ok(g['draaien_voor_de_flanken_raken'] > P.MAX_HOEK)} Het mes kan {g['draaien_voor_de_flanken_raken']:.0f}° "
          f"draaien voor zijn flanken de groef raken; de slinger slaat hooguit {P.MAX_HOEK:.0f}° uit.",
          "- Uitslag na een duw van 35°, om de 5 seconden (zonder muntjes):"]
    for p, rij in g["uitdemping"].items():
        L.append(f"  - wrijving van het mes als een pen van 4 mm met wrijvingsgetal {p}: "
                 + ", ".join(f"{a:.0f}°" for a in rij))
    L += ["  Een gewone geprinte pen (wrijvingsgetal 0,2 tot 0,4) valt na 10 tot 20 seconden stil: daarom het mes."]
    sp = r["spelingen"]
    L += ["", "## 4. Spelingen en botsingen over de hele uitslag (±35°)", "",
          f"- {ok(not sp['botsingen'])} Botsingen: " + ("geen." if not sp["botsingen"] else f"{sp['botsingen'][:6]}"),
          f"- {ok(sp['mes_op_groef'] < 0.05)} Het mes rust op de bodem van de groef ({sp['mes_op_groef']} mm).", "",
          "| Paar | Kleinste speling | Bij hoek |", "|---|---|---|"]
    for p in sp["paren"]:
        L.append(f"| {p['paar']} | {ok(p['speling'] >= 0.3)} {p['speling']:.2f} | {p['hoek']:.0f}° |")
    L += ["", "## 5. Verbindingen", "",
          "Er zijn geen klikhaakjes: alles klemt of schuift. Speling per kant in het CAD, gemeten tussen de onderdelen "
          "in de montagestand. Hoe strak het geprint wordt, hangt af van de printer: een staand gat komt meestal "
          "0,1 tot 0,2 mm kleiner in diameter uit. Print daarom eerst plaat 1 (de passtrip) en zet de gekozen maten in "
          "`cad/params.py`.", "",
          "| Verbinding | Speling per kant | Bedoeld | Kies met |", "|---|---|---|---|"]
    for v in r["passingen"]:
        L.append(f"| {v['wat']} | {v['speling']:.2f} mm | {v['soort']} | {('`' + v['param'] + '`') if v['param'] else ''} |")
    L += ["", "## 6. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)", "",
          "| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |", "|---|---|---|---|---|---|"]
    for naam, i in r["print"].items():
        f = P.FILAMENTEN[i["kleur"]][0]
        maat = " × ".join(f"{v:.0f}" for v in i["maat_mm"])
        oh = sum(o["oppervlak_mm2"] for o in i["overhang"])
        dun = "geen" if not i["dunne_details"] else f"{len(i['dunne_details'])} plekken"
        L.append(f"| {naam} | {f} | {maat} | {ok(i['past_op_A1'])} | {ok(oh < 5)} {oh:.0f} | {ok(not i['dunne_details'])} {dun} |")
    if r.get("slicer"):
        L += ["", "### Slicer (PrusaSlicer, A1-achtig profiel)", "", "| Plaat | Filament | Tijd (indicatie) | Support nodig |",
              "|---|---|---|---|"]
        for naam, s_ in r["slicer"].items():
            L.append(f"| {naam} | {s_['gram']:.1f} g | {s_['tijd']} | {ok(not s_['support'])} {'ja' if s_['support'] else 'nee'} |")
    return "\n".join(L) + "\n"


def main(argv):
    UIT.mkdir(exist_ok=True)
    rondes = 4 if "--snel" in argv else 8
    tabel, vast, uitslagen = S.mikproef(rondes=rondes)
    telling = S.splitsingen(rondes=60 if "--snel" in argv else 300)
    r = {"doorgang": ligplekken(), "slinger": slinger(), "spelingen": spelingen(), "passingen": passingen(),
         "print": printbaar(),
         "simulatie": {"tabel": [(m, p, [round(x, 3) for x in rij]) for m, p, rij in tabel], "vast": vast,
                       "totaal": len(tabel) * 3 * rondes * 8, "muntjes_per_cel": rondes * 8,
                       "kleinste_uitslag": min(uitslagen), "mediaan_uitslag": statistics.median(uitslagen),
                       "per_mik": [statistics.mean(rij[i] for *_, rij in tabel) for i in range(3)],
                       "mikeffect_per_stuit": {str(st): round(statistics.mean(rij[0] - rij[2] for m, _, rij in tabel
                                                                              if m["stuit"] == st), 3)
                                               for st in sorted({m["stuit"] for m, _, _ in tabel})},
                       "splitsingen": telling}}
    sl = UIT / "slicer.json"
    if sl.exists():
        r["slicer"] = json.loads(sl.read_text())
    tekening(UIT / "valbord.png", banen(), telling)
    (UIT / "controle.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str) + "\n")
    tekst = rapport(r)
    (UIT / "RAPPORT.md").write_text(tekst)
    print(tekst)


if __name__ == "__main__":
    main(sys.argv[1:])
