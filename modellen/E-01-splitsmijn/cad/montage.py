"""Montage van E-01 De Splitsmijn (eindmodel), stap voor stap zoals in de README.

Gebruik:  python montage.py        (controle.py neemt dit ook op in controle/RAPPORT.md)

Elk onderdeel, of een groep die je samen beweegt, schuift van ver weg langs zijn richting tot op zijn plaats, in
stapjes van 0,25 mm. Bij elk stapje wordt gekeken of het iets raakt van wat er al staat (overlap > 0,01 mm³).
Daarna het omgekeerde: wat je bij het spelen weer uit elkaar haalt (karretjes uittrekken, bord eruit tillen). En de
inworp: kan een kind een muntje met twee vingers rechtop in de trechter laten zakken als het portaal erop staat?

Assen zoals overal: x opzij (+x is links voor het kind), y naar het kind toe, z omhoog. Het bord ligt bij stap 1 tot
6 plat op tafel met de pinnen omhoog; dan is 'erop leggen' bewegen langs -y.
"""
import numpy as np
from manifold3d import Manifold, OpType

import _pad  # noqa: F401
import mijn as M
import params as P
import valbord as V
from gereedschap.geom import box3

STAP = 0.25           # mm per stapje
OVERLAP = 0.01        # mm³: minder is aanraken, geen botsing

HOUWEEL = ("plank", "steel", "mesblok", "kop")
PLAAT = ("achterplaat", "plank", "voorplaat", "mijnhout")
BORD = PLAAT + HOUWEEL[1:] + ("beugel",)
BERG = ("mijnberg", "verdelervin", "kokervoet L", "kokervoet R", "karretje L", "karretje R")

# (stap, wat, onderdelen die bewegen, richting, afstand, vaste onderdelen)
STAPPEN = [
    ("2", "kantelplank op de achterplaat", ("plank",), (0, -1, 0), 15, PLAAT[:1]),
    ("3", "voorplaat erop: 26 nokjes in hun gaatjes, de naaf door het ronde gat", ("voorplaat",), (0, -1, 0), 25,
     PLAAT[:2]),
    ("4", "mijnhout op de voorplaat, nokjes in de gaatjes", ("mijnhout",), (0, -1, 0), 25, PLAAT[:3]),
    ("5a", "mesblok op de pennetjes van de steel", ("mesblok",), (0, -1, 0), 15, ("steel",)),
    ("5b", "houweelkop op de steel (de gleuf zit achteraan)", ("kop",), (0, -1, 0), 25, ("steel", "mesblok")),
    ("5c", "houweel met het zeskantgat op de zeskant van de plank", ("steel", "mesblok", "kop"), (0, -1, 0), 25,
     PLAAT),
    ("6", "beugel op de pennen onder de naaf, het mes glijdt in de groef", ("beugel",), (0, -1, 0), 25,
     PLAAT + HOUWEEL[1:]),
    ("7a", "verdelervin met haar tong in de sleuf van het dek", ("verdelervin",), (0, 0, -1), 15, BERG[:1]),
    ("7b", "kokervoet L in zijn uitsparing", ("kokervoet L",), (0, 0, -1), 20, BERG[:2]),
    ("7c", "kokervoet R in zijn uitsparing", ("kokervoet R",), (0, 0, -1), 20, BERG[:3]),
    ("7d", "karretje L van voren in zijn tunnel", ("karretje L",), (0, -1, 0), 120, BERG[:4]),
    ("7e", "karretje R van voren in zijn tunnel", ("karretje R",), (0, -1, 0), 120, BERG[:5]),
    ("8", "bord rechtop in de mijnberg: zijranden en mijnhout in de kokervoeten, vin in de spleet", BORD, (0, 0, -1),
     45, BERG),
    ("9a", "kokerkap L over de bovenhoek", ("kokerkap L",), (0, 0, -1), 25, BORD + BERG),
    ("9b", "kokerkap R over de bovenhoek", ("kokerkap R",), (0, 0, -1), 25, BORD + BERG + ("kokerkap L",)),
    ("9c", "portaal met zijn tongen in de kokerkappen", ("portaal", "letters"), (0, 0, -1), 50,
     BORD + BERG + ("kokerkap L", "kokerkap R")),
]
OPTILLEN = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)   # stap 6 met de houweel zoveel hoger (+) of lager (-) dan op het mes

# wat je bij het spelen of opbergen weer losmaakt: (wat, onderdelen, richting, afstand, vaste onderdelen)
LOS = [
    ("karretje L helemaal uittrekken", ("karretje L",), (0, 1, 0), 120, "rest"),
    ("karretje R helemaal uittrekken", ("karretje R",), (0, 1, 0), 120, "rest"),
    ("portaal eraf tillen", ("portaal", "letters"), (0, 0, 1), 60, "rest"),
    ("bord uit de mijnberg tillen (portaal en kokerkappen eraf)", BORD, (0, 0, 1), 45, BERG),
]

# inworp: een muntje rechtop, met de duim vooraan en de wijsvinger achteraan rond de bovenkant
VINGER = (16.0, 12.0, 8.0, 15.0)     # breedte (x), dikte (y), hoe ver over het muntje, hoe ver erboven
AANVOER = 1.0                         # de hand komt van voren met de onderkant van het muntje zo hoog boven het bord
LOS_LATEN = 6.0                       # en laat los als het muntje zo diep in de trechter zit


def _unie(ms):
    return Manifold.batch_boolean(list(ms), OpType.Add) if len(ms) > 1 else ms[0]


def schuif(bewegend, vast, richting, van, tot, stap=STAP):
    """Verschuift `bewegend` over richting * t, met t van `van` tot `tot`, en geeft de eerste botsing: None, of
    (t, naam van het vaste deel, overlap in mm³). Inschuiven over 20 mm langs v: van=-20, tot=0."""
    v = np.asarray(richting, dtype=float)
    a, b = np.array(bewegend.bounding_box()).reshape(2, 3)
    lo = np.minimum(a + v * van, a + v * tot) - 1
    hi = np.maximum(b + v * van, b + v * tot) + 1
    gebied = box3(*lo, *hi)
    vast = [(n, m ^ gebied) for n, m in vast.items()]
    vast = [(n, m) for n, m in vast if not m.is_empty()]
    n = max(1, int(round(abs(tot - van) / stap)))
    for t in np.linspace(van, tot, n + 1):
        m = bewegend.translate(list(v * t))
        for naam, f in vast:
            o = (m ^ f).volume()
            if o > OVERLAP:
                return round(float(t), 2), naam, round(o, 3)
    return None


def los_van(bewegend, vast, richting, t):
    """Staat `bewegend`, verschoven over richting * t, helemaal los: niets van de vaste delen binnen zijn
    omhullende doos?"""
    a, b = np.array(bewegend.translate(list(np.asarray(richting, dtype=float) * t)).bounding_box()).reshape(2, 3)
    doos = box3(*a, *b)
    return all((m ^ doos).volume() <= OVERLAP for m in vast.values())


def hand_met_munt(zb):
    """Muntje rechtop midden in de spleet met de onderkant op z = zb, en twee vingers rond de bovenkant."""
    b, d, over, boven = VINGER
    y = (P.ACHTER_T + P.Y_VOOR) / 2
    top = zb + P.MUNT_D
    munt = V.munt_op(0.0, zb + P.MUNT_D / 2, y)
    voor = box3(-b / 2, y + P.MUNT_T / 2, top - over, b / 2, y + P.MUNT_T / 2 + d, top + boven)
    achter = box3(-b / 2, y - P.MUNT_T / 2 - d, top - over, b / 2, y - P.MUNT_T / 2, top + boven)
    return munt, _unie([voor, achter])


def montage():
    st = {k: m for k, (_, m) in M.stand(0.0).items()}
    uit = {"stappen": [], "los": [], "inworp": {}, "optillen": {}}
    for nr, wat, delen, richting, afstand, vast in STAPPEN:
        m, f = _unie([st[d] for d in delen]), {n: st[n] for n in vast}
        uit["stappen"].append({"stap": nr, "wat": wat, "richting": richting, "afstand": afstand, "vast": list(vast),
                               "botsing": schuif(m, f, richting, -afstand, 0.0),
                               "los_bij_begin": los_van(m, f, richting, -afstand)})
    # stap 6 nog eens, met de houweel wat hoger of lager dan zijn plaats op het mes: de naaf heeft 1 mm speling in de
    # voorplaat, dus zonder beugel hangt hij waar je hem houdt
    for dz in OPTILLEN:
        vast = {n: (st[n].translate([0, 0, dz]) if n in HOUWEEL else st[n]) for n in PLAAT + HOUWEEL[1:]}
        uit["optillen"][dz] = schuif(st["beugel"], vast, (0, -1, 0), -25, 0.0)
    for wat, delen, richting, afstand, vast in LOS:
        vast = tuple(n for n in st if n not in delen) if vast == "rest" else vast
        m, f = _unie([st[d] for d in delen]), {n: st[n] for n in vast}
        uit["los"].append({"wat": wat, "richting": richting, "afstand": afstand,
                           "botsing": schuif(m, f, richting, 0.0, afstand),
                           "los_op_het_eind": los_van(m, f, richting, afstand)})
    # inworp: muntje en vingers komen van voren, net boven het bord, onder het portaal door tot boven de spleet, en
    # zakken dan tot het muntje LOS_LATEN mm in de trechter zit; daarna valt het muntje alleen tot op de kantelplank
    z_aan, z_los = P.BORD_H + AANVOER, P.BORD_H - LOS_LATEN
    munt, vingers = hand_met_munt(z_aan)
    hand = _unie([munt, vingers])
    uit["inworp"] = {
        "van_voren": schuif(hand, st, (0, -1, 0), -80.0, 0.0),
        "zakken": schuif(hand, st, (0, 0, -1), 0.0, z_aan - z_los),
        "munt_valt": schuif(munt.translate([0, 0, z_los - z_aan]), st, (0, 0, -1), 0.0,
                            z_los - (P.Z_PLANK + P.PLANK_H / 2 + 0.5)),
        "vingers_tot_portaal_mm": round(P.PORTAAL_Z[0] - (z_aan + P.MUNT_D + VINGER[3]), 1),
    }
    return uit


def regels(r):
    """Markdown voor controle/RAPPORT.md."""
    ok = lambda b: "✅" if b else "⚠️"  # noqa: E731
    pijl = {(0, -1, 0): "−y (erop / naar achteren)", (0, 1, 0): "+y (naar voren)", (0, 0, -1): "−z (omlaag)",
            (0, 0, 1): "+z (omhoog)"}

    def uitkomst(b, los=True):
        if b is not None:
            return f"⚠️ raakt {b[1]} op {str(abs(b[0])).replace('.', ',')} mm ({b[2]} mm³)"
        return "✅ vrij" if los else "⚠️ te korte weg: het begint of eindigt niet helemaal los"

    L = ["| Stap | Wat | Richting | Afstand | Tegen | Pad |", "|---|---|---|---|---|---|"]
    for s in r["stappen"]:
        tegen = ", ".join(s["vast"]) if len(s["vast"]) <= 3 else f"{len(s['vast'])} onderdelen"
        L.append(f"| {s['stap']} | {s['wat']} | {pijl[tuple(s['richting'])]} | {s['afstand']} mm | {tegen} | "
                 f"{uitkomst(s['botsing'], s['los_bij_begin'])} |")
    vrij = [dz for dz, b in r["optillen"].items() if b is None]
    L += ["", "Stap 6 met de houweel hoger of lager dan zijn plaats op het mes (de naaf heeft 1 mm speling, dus zonder "
          "beugel hangt hij waar je hem houdt):", "",
          "| Houweel | " + " | ".join(f"{dz:+.2f} mm".replace(".", ",") for dz in r["optillen"]) + " |",
          "|---|" + "---|" * len(r["optillen"]),
          "| Beugel erop schuiven | "
          + " | ".join("✅ vrij" if b is None else f"⚠️ raakt {b[1]}" for b in r["optillen"].values()) + " |", "",
          f"- {ok(len(vrij) >= 3)} Het mes glijdt in de groef als de houweel tussen "
          f"{str(min(vrij)).replace('.', ',')} en {str(max(vrij)).replace('.', ',')} mm van zijn plaats hangt. "
          "Hangt hij lager, dan haakt het mes achter de groef: til de houweel dus een beetje op."]
    L += ["", "Uit elkaar (bij het spelen en opbergen), met al de rest op zijn plaats:", "",
          "| Wat | Richting | Afstand | Pad |", "|---|---|---|---|"]
    for s in r["los"]:
        L.append(f"| {s['wat']} | {pijl[tuple(s['richting'])]} | {s['afstand']} mm | "
                 f"{uitkomst(s['botsing'], s['los_op_het_eind'])} |")
    i = r["inworp"]
    L += ["", f"**Inworp**, met alles erop: een muntje rechtop, met de duim vooraan en de wijsvinger achteraan "
          f"({VINGER[0]:.0f} mm breed, {VINGER[1]:.0f} mm dik, tot {VINGER[3]:.0f} mm boven het muntje). De hand komt "
          f"van voren, met het muntje {AANVOER:.0f} mm boven het bord, onder het portaal door tot boven de spleet. Dan "
          f"zakt ze tot het muntje {LOS_LATEN:.0f} mm in de trechter zit, en laat los.", "",
          f"- {ok(i['van_voren'] is None)} Van voren onder het portaal door: {uitkomst(i['van_voren'])}. Tussen de "
          f"vingers en de dwarsbalk: {str(i['vingers_tot_portaal_mm']).replace('.', ',')} mm.",
          f"- {ok(i['zakken'] is None)} Laten zakken in de trechter: {uitkomst(i['zakken'])}.",
          f"- {ok(i['munt_valt'] is None)} Daarna valt het muntje alleen tot op de kantelplank: "
          f"{uitkomst(i['munt_valt'])}."]
    return L


def main():
    r = montage()
    print("\n".join(regels(r)))


if __name__ == "__main__":
    main()
