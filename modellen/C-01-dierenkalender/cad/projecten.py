"""Bambu Studio-projecten (.3mf) voor C-01: per dier en taal een complete set, plus gedeelde projecten.

Elke variant (b.v. varianten/01-vos/nl/dierenkalender_vos_nl.3mf) heeft vijf printplaten:
  1 front (4 kleuren)  2 voet  3 kaartjes maanden  4 kaartjes dagen  5 kaartjes cijfers
Gedeeld: voet, kaartjes per taal (3 platen) en de pasproef per taal (1 plaat).

Instellingen (bovenop "0.20mm Standard @BBL A1", Textured PEI Plate) staan per object, zodat Bambu Studio
ze bij het openen niet terugzet naar het systeemprofiel.
"""
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

import kalender as K  # noqa: E402
from bambu import project as B  # noqa: E402
from dieren import ALLE, bouwbaar  # noqa: E402
from geom import cs_to_shapely, to_trimesh  # noqa: E402
from params import FILAMENTEN, KAART_WIT, KAART_ZWART, ORANJE, kort  # noqa: E402
from talen import TALEN  # noqa: E402

# Instellingen per object. Uitgangspunt: Bambu "0.20mm Standard @BBL A1" (2 wanden, 15 % grid, 5 top/3 bodem).
FRONT = {"wall_loops": "3",                 # stevige randen langs het silhouet en de geleiders
         "sparse_infill_density": "20%",    # de 3 mm plaat blijft vlak en stijf
         "brim_type": "no_brim",            # groot vlak onderdeel op Textured PEI: hecht goed, strakke rand
         "flush_into_infill": "1"}          # spoelen in de (onzichtbare) infill: kleinere spoeltoren
VOET = {"wall_loops": "3", "brim_type": "no_brim"}   # 3 wanden rond de sleuf en de bakken
KAART = {"brim_type": "no_brim"}                     # 1,6 mm, volledig massief (3 bodem + 5 top)
PROEF = {"wall_loops": "3", "brim_type": "no_brim"}


def delen(kleurdelen, slots, naam, taal):
    """{filament: Manifold} -> [Deel] met het slotnummer van dat filament."""
    out = []
    for code, m in kleurdelen.items():
        tm = to_trimesh(m.simplify(1e-3))
        out.append(B.Deel(f"{naam}_{kort(code, taal)}", tm.vertices.copy(), tm.faces.copy(), slots[code]))
    return out


def object_(naam, kleurdelen, slots, taal, instellingen, positie):
    return B.Object(naam, delen(kleurdelen, slots, naam, taal), dict(instellingen), positie)


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


def filamentlijst(codes):
    return [B.Filament(FILAMENTEN[c][2], FILAMENTEN[c][1], f"{FILAMENTEN[c][0]} {c}") for c in codes]


def kaartplaten(taal, slots):
    t = TALEN[taal]
    p = t["plaat"]
    return [
        B.Plaat(p["maanden"], [object_(f"{t['kaartjes']}_{t['maanden_bestand']}", K.kaartjes(taal, "maanden"),
                                       slots, taal, KAART, (128, 70))], toren=(110, 150)),
        B.Plaat(p["dagen"], [object_(f"{t['kaartjes']}_{t['dagen_bestand']}", K.kaartjes(taal, "dagen"),
                                     slots, taal, KAART, (128, 40))], toren=(110, 110)),
        B.Plaat(p["cijfers"], [object_(f"{t['kaartjes']}_{t['cijfers_bestand']}", K.kaartjes(taal, "cijfers"),
                                       slots, taal, KAART, (95, 40))], toren=(205, 25)),
    ]


def met_miniaturen(platen, filamenten):
    for p in platen:
        p.miniatuur = miniatuur(p, filamenten)
    return platen


def variant(dier, taal):
    t = TALEN[taal]
    fr, _ = K.front(dier.sleutel, taal)
    # vaste volgorde: bordkleur, overige kleuren, dan wit en zwart (de kaartjes) achteraan
    codes = [dier.bord] + [c for c in fr if c not in (dier.bord, KAART_WIT, KAART_ZWART)]
    codes += [c for c in (KAART_WIT, KAART_ZWART) if c not in codes]
    slots = {c: i + 1 for i, c in enumerate(codes)}
    filamenten = filamentlijst(codes)
    naam = dier.naam[taal]
    platen = [
        B.Plaat(t["plaat"]["front"].format(dier=naam),
                [object_(f"{t['front']}_{naam}", fr, slots, taal, FRONT, (103, 110))], toren=(215, 30)),
        B.Plaat(t["plaat"]["voet"], [object_(t["voet"], {dier.bord: K.foot()["voet"]}, slots, taal, VOET, (128, 60))]),
    ] + kaartplaten(taal, slots)
    met_miniaturen(platen, filamenten)
    from bouw import bestandsnaam, variant_map
    pad = variant_map(dier, taal) / f"{bestandsnaam(dier, taal)}.3mf"
    beschrijving = f"C-01 {t['project']} – {naam} ({taal.upper()})"
    if dier.fanart:
        beschrijving += (" – fan-art, alleen voor eigen gebruik; niet publiceren of verkopen"
                         if taal == "nl" else " – fan art, personal use only; do not publish or sell")
    info = B.schrijf(pad, f"C-01 {t['project']} {naam}", filamenten, platen, beschrijving=beschrijving)
    return pad, info, codes


def gedeeld():
    from bouw import GEDEELD
    out = []
    # voet: één filament, kies zelf de kleur in Bambu Studio
    fil = filamentlijst([ORANJE])
    platen = met_miniaturen([B.Plaat("Voet / base", [object_("voet", {ORANJE: K.foot()["voet"]}, {ORANJE: 1}, "nl",
                                                              VOET, (128, 60))])], fil)
    out.append((GEDEELD / "voet" / "voet.3mf", B.schrijf(GEDEELD / "voet" / "voet.3mf", "C-01 voet / base", fil, platen)))
    for taal, t in TALEN.items():
        codes = [KAART_WIT, KAART_ZWART]
        slots = {c: i + 1 for i, c in enumerate(codes)}
        fil = filamentlijst(codes)
        platen = met_miniaturen(kaartplaten(taal, slots), fil)
        pad = GEDEELD / "kaartjes" / taal / f"{t['kaartjes']}_{taal}.3mf"
        out.append((pad, B.schrijf(pad, f"C-01 {t['kaartjes']} {taal.upper()}", fil, platen)))
        # pasproef: kanaalstrook, sleuvenblok en vier proefkaartjes op één plaat
        codes = [ORANJE, KAART_WIT, KAART_ZWART]
        slots = {c: i + 1 for i, c in enumerate(codes)}
        fil = filamentlijst(codes)
        objs = [object_("kanalen", {ORANJE: K.proef_kanalen()["proef"]}, slots, taal, PROEF, (50, 40)),
                object_("sleuven", {ORANJE: K.proef_sleuven()["proef"]}, slots, taal, PROEF, (180, 25)),
                object_(t["pasproef"] + "_kaartjes", K.proef_kaartjes(taal), slots, taal, KAART, (80, 100))]
        platen = met_miniaturen([B.Plaat(t["plaat"]["pasproef"], objs, toren=(190, 150))], fil)
        pad = GEDEELD / "pasproef" / taal / f"{t['pasproef']}_{taal}.3mf"
        out.append((pad, B.schrijf(pad, f"C-01 {t['plaat']['pasproef']} {taal.upper()}", fil, platen)))
    return out


def main(dieren=None):
    print("Bambu Studio-projecten")
    for pad, info in gedeeld():
        print(f"  {pad.name:34s} {info}")
    for dier in dieren or bouwbaar(ALLE):
        for taal in TALEN:
            pad, info, codes = variant(dier, taal)
            print(f"  {pad.name:34s} {info}  filamenten: {', '.join(FILAMENTEN[c][0] for c in codes)}")


if __name__ == "__main__":
    main()
