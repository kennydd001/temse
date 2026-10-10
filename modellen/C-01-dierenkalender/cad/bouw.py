"""Schrijft alle printbestanden van C-01 in de mappenstructuur van het model.

Gebruik:  python bouw.py                 alles (alle dieren, NL + EN, STL + 3MF)
          python bouw.py vos koe         alleen deze dieren (plus de gedeelde onderdelen)
          python bouw.py --geen-3mf      alleen STL's

Mappen (vanaf modellen/C-01-dierenkalender):
  gedeeld/voet/                 voet (zelfde voor elk dier; kleur kies je in de slicer)
  gedeeld/kaartjes/             cijfers (alle talen), nl/ en en/ voor dagen en maanden
  gedeeld/pasproef/             kanaal- en sleufproef, nl/ en en/ voor de proefkaartjes
  varianten/NN-dier/nl|en/      dierenfront per kleur (STL) en het complete Bambu-project (3MF)
  fanart/NN-naam/nl|en/         idem voor bestaande personages: alleen voor eigen gebruik
"""
import json
import sys
from pathlib import Path

import trimesh

import kalender as K
from dieren import ALLE, PER_SLEUTEL, bouwbaar
import _pad  # noqa: F401,E402  (maakt gereedschap/ importeerbaar)
from gereedschap.geom import save_stl
from params import kort
from talen import TALEN

ROOT = Path(__file__).resolve().parent.parent
GEDEELD = ROOT / "gedeeld"
VARIANTEN = ROOT / "varianten"
FANART = ROOT / "fanart"   # bestaande personages: alleen voor eigen gebruik, niet publiceren
INDEX = ROOT / "onderdelen.json"


def variant_map(dier, taal):
    return (FANART if dier.fanart else VARIANTEN) / f"{dier.nr:02d}-{dier.sleutel}" / taal


def bestandsnaam(dier, taal):
    t = TALEN[taal]
    return f"{t['project'].lower().replace(' ', '-')}_{dier.naam[taal]}_{taal}"


def schrijf(naam, delen, folder, taal, info):
    """delen: {filamentcode of rolnaam: Manifold}. Meerdere kleuren -> naam__kleur.stl."""
    files = {}
    for key, m in delen.items():
        if m.is_empty():
            continue
        suffix = f"__{kort(key, taal)}" if key.isdigit() and len(delen) > 1 else ""
        path = folder / f"{naam}{suffix}.stl"
        save_stl(m, path)
        tm = trimesh.load(path)  # opnieuw inlezen zoals een slicer dat doet
        assert tm.is_watertight and tm.is_winding_consistent, path
        files[key] = {"bestand": str(path.relative_to(ROOT)), "volume_cm3": round(float(tm.volume) / 1000, 2),
                      "maat_mm": [round(float(v), 1) for v in tm.extents]}
    info[str(Path(folder.relative_to(ROOT)) / naam)] = files
    print(f"  {str(folder.relative_to(ROOT)) + '/' + naam:52s}", ", ".join(
        f"{kort(k, taal) if k.isdigit() else k} {v['maat_mm']}" for k, v in files.items()))
    return files


def gedeeld(info):
    print("gedeelde onderdelen")
    schrijf("voet", K.foot(), GEDEELD / "voet", "nl", info)
    schrijf("cijfers", K.kaartjes("nl", "cijfers"), GEDEELD / "kaartjes", "nl", info)
    for taal, t in TALEN.items():
        for soort in ("maanden", "dagen"):
            schrijf(t[f"{soort}_bestand"], K.kaartjes(taal, soort), GEDEELD / "kaartjes" / taal, taal, info)
        schrijf("proefkaartjes" if taal == "nl" else "test-cards", K.proef_kaartjes(taal),
                GEDEELD / "pasproef" / taal, taal, info)
    schrijf("kanalen", K.proef_kanalen(), GEDEELD / "pasproef", "nl", info)
    schrijf("sleuven", K.proef_sleuven(), GEDEELD / "pasproef", "nl", info)


def varianten(dieren, info):
    for dier in dieren:
        for taal in TALEN:
            print(f"{dier.naam['nl']} / {taal}")
            fr, _ = K.front(dier.sleutel, taal)
            schrijf(f"front_{dier.naam[taal]}", fr, variant_map(dier, taal), taal, info)


def main(argv):
    namen = [a for a in argv if not a.startswith("--")]
    dieren = bouwbaar([PER_SLEUTEL[n] for n in namen] if namen else ALLE)
    info = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    gedeeld(info)
    varianten(dieren, info)
    INDEX.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    if "--geen-3mf" not in argv:
        import projecten
        projecten.main(dieren)


if __name__ == "__main__":
    main(sys.argv[1:])
