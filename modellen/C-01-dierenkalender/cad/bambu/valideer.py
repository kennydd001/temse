"""Controleert een Bambu Studio-project (.3mf) op de regels die de lezer van Bambu Studio hanteert.

Gebruik:  python -m bambu.valideer pad/naar/project.3mf [...]

Gecontroleerd (zie bbs_3mf.cpp / PresetBundle.cpp):
  - verplichte bestanden, geldige XML/JSON, Application = BambuStudio-<versie>
  - componenten verwijzen naar bestaande objecten in hun sub-model
  - model_settings: object- en part-id's kloppen met 3dmodel.model, filamentslots 1..N
  - project_settings: alleen strings, lijsten per filament hebben lengte N, spoelmatrix N x N,
    spoeltoren-posities per plaat
  - elk object ligt volledig op de plaat waar het hoort (Bambu kiest de plaat op positie),
    staat op z = 0 en overlapt de spoeltoren niet
"""
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .project import BED, plaat_oorsprong

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
      "p": "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"}
P_PATH = "{%s}path" % NS["p"]
SJABLOON = Path(__file__).resolve().parent / "profielen" / "project_sjabloon_A1.json"


def _matrix(t):
    v = [float(x) for x in t.split()]
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = v[0:3], v[3:6], v[6:9], v[9:12]
    return m


def valideer(pad):
    fouten, z = [], zipfile.ZipFile(pad)
    namen = set(z.namelist())
    for vereist in ("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "Metadata/model_settings.config",
                    "Metadata/project_settings.config"):
        if vereist not in namen:
            fouten.append(f"ontbreekt: {vereist}")
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    app = [m.text for m in root.findall("m:metadata", NS) if m.get("name") == "Application"]
    if not app or not (app[0] or "").startswith("BambuStudio-"):
        fouten.append("Application-metadata begint niet met BambuStudio-")

    # sub-modellen en hun meshes
    subs = {}
    for naam in namen:
        if naam.startswith("3D/Objects/") and naam.endswith(".model"):
            sm = ET.fromstring(z.read(naam))
            for o in sm.findall("m:resources/m:object", NS):
                v = np.array([[float(x.get(a)) for a in "xyz"] for x in o.findall("m:mesh/m:vertices/m:vertex", NS)])
                subs[("/" + naam, o.get("id"))] = v

    objecten = {}
    for o in root.findall("m:resources/m:object", NS):
        comps = []
        for c in o.findall("m:components/m:component", NS):
            key = (c.get(P_PATH), c.get("objectid"))
            if key not in subs:
                fouten.append(f"component verwijst naar onbekend object {key}")
                continue
            comps.append((c.get("objectid"), _matrix(c.get("transform")), subs[key]))
        objecten[o.get("id")] = comps
    items = {}
    for it in root.findall("m:build/m:item", NS):
        if it.get("objectid") not in objecten:
            fouten.append(f"build-item verwijst naar onbekend object {it.get('objectid')}")
        items[it.get("objectid")] = _matrix(it.get("transform"))

    proj = json.loads(z.read("Metadata/project_settings.config"))
    for k, v in proj.items():
        ok = isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v))
        if not ok:
            fouten.append(f"project_settings[{k}] is geen string of lijst van strings")
    n = len(proj.get("filament_colour", []))
    if n == 0:
        fouten.append("filament_colour is leeg")
    sjabloon = json.loads(SJABLOON.read_text())
    n_sj = len(sjabloon["filament_colour"])
    per_fil = [k for k, v in sjabloon.items() if isinstance(v, list) and len(v) == n_sj]
    for k in per_fil:
        if k in proj and len(proj[k]) != n:
            fouten.append(f"{k} heeft {len(proj[k])} waarden, verwacht {n}")
    if proj.get("filament_self_index") != [str(i + 1) for i in range(n)]:
        fouten.append("filament_self_index klopt niet")
    if len(proj.get("flush_volumes_matrix", [])) != n * n:
        fouten.append("flush_volumes_matrix is niet N x N")
    if len(proj.get("different_settings_to_system", [])) != n + 2:
        fouten.append("different_settings_to_system heeft niet N+2 waarden")
    if len(proj.get("filament_extruder_variant", [])) != n:
        fouten.append("filament_extruder_variant heeft niet N waarden")

    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    for o in cfg.findall("object"):
        oid = o.get("id")
        if oid not in objecten:
            fouten.append(f"model_settings-object {oid} bestaat niet in 3dmodel.model")
            continue
        part_ids = [p.get("id") for p in o.findall("part")]
        comp_ids = [c[0] for c in objecten[oid]]
        if part_ids != comp_ids:
            fouten.append(f"object {oid}: parts {part_ids} komen niet overeen met componenten {comp_ids}")
        for p in o.findall("part"):
            ext = [m.get("value") for m in p.findall("metadata") if m.get("key") == "extruder"]
            if ext and not 1 <= int(ext[0]) <= n:
                fouten.append(f"object {oid} part {p.get('id')}: filamentslot {ext[0]} buiten 1..{n}")

    platen = cfg.findall("plate")
    tx, ty = proj.get("wipe_tower_x", []), proj.get("wipe_tower_y", [])
    if len(tx) != len(platen) or len(ty) != len(platen):
        fouten.append("wipe_tower_x/y hebben niet één waarde per plaat")
    samenvatting = []
    for k, pl in enumerate(platen):
        ox, oy = plaat_oorsprong(k, len(platen))
        ids = [i.find("metadata[@key='object_id']").get("value") for i in pl.findall("model_instance")]
        slots = set()
        for oid in ids:
            M = items[oid]
            pts = np.vstack([(M @ C @ np.c_[v, np.ones(len(v))].T).T[:, :3] for _, C, v in objecten[oid]])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            lokaal_lo, lokaal_hi = lo[:2] - (ox, oy), hi[:2] - (ox, oy)
            if (lokaal_lo < -0.01).any() or (lokaal_hi > BED + 0.01).any():
                fouten.append(f"plaat {k + 1}, object {oid}: ligt (deels) buiten de plaat {lokaal_lo.round(1)}..{lokaal_hi.round(1)}")
            if abs(lo[2]) > 0.01:
                fouten.append(f"plaat {k + 1}, object {oid}: onderkant op z = {lo[2]:.3f} in plaats van 0")
            o = cfg.find(f"object[@id='{oid}']")
            obj_ext = o.find("metadata[@key='extruder']")
            for p in o.findall("part"):
                ext = p.find("metadata[@key='extruder']")   # zonder eigen slot erft een deel dat van het object
                slots.add((ext if ext is not None else obj_ext).get("value") if (ext is not None or obj_ext is not None) else "1")
            samenvatting.append((k + 1, oid, lokaal_lo.round(1).tolist(), lokaal_hi.round(1).tolist()))
            if len(tx) > k:
                t0 = np.array([float(tx[k]), float(ty[k])])
                if not ((lokaal_hi < t0).any() or (lokaal_lo > t0 + 35).any()):
                    if len(slots) > 1:
                        fouten.append(f"plaat {k + 1}: spoeltoren op {t0} overlapt object {oid}")
        if len(slots) > 4:
            fouten.append(f"plaat {k + 1}: {len(slots)} filamenten, AMS Lite heeft er 4")
    return fouten, samenvatting


def main(paden):
    totaal = 0
    for pad in paden:
        fouten, _ = valideer(pad)
        totaal += len(fouten)
        print(f"{'OK  ' if not fouten else 'FOUT'} {pad}")
        for f in fouten:
            print("     -", f)
    return totaal


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
