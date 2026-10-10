"""Schrijft een bestaand project (.3mf van project.schrijf) opnieuw met de huidige regels, zonder de CAD-bron.

Voor projecten waarvan de bron hier niet draait, zoals de fan-art van C-01 (de bronafbeeldingen staan
niet in de repo). De meshes, objecten, delen, instellingen per object en platen blijven gelijk; de
projectinstellingen, de spoeltorens en de miniaturen worden opnieuw gemaakt.

Gebruik (vanuit de root van de repo):  python -m gereedschap.bambu.herschrijf pad/naar/project.3mf [...]
"""
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import project as B
from .miniatuur import miniatuur
from .valideer import NS, P_PATH

PRESET = {v[0]: k for k, v in B.FILAMENT_PRESET.items()}


def lees(pad):
    """{titel, filamenten, platen, ontwerper, beschrijving} zoals project.schrijf ze verwacht."""
    z = zipfile.ZipFile(pad)
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    meta = {m.get("name"): m.text or "" for m in root.findall("m:metadata", NS)}
    proj = json.loads(z.read("Metadata/project_settings.config"))
    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    filamenten = [B.Filament(kleur, PRESET[preset]) for kleur, preset in
                  zip(proj["filament_colour"], proj["filament_settings_id"])]

    meshes = {}
    for naam in z.namelist():
        if naam.startswith("3D/Objects/") and naam.endswith(".model"):
            for o in ET.fromstring(z.read(naam)).findall("m:resources/m:object", NS):
                v = np.array([[float(x.get(a)) for a in "xyz"] for x in o.findall("m:mesh/m:vertices/m:vertex", NS)])
                f = np.array([[int(t.get(a)) for a in ("v1", "v2", "v3")]
                              for t in o.findall("m:mesh/m:triangles/m:triangle", NS)])
                meshes[("/" + naam, o.get("id"))] = (v, f)
    componenten = {o.get("id"): [meshes[(c.get(P_PATH), c.get("objectid"))]
                                 for c in o.findall("m:components/m:component", NS)]
                   for o in root.findall("m:resources/m:object", NS)}
    plek = {it.get("objectid"): [float(x) for x in it.get("transform").split()[9:11]]
            for it in root.findall("m:build/m:item", NS)}

    platen = cfg.findall("plate")
    uit = []
    for k, pl in enumerate(platen):
        ox, oy = B.plaat_oorsprong(k, len(platen))
        objecten = []
        for inst in pl.findall("model_instance"):
            oid = inst.find("metadata[@key='object_id']").get("value")
            o = cfg.find(f"object[@id='{oid}']")
            md = {m.get("key"): m.get("value") for m in o.findall("metadata")}
            delen = []
            for part, (v, f) in zip(o.findall("part"), componenten[oid]):
                pm = {m.get("key"): m.get("value") for m in part.findall("metadata")}
                delen.append(B.Deel(pm["name"], v, f, int(pm.get("extruder", md.get("extruder", "1")))))
            instellingen = {kk: vv for kk, vv in md.items() if kk not in ("name", "extruder")}
            objecten.append(B.Object(md["name"], delen, instellingen, (plek[oid][0] - ox, plek[oid][1] - oy)))
        uit.append(B.Plaat(pl.find("metadata[@key='plater_name']").get("value"), objecten))
    return {"titel": meta.get("Title", ""), "filamenten": filamenten, "platen": uit,
            "ontwerper": meta.get("Designer", ""), "beschrijving": meta.get("Description", "")}


def herschrijf(pad):
    p = lees(pad)
    B.plaats_torens(p["filamenten"], p["platen"])
    for plaat in p["platen"]:
        plaat.miniatuur = miniatuur(plaat, p["filamenten"])
    return B.schrijf(pad, p["titel"], p["filamenten"], p["platen"], ontwerper=p["ontwerper"],
                     beschrijving=p["beschrijving"])


if __name__ == "__main__":
    for pad in sys.argv[1:]:
        print(pad, herschrijf(pad))
