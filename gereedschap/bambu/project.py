"""Schrijft een Bambu Studio-project (.3mf) voor de Bambu Lab A1 (0.4 mm) met AMS Lite.

Een project heeft meerdere printplaten; elke plaat bevat objecten, elk object bestaat uit delen
(één per kleur) die elk een filamentslot krijgen. Instellingen per object (wanden, infill, brim)
staan in model_settings.config, zodat Bambu Studio ze niet terugzet naar het systeemprofiel.

Het formaat volgt de lees-/schrijfcode van Bambu Studio (src/libslic3r/Format/bbs_3mf.cpp):
  - plaat k (0-based) ligt op (col * 307.2, -row * 307.2) met cols = compute_colum_count(N);
    Bambu Studio kiest de plaat van een object op basis van zijn positie
  - project_settings.config: alle waarden als JSON-strings; lijsten per filament hebben lengte N
  - wipe_tower_x/y per plaat: de spoeltoren van Bambu Studio 2.x is groter dan prime_tower_width;
    zonder vaste plek zoekt plaats_torens() er een die Bambu zelf ook goedkeurt (zie toren.py)
"""
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from . import flushvol, toren as T

HIER = Path(__file__).resolve().parent
PROFIELEN = HIER / "profielen"
APP_VERSIE = "02.00.02.01"     # zelfde als Bambu's eigen A1-kalibratieproject; nieuwere versies lezen dit gewoon
BED = 256.0
STRIDE = BED * 1.2             # LOGICAL_PART_PLATE_GAP = 1/5

PRINTER = "Bambu Lab A1 0.4 nozzle"
PROCES = "0.20mm Standard @BBL A1"
FILAMENT_PRESET = {"PLA Basic": ("Bambu PLA Basic @BBL A1", "pla_basic_A1.json"),
                   "PLA Matte": ("Bambu PLA Matte @BBL A1", "pla_matte_A1.json")}


@dataclass
class Deel:
    naam: str
    vertices: np.ndarray       # (n, 3) in mm
    faces: np.ndarray          # (m, 3)
    slot: int                  # 1-based filamentslot


@dataclass
class Object:
    naam: str
    delen: list
    instellingen: dict = field(default_factory=dict)   # per-object overrides, b.v. {"wall_loops": "3"}
    positie: tuple = (128.0, 128.0)                     # midden (x, y) op de plaat in mm


@dataclass
class Plaat:
    naam: str
    objecten: list
    toren: tuple = None                                 # spoeltoren linksonder (x, y); None: plaats_torens() kiest
    miniatuur: bytes = b""                              # PNG
    toren_maat: tuple = None                            # (breedte, diepte) met ribben, ingevuld door plaats_torens()


@dataclass
class Filament:
    kleur: str                 # "#RRGGBB"
    materiaal: str             # "PLA Basic" of "PLA Matte"
    naam: str = ""             # vrije naam, alleen voor de documentatie


def compute_colum_count(n):
    v = math.sqrt(n)
    r = round(v)
    return r + 1 if v > r else r


def plaat_oorsprong(k, n):
    cols = compute_colum_count(n)
    return (k % cols) * STRIDE, -(k // cols) * STRIDE


def _laad(naam):
    return json.loads((PROFIELEN / naam).read_text())


def _s(v):
    """Alles als string (of lijst van strings), zoals Bambu Studio het schrijft."""
    if isinstance(v, (list, tuple)):
        return [_s(x) for x in v]
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def project_instellingen(filamenten, torens=()):
    """Volledige project_settings.config op basis van het A1-sjabloon en de systeemprofielen.

    torens: linkeronderhoek (x, y) van de spoeltoren per plaat."""
    n = len(filamenten)
    sjabloon = _laad("project_sjabloon_A1.json")
    n_sjabloon = len(sjabloon["filament_colour"])
    machine = _laad("machine_A1_0.4.json")
    proces = _laad("proces_0.20_standard_A1.json")
    meta = {"name", "inherits", "from", "setting_id", "type", "instantiation", "description", "_flatten_meta",
            "filament_id", "include", "version"}

    cfg = {k: v for k, v in sjabloon.items()}
    for prof in (machine, proces):
        cfg.update({k: v for k, v in prof.items() if k not in meta})

    # lijsten per filament: uit het profiel van elk slot
    profielen = [_laad(FILAMENT_PRESET[f.materiaal][1]) for f in filamenten]
    per_filament = set(k for k, v in sjabloon.items() if isinstance(v, list) and len(v) == n_sjabloon)
    per_filament |= set(k for k in profielen[0] if isinstance(profielen[0][k], list) and k not in meta)
    per_filament -= {"flush_volumes_matrix", "flush_volumes_vector", "different_settings_to_system"}
    for k in per_filament:
        waarden = []
        for p in profielen:
            if k in p and isinstance(p[k], list) and p[k]:
                waarden.append(p[k][0])
            elif k in sjabloon and isinstance(sjabloon[k], list) and sjabloon[k]:
                waarden.append(sjabloon[k][0])
        if len(waarden) == n:
            cfg[k] = waarden

    kleuren = [f.kleur.upper() for f in filamenten]
    ids = [_laad(FILAMENT_PRESET[f.materiaal][1])["filament_id"] for f in filamenten]
    matrix = flushvol.flush_volumes_matrix(kleuren, filament_ids=ids)
    cfg.update({
        "name": "project_settings", "from": "project", "version": APP_VERSIE,
        "printer_settings_id": PRINTER, "print_settings_id": PROCES,
        "filament_settings_id": [FILAMENT_PRESET[f.materiaal][0] for f in filamenten],
        "filament_colour": kleuren,
        "default_filament_colour": [""] * n,
        "filament_ids": ids,
        "filament_self_index": [str(i + 1) for i in range(n)],
        "filament_map": ["1"] * n,
        "filament_extruder_variant": ["Direct Drive Standard"] * n,
        "filament_type": ["PLA"] * n,
        "filament_vendor": ["Bambu Lab"] * n,
        "flush_volumes_matrix": [str(v) for v in matrix],
        "flush_volumes_vector": ["140"] * (2 * n),
        "flush_multiplier": ["1"],
        "different_settings_to_system": [""] * (n + 2),
        "curr_bed_type": "Textured PEI Plate",
        "filament_map_mode": "Auto For Flush",
        "print_compatible_printers": [PRINTER],
        "printer_model": "Bambu Lab A1",
        "printer_variant": "0.4",
        "enable_prime_tower": "1",
        "wipe_tower_x": [_s(round(x, 2)) for x, _ in torens],
        "wipe_tower_y": [_s(round(y, 2)) for _, y in torens],
    })
    for k in ("inherits_group", "compatible_machine_expression_group", "compatible_process_expression_group"):
        cfg.pop(k, None)
    return {k: _s(v) for k, v in sorted(cfg.items())}


def _uuid(n, suffix):
    return f"{n:08x}{suffix}"


def _midden(obj):
    """Midden van de voetafdruk en onderkant van een object: dat punt komt op obj.positie, z = 0."""
    alle = np.vstack([d.vertices for d in obj.delen])
    lo, hi = alle.min(axis=0), alle.max(axis=0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def plaats_torens(filamenten, platen):
    """Vult plaat.toren (waar die None is) en plaat.toren_maat in; een vaste plek wordt gecontroleerd.

    Ook een plaat met één filament krijgt een veilige plek: daar komt geen toren, maar wie objecten
    tussen platen verschuift (b.v. met Rangschikken), krijgt de toren van die plaat. Is zo'n plaat te vol
    voor een vrije plek, dan ligt de toren alleen binnen de plaat en het gebied van Bambu.
    """
    p = T.parameters(project_instellingen(filamenten))
    for k, plaat in enumerate(platen):
        delen = [(d.slot, d.vertices - m + (obj.positie[0], obj.positie[1], 0), d.faces)
                 for obj in plaat.objecten for m in [_midden(obj)] for d in obj.delen]
        slots = {s for s, _, _ in delen}
        hoogte = max((float(v[:, 2].max()) for _, v, _ in delen), default=0.0)
        obst = T.obstakels(delen) if delen else None
        if len(slots) > 1:
            n, per_laag = len(slots), T.filamenten_per_laag(delen)
        else:
            n = per_laag = min(max(len(filamenten), 2), 4)
        if plaat.toren is None:
            plaat.toren = T.plaats(p, n, per_laag, hoogte, obst)
            if plaat.toren is None and len(slots) <= 1:   # plaat vol en één kleur: hier komt geen toren
                plaat.toren = T.plaats(p, n, per_laag, hoogte)
            if plaat.toren is None:
                raise ValueError(f"plaat {k + 1} ({plaat.naam}): geen plek voor de spoeltoren")
        fouten = T.controleer(p, *plaat.toren, n, per_laag, hoogte, obst if len(slots) > 1 else None)
        if fouten:
            raise ValueError(f"plaat {k + 1} ({plaat.naam}): " + "; ".join(fouten))
        plaat.toren_maat = T.wand(p, per_laag, hoogte)
    return platen


def schrijf(pad, titel, filamenten, platen, ontwerper="", beschrijving=""):
    """Schrijft het project naar `pad`. Geeft een samenvatting terug (aantal objecten, delen)."""
    pad = Path(pad)
    pad.parent.mkdir(parents=True, exist_ok=True)
    plaats_torens(filamenten, platen)
    teller = 0
    objecten = []          # (root_id, Object, [(part_id, Deel, vertices_lokaal)], wereldpositie, plaatindex)
    for k, plaat in enumerate(platen):
        ox, oy = plaat_oorsprong(k, len(platen))
        for obj in plaat.objecten:
            alle = np.vstack([d.vertices for d in obj.delen])
            lo, hi = alle.min(axis=0), alle.max(axis=0)
            midden = _midden(obj)
            delen = []
            for d in sorted(obj.delen, key=lambda d: d.slot):
                teller += 1
                delen.append((teller, d, d.vertices - midden))
            teller += 1
            wereld = (ox + obj.positie[0], oy + obj.positie[1], 0.0)
            objecten.append((teller, obj, delen, wereld, k, (hi - lo)))

    vandaag = date.today().isoformat()
    meta = {"Application": f"BambuStudio-{APP_VERSIE}", "BambuStudio:3mfVersion": "1", "Copyright": "",
            "CreationDate": vandaag, "Description": beschrijving, "Designer": ontwerper, "DesignerCover": "",
            "DesignerUserId": "", "License": "", "ModificationDate": vandaag, "Origin": "", "Title": titel}
    if platen and platen[0].miniatuur:
        meta["Thumbnail_Middle"] = "/Metadata/plate_1.png"
        meta["Thumbnail_Small"] = "/Metadata/plate_1.png"
    ns = ('unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
          'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
          'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"')

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))

    bestanden = {}

    class _Zip:  # verzamelt eerst, schrijft daarna in de volgorde die Bambu Studio zelf gebruikt
        def writestr(self, naam, data):
            bestanden[naam] = data

    if True:
        z = _Zip()
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                   ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                   ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                   ' <Default Extension="png" ContentType="image/png"/>\n'
                   ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n'
                   '</Types>')
        rels = ['<?xml version="1.0" encoding="UTF-8"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
                'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>']
        if platen and platen[0].miniatuur:
            rels += [' <Relationship Target="/Metadata/plate_1.png" Id="rel-2" '
                     'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>',
                     ' <Relationship Target="/Metadata/plate_1.png" Id="rel-4" '
                     'Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>',
                     ' <Relationship Target="/Metadata/plate_1.png" Id="rel-5" '
                     'Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>']
        z.writestr("_rels/.rels", "\n".join(rels + ["</Relationships>"]))

        # één sub-model per object, met een mesh per deel
        model_rels = []
        for i, (root, obj, delen, _, _, _) in enumerate(objecten, start=1):
            sub = f"3D/Objects/object_{i}.model"
            model_rels.append(f' <Relationship Target="/{sub}" Id="rel-{i}" '
                              'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>')
            buf = io.StringIO()
            buf.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<model {ns}>\n'
                      ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n')
            for j, (pid, d, v) in enumerate(delen):
                buf.write(f'  <object id="{pid}" p:UUID="{(i << 16 | j):08x}-81cb-4c03-9d28-80fed5dfa1dc" type="model">\n'
                          '   <mesh>\n    <vertices>\n')
                buf.write("".join(f'     <vertex x="{x:.9g}" y="{y:.9g}" z="{zz:.9g}"/>\n' for x, y, zz in v))
                buf.write('    </vertices>\n    <triangles>\n')
                buf.write("".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in d.faces))
                buf.write('    </triangles>\n   </mesh>\n  </object>\n')
            buf.write(' </resources>\n <build/>\n</model>\n')
            z.writestr(sub, buf.getvalue())
        z.writestr("3D/_rels/3dmodel.model.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                   + "\n".join(model_rels) + "\n</Relationships>")

        root_xml = [f'<?xml version="1.0" encoding="UTF-8"?>\n<model {ns}>']
        root_xml += [f' <metadata name="{k}">{esc(v)}</metadata>' for k, v in meta.items()]
        root_xml.append(" <resources>")
        for i, (root, obj, delen, _, _, _) in enumerate(objecten, start=1):
            root_xml.append(f'  <object id="{root}" p:UUID="{i:08x}-61cb-4c03-9d28-80fed5dfa1dc" type="model">')
            root_xml.append("   <components>")
            for j, (pid, d, v) in enumerate(delen):
                root_xml.append(f'    <component p:path="/3D/Objects/object_{i}.model" objectid="{pid}" '
                                f'p:UUID="{(i << 16 | j):08x}-b206-40ff-9872-83e8017abed1" '
                                'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
            root_xml += ["   </components>", "  </object>"]
        root_xml.append(" </resources>")
        root_xml.append(' <build p:UUID="2c7c17d8-22b5-4d84-8835-1976022ea369">')
        for root, obj, delen, (wx, wy, wz), _, _ in objecten:
            root_xml.append(f'  <item objectid="{root}" p:UUID="{root:08x}-b1ec-4553-aec9-835e5b724bb4" '
                            f'transform="1 0 0 0 1 0 0 0 1 {wx:.6f} {wy:.6f} {wz:.6f}" printable="1"/>')
        root_xml += [" </build>", "</model>"]
        z.writestr("3D/3dmodel.model", "\n".join(root_xml))

        # model_settings.config: objecten, delen met filamentslot, platen
        cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
        for root, obj, delen, _, _, _ in objecten:
            cfg.append(f'  <object id="{root}">')
            cfg.append(f'    <metadata key="name" value="{esc(obj.naam)}"/>')
            cfg.append(f'    <metadata key="extruder" value="{min(d.slot for _, d, _ in delen)}"/>')
            for k, v in obj.instellingen.items():
                cfg.append(f'    <metadata key="{k}" value="{esc(v)}"/>')
            for pid, d, v in delen:
                cfg += [f'    <part id="{pid}" subtype="normal_part">',
                        f'      <metadata key="name" value="{esc(d.naam)}"/>',
                        '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>',
                        f'      <metadata key="extruder" value="{d.slot}"/>',
                        f'      <mesh_stat face_count="{len(d.faces)}" edges_fixed="0" degenerate_facets="0" '
                        'facets_removed="0" facets_reversed="0" backwards_edges="0"/>',
                        "    </part>"]
            cfg.append("  </object>")
        ident = 100
        for k, plaat in enumerate(platen):
            cfg += ["  <plate>", f'    <metadata key="plater_id" value="{k + 1}"/>',
                    f'    <metadata key="plater_name" value="{esc(plaat.naam)}"/>',
                    '    <metadata key="locked" value="false"/>',
                    '    <metadata key="filament_map_mode" value="Auto For Flush"/>']
            if plaat.miniatuur:
                cfg.append(f'    <metadata key="thumbnail_file" value="Metadata/plate_{k + 1}.png"/>')
            for root, obj, delen, _, pk, _ in objecten:
                if pk != k:
                    continue
                ident += 1
                cfg += ["    <model_instance>", f'      <metadata key="object_id" value="{root}"/>',
                        '      <metadata key="instance_id" value="0"/>',
                        f'      <metadata key="identify_id" value="{ident}"/>', "    </model_instance>"]
            cfg.append("  </plate>")
        cfg.append("</config>")
        z.writestr("Metadata/model_settings.config", "\n".join(cfg) + "\n")

        instellingen = project_instellingen(filamenten, [p.toren for p in platen])
        z.writestr("Metadata/project_settings.config", json.dumps(instellingen, indent=4, ensure_ascii=False))
        z.writestr("Metadata/slice_info.config",
                   '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n'
                   '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
                   f'    <header_item key="X-BBL-Client-Version" value="{APP_VERSIE}"/>\n  </header>\n</config>\n')
        for k, plaat in enumerate(platen):
            if plaat.miniatuur:
                z.writestr(f"Metadata/plate_{k + 1}.png", plaat.miniatuur)

    def rang(naam):
        volgorde = ["[Content_Types].xml", "Metadata/plate_", "3D/3dmodel.model", "3D/_rels/", "3D/Objects/",
                    "Metadata/project_settings.config", "Metadata/model_settings.config",
                    "Metadata/slice_info.config", "_rels/.rels"]
        return next(i for i, v in enumerate(volgorde) if naam.startswith(v)), naam

    with zipfile.ZipFile(pad, "w", zipfile.ZIP_DEFLATED) as zf:
        for naam in sorted(bestanden, key=rang):
            data = bestanden[naam]
            zf.writestr(naam, data, compress_type=zipfile.ZIP_STORED if naam.endswith(".png") else zipfile.ZIP_DEFLATED)
    return {"objecten": len(objecten), "delen": sum(len(o[2]) for o in objecten), "platen": len(platen)}
