"""Slicet elke printplaat met PrusaSlicer en een profiel dat de Bambu Lab A1 benadert.

Gebruik:  python slice.py        (vereist `prusa-slicer` op het pad)
Schrijft  controle/slicer.json   (gewicht, tijd, wissels, support) en controle/a1_benadering.ini.

Bambu Studio zelf draait hier niet. Het profiel neemt de A1-standaard over (0,2 mm laag, 2 wanden,
15 % grid-infill, A1-snelheden), dus het filamentgewicht is betrouwbaar; de printtijd en de
spoelhoeveelheid van de AMS Lite wijken af en zijn alleen een indicatie.
"""
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "controle"
VOLGORDE = ["oranje", "wit", "zwart", "groen"]  # extruder 1..4

A1 = {
    "printer_technology": "FFF", "gcode_flavor": "marlin2", "bed_shape": "0x0,256x0,256x256,0x256",
    "max_print_height": 256, "nozzle_diameter": "0.4", "filament_diameter": "1.75",
    "filament_density": "1.24", "filament_type": "PLA", "temperature": 220, "first_layer_temperature": 220,
    "bed_temperature": 60, "first_layer_bed_temperature": 60,
    "layer_height": 0.2, "first_layer_height": 0.2, "perimeters": 2, "top_solid_layers": 5,
    "bottom_solid_layers": 3, "fill_density": "15%", "fill_pattern": "grid", "infill_every_layers": 1,
    "external_perimeter_speed": 200, "perimeter_speed": 300, "infill_speed": 270, "solid_infill_speed": 250,
    "top_solid_infill_speed": 200, "gap_fill_speed": 250, "bridge_speed": 50, "travel_speed": 700,
    "first_layer_speed": 50, "small_perimeter_speed": 50,
    "default_acceleration": 10000, "perimeter_acceleration": 5000, "external_perimeter_acceleration": 5000,
    "infill_acceleration": 10000, "top_solid_infill_acceleration": 2000, "first_layer_acceleration": 500,
    "travel_acceleration": 10000,
    "machine_max_acceleration_extruding": 12000, "machine_max_acceleration_x": 12000,
    "machine_max_acceleration_y": 12000, "machine_max_feedrate_x": 500, "machine_max_feedrate_y": 500,
    "machine_max_jerk_x": 9, "machine_max_jerk_y": 9,
    "elefant_foot_compensation": 0.15, "skirts": 0, "brim_width": 0, "complete_objects": 0,
    "support_material": 0, "support_material_auto": 1, "support_material_threshold": 30,
    "support_material_extruder": 0, "support_material_interface_extruder": 0,
    "thin_walls": 0, "ensure_vertical_shell_thickness": 1, "gcode_comments": 0, "gcode_label_objects": 0,
    "min_skirt_length": 0, "wipe_tower_x": 210, "wipe_tower_y": 210, "wipe_tower_width": 35,
    "use_relative_e_distances": 1, "layer_gcode": "G92 E0",
}


def ini(cfg, path):
    path.write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()))
    return path


def multi_3mf(parts, path, name):
    """PrusaSlicer-3MF: één object, één deel per kleur, elk deel op een eigen extruder."""
    verts, tris, vols = [], [], []
    offset = 0
    for color, tm in parts:
        first = sum(len(t) for t in tris)
        verts.append(tm.vertices)
        tris.append(tm.faces + offset)
        offset += len(tm.vertices)
        vols.append((first, first + len(tm.faces) - 1, color))
    V, F = np.vstack(verts), np.vstack(tris)
    V = V - V.min(axis=0) + [3, 3, 0]  # linksonder op het bed; de spoeltoren komt ernaast
    vx = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in V)
    tx = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
             '<metadata name="slic3rpe:Version3mf">1</metadata>'
             f'<resources><object id="1" type="model"><mesh><vertices>{vx}</vertices>'
             f'<triangles>{tx}</triangles></mesh></object></resources>'
             '<build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/></build></model>')
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>', '<object id="1" instances_count="1">',
           f'<metadata type="object" key="name" value="{name}"/>']
    for a, b, color in vols:
        cfg += [f'<volume firstid="{a}" lastid="{b}">',
                f'<metadata type="volume" key="name" value="{color}"/>',
                '<metadata type="volume" key="volume_type" value="ModelPart"/>',
                f'<metadata type="volume" key="extruder" value="{VOLGORDE.index(color) + 1}"/>',
                '</volume>']
    cfg += ["</object>", "</config>"]
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
             'content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
             'relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-'
             '3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/Slic3r_PE_model.config", "\n".join(cfg))
    return path


def run(model, cfg_path, gcode):
    cmd = ["prusa-slicer", "--export-gcode", "--load", str(cfg_path), "--output", str(gcode), str(model)]
    if str(model).endswith(".stl"):
        cmd[1:1] = ["--center", "128,128"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return gcode.read_text(errors="ignore")


def parse(g):
    def grab(key):
        m = re.search(rf"^; {re.escape(key)} = (.+)$", g, re.M)
        return m.group(1).strip() if m else None
    grams = grab("filament used [g]")
    per = [float(v) for v in grams.split(",")] if grams else []
    return {"tijd": grab("estimated printing time (normal mode)"),
            "gram": round(sum(per), 1), "gram_per_extruder": per,
            "wissels": len(re.findall(r"^T[0-9]\s*$", g, re.M))}


def support_mm(model, cfg, tmp):
    """Rijdt de slicer nog eens met automatische support (drempel 30°, zoals Bambu) en meet de supportlengte."""
    c = dict(cfg, support_material=1, gcode_comments=1)
    g = run(model, ini(c, tmp / "sup.ini"), tmp / "sup.gcode")
    total, on = 0.0, False
    for line in g.splitlines():
        if line.startswith(";TYPE:"):
            on = "Support" in line
        elif on and line.startswith("G1") and " E" in line:
            m = re.search(r" E([0-9.]+)", line)
            total += float(m.group(1)) if m else 0
    return round(total, 1)


def main():
    OUT.mkdir(exist_ok=True)
    groups = {}
    for f in sorted((ROOT / "stl").glob("*/*.stl")):
        groups.setdefault(f.stem.split("__")[0], []).append(f)
    result = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, files in groups.items():
            parts = sorted(((f.stem.split("__")[1], trimesh.load(f)) for f in files),
                           key=lambda p: VOLGORDE.index(p[0]))
            n = len(parts)
            cfg = dict(A1)
            if n > 1:
                cfg.update({"nozzle_diameter": ",".join(["0.4"] * 4), "filament_diameter": ",".join(["1.75"] * 4),
                            "filament_density": ",".join(["1.24"] * 4), "filament_type": ",".join(["PLA"] * 4),
                            "temperature": ",".join(["220"] * 4), "first_layer_temperature": ",".join(["220"] * 4),
                            "extruder_colour": "#FF9016;#FFFFFF;#000000;#3F8E43",
                            "single_extruder_multi_material": 1, "wipe_tower": 1})
                model = multi_3mf(parts, tmp / f"{name}.3mf", name)
                lo = np.min([tm.bounds[0] for _, tm in parts], axis=0)
                hi = np.max([tm.bounds[1] for _, tm in parts], axis=0)
                w, h = (hi - lo)[:2]
                if 256 - (3 + w) >= 45:
                    cfg.update(wipe_tower_x=round(3 + w + 6), wipe_tower_y=20)
                else:
                    cfg.update(wipe_tower_x=20, wipe_tower_y=round(3 + h + 6))
            else:
                model = files[0]
            cfg_path = ini(cfg, tmp / f"{name}.ini")
            r = parse(run(model, cfg_path, tmp / f"{name}.gcode"))
            r["kleuren"] = [c for c, _ in parts]
            r["volume_cm3"] = round(sum(float(tm.volume) for _, tm in parts) / 1000, 2)
            r["support_extrusie_mm"] = support_mm(model, cfg, tmp)
            r["support"] = r["support_extrusie_mm"] > 50
            result[name] = r
            print(name, r)
    ini(A1, OUT / "a1_benadering.ini")
    (OUT / "slicer.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
