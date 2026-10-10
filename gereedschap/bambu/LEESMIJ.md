# Bambu Studio-hulpbestanden

- `profielen/*.json`: de systeemprofielen "Bambu Lab A1 0.4 nozzle", "0.20mm Standard @BBL A1",
  "Bambu PLA Basic @BBL A1" en "Bambu PLA Matte @BBL A1", plat gemaakt (alle `inherits` opgelost) uit
  `bambulab/BambuStudio` (resources/profiles/BBL, profielversie 02.08.00.09).
  `project_sjabloon_A1.json` is de `project_settings.config` uit Bambu's eigen kalibratieproject
  `resources/calib/pressure_advance/auto_pa_line_single.3mf` (Bambu Studio 02.00.02.01, A1).
- `flushvol.py`: Python-versie van Bambu Studio's automatische spoelvolumes (FlushVolCalc /
  FlushVolPredictor), inclusief de meettabel. Geeft voor het A1-voorbeeldproject exact dezelfde waarden.

Deze bestanden zijn afgeleid van Bambu Studio en vallen onder de licentie daarvan (GNU AGPL-3.0).
Ze worden alleen gebruikt om de projectbestanden (`.3mf`) te maken.
