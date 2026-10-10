"""Gedeeld gereedschap voor alle modellen in deze repo.

  geom.py         2D (shapely) en 3D (manifold3d/trimesh): vormen, extruderen, booleans, tekst, STL
  bambu/          Bambu Studio-projecten (.3mf) schrijven en controleren, A1-profielen, spoelvolumes
  slicer.py       PrusaSlicer met een A1-achtig profiel: gewicht, tijd, wissels, support
  printcheck.py   printbaarheid van een plaat: maat, overhang, te dunne details, kleurwissels
  fonts/          Fredoka (SIL OFL)

Een model gebruikt dit via een `pad.py` in zijn cad-map (zet de root van de repo op sys.path).
"""
