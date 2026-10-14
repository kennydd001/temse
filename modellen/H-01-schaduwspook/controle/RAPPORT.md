# Controle H-01 Schaduwspook

Automatisch gegenereerd door `cad/controle.py`. Maten in mm. ✅ = in orde, ⚠️ = bekijken.

## 1. Schaduw

De projectie van het beeldje in elke kijkrichting, rij per rij (0,2 mm), tegen het silhouet zoals het geprint wordt. *Ontbreekt* is wat de andere silhouetten wegsnijden (bij vier schaduwen een randje aan de breedste plekken). *Aanpassing* is wat het zelfdragend maken toevoegt aan het getekende silhouet: spitse bovenkanten van gaten, wiggen onder overhangen en dunne steunpilaartjes.

| Variant | Schaduw | Hoek | Ontbreekt | Te veel | Aanpassing |
|---|---|---|---|---|---|
| vier schaduwen | ✅ pompoen | 0° | 2,8 mm² (0,1 %) | 0,0 mm² | 46,2 mm² |
| vier schaduwen | ✅ spook | 45° | 32,5 mm² (1,5 %) | 0,0 mm² | 26,1 mm² |
| vier schaduwen | ✅ schedel | 90° | 5,2 mm² (0,3 %) | 0,0 mm² | 75,2 mm² |
| vier schaduwen | ✅ grafsteen | 135° | 40,8 mm² (2,1 %) | 0,0 mm² | 42,3 mm² |
| BOO en schedel | ✅ BOO | 0° | 0,0 mm² (0,0 %) | 0,0 mm² | 98,9 mm² |
| BOO en schedel | ✅ schedel | 90° | 0,0 mm² (0,0 %) | 0,0 mm² | 61,4 mm² |
| TRICK en TREAT | ✅ TRICK | 0° | 0,0 mm² (0,0 %) | 0,0 mm² | 92,0 mm² |
| TRICK en TREAT | ✅ TREAT | 90° | 0,0 mm² (0,0 %) | 0,0 mm² | 190,8 mm² |

![Silhouetten](silhouetten.png)

## 2. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)

| Variant | Maat | Eén stuk, waterdicht | A1 / A1 mini | Overhang > 48° (mm²) | Te smal (< 0,4 mm) | Gewicht (schatting) |
|---|---|---|---|---|---|---|
| vier schaduwen | 56 × 52 × 60 | ✅ 1 stuk | ✅ / ✅ | ✅ 0 | ✅ geen | ~20 g |
| BOO en schedel | 120 × 48 × 55 | ✅ 1 stuk | ✅ / ✅ | ✅ 0 | ✅ geen | ~53 g |
| TRICK en TREAT | 112 × 124 × 36 | ✅ 1 stuk | ✅ / ✅ | ✅ 0 | ✅ geen | ~82 g |

*Te smal*: kleiner dan 1,5 mm² per laag telt niet (de punt van een boog of steeltje; de slicer laat dat weg). Het gewicht is geschat uit de doorsneden (2 wanden, 15 % infill), niet met een slicer.

Niet getest: de schaduw met een echte gsm-zaklamp. De renders gebruiken een puntlamp van 2 mm op 45 cm; een grotere of dichterbij gehouden lamp geeft een zachtere, iets vertekende schaduw.
