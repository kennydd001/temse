# Controle H-03 Poepmonster

Automatisch gegenereerd door `cad/controle.py`. Maten in mm. ✅ = in orde, ⚠️ = bekijken.

## 1. Gezicht

*Dichtgemaakt* is wat het zelfdragend maken van de getekende gaten afhaalt (spitse bovenkant van de ogen, randjes van één rij), *bijgemaakt* wat het toevoegt. De spleetjes in de buik moeten smaller zijn dan de kleinste poepklodders (~8 mm).

| Variant | Dichtgemaakt | Bijgemaakt | Spleetjes in de buik | Breedste spleet | Dunste stuk tussen twee gaten |
|---|---|---|---|---|---|
| laag | 30,7 mm² | 0,0 mm² | 6 | ✅ 6,6 | ✅ 2,4 |
| hoog | 31,5 mm² | 0,0 mm² | 6 | ✅ 6,6 | ✅ 2,4 |

![Gezicht](gezicht.png)

## 2. Rand en inhoud

| Variant | Rand achteraan (dal / punt) | Rand vooraan (dal / punt) | Inhoud tot de achterrand |
|---|---|---|---|
| laag | 40 / 48 | 75 / 83 | ~301 cm³ |
| hoog | 65 / 73 | 100 / 108 | ~528 cm³ |

De afvalgoot van de printer moet hoger liggen dan de punten van de tanden achteraan.

## 3. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)

| Variant | Maat | Eén stuk, waterdicht | A1 / A1 mini | Overhang > 48° | Vlakke plafonds (bruggen) | Te smal | Gewicht (schatting) |
|---|---|---|---|---|---|---|---|
| laag | 116 × 116 × 83 | ✅ 1 stuk | ✅ / ✅ | ✅ 0 mm² | 0 mm² | ✅ geen | ~58 g |
| hoog | 116 × 116 × 108 | ✅ 1 stuk | ✅ / ✅ | ✅ 0 mm² | 0 mm² | ✅ geen | ~78 g |

*Vlakke plafonds*: de bovenkant van de spleetjes in de buik, 6,6 mm breed van hechting tot hechting. In een wand van 2 mm telt printcheck ze als korte brug.

Het gewicht is geschat uit de doorsneden (3 wanden, boven- en onderlagen, 15 % infill), niet met een slicer.

Niet getest: een echte print, en of hij bij jouw A1 goed onder de afvalgoot staat. Meet dat na (zie de README).
