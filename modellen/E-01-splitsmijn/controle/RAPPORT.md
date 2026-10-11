# Controle E-01 Splitsmijn: pasproef van het valbord

Automatisch gegenereerd door `cad/controle.py` (v0.1). Maten in mm. ✅ = in orde, ⚠️ = bekijken.

![Valbord](valbord.png)

## 1. Muntdoorgang

- ✅ Kleinste opening tussen twee pinnen of een pin en de wand: 29.0 mm, voor een muntje van 25 mm.
- ✅ Plekken waar een muntje op twee steunpunten kan blijven liggen (ook met wrijving 0,3 meegerekend): geen.
- ✅ Boven de vlakke kantelplank is er nog 4 mm tot het plafond: een muntje rolt er vrij over. De verdeler staat recht onder de middelste pin van de onderste rij (tussenruimte kleiner dan een muntje): een muntje valt er altijd naast.

## 2. Valsimulatie

Het valbord nagebouwd in een 2D-fysicamodel (pymunk, zie `cad/simulatie.py`). Een ronde zoals in de klas: één duw tegen de slinger, dan 8 muntjes met 0,8 tot 2 seconden ertussen, steeds gemikt op dezelfde plek. Doorgerekend voor 12 combinaties van stuit, wrijving en demping, en twee waarden voor de wrijving van het mes.

- ✅ Kans op links, voor elke combinatie en elk mikpunt (links, midden, rechts in de trechter): 0.33 tot 0.64. Met 64 muntjes per vakje is ±0,15 gewoon toeval.
- ✅ Gemiddeld over alle combinaties: mikken links 0.51, midden 0.47, rechts 0.48 naar links (de drie mikpunten met dezelfde duwen en tijden). Het mikpunt beslist niet meer.
- ✅ Mikeffect (links mikken min rechts mikken) per stuit: 0.3: +0.04, 0.5: +0.02, 0.7: +0.03. Zonder slinger is dat 0,3 tot 0,9.
- ✅ Muntjes die bleven hangen: 0 van 4608.
- De slinger had bij het laatste muntje van een ronde nog 29° uitslag (mediaan), minstens 16°: één duw is genoeg voor een hele ronde.

Hoe vaak elke splitsing van 8 voorkomt (links + rechts), naast een eerlijk muntstuk:

| Splitsing | 0+8 | 1+7 | 2+6 | 3+5 | 4+4 | 5+3 | 6+2 | 7+1 | 8+0 |
|---|---|---|---|---|---|---|---|---|---|
| Simulatie | 0 % | 2 % | 13 % | 24 % | 26 % | 16 % | 13 % | 4 % | 0 % |
| Binomiaal | 0 % | 3 % | 11 % | 22 % | 27 % | 22 % | 11 % | 3 % | 0 % |

Het muntje rolt groot naast de pinnen in een vaste baan over de pinnen: zonder slinger beslist het mikpunt (in de simulatie 65 tot 95 % naar de kant waar het kind mikt). De kantelplank op de slinger maakt het eerlijk: welke kant op dat moment lager ligt.

## 3. Slinger

- Slingertijd 0.77 s, massa ~42 g (gewicht helemaal massief).
- ✅ Het zwaartepunt van wat draait ligt op y = 18.4, boven het stuk waar het mes op de groef rust (16.9 tot 22.9): de slinger kantelt niet naar voren of naar achteren.
- ✅ Het mes kan 45° draaien voor zijn flanken de groef raken; de slinger slaat hooguit 35° uit.
- Uitslag na een duw van 35°, om de 5 seconden (zonder muntjes):
  - wrijving van het mes als een pen van 4 mm met wrijvingsgetal 0.02: 35°, 34°, 34°, 33°, 32°, 31°
  - wrijving van het mes als een pen van 4 mm met wrijvingsgetal 0.06: 35°, 33°, 32°, 30°, 28°, 27°
  Een gewone geprinte pen (wrijvingsgetal 0,2 tot 0,4) valt na 10 tot 20 seconden stil: daarom het mes.

## 4. Spelingen en botsingen over de hele uitslag (±35°)

- ✅ Botsingen: geen.
- ✅ Het mes rust op de bodem van de groef (0.0 mm).

| Paar | Kleinste speling | Bij hoek |
|---|---|---|
| plank - achterplaat | ✅ 0.50 | -35° |
| plank en naaf - voorplaat | ✅ 0.50 | -35° |
| slingerarm - voorplaat | ✅ 0.50 | -35° |
| slingerarm - beugel | ✅ 0.50 | -35° |
| gewicht - voet | ✅ 3.60 | -5° |
| mesblok - voorplaat | ✅ 5.50 | -35° |

## 5. Verbindingen

Er zijn geen klikhaakjes: alles klemt of schuift. Speling per kant in het CAD, gemeten tussen de onderdelen in de montagestand. Hoe strak het geprint wordt, hangt af van de printer: een staand gat komt meestal 0,05 tot 0,1 mm kleiner uit. Print daarom eerst plaat 1 (de passtrip) en zet de gekozen maten in `cad/params.py`.

| Verbinding | Speling per kant | Bedoeld | Kies met |
|---|---|---|---|
| nokje Ø 3.0 in de voorplaat (26 stuks tegelijk) | 0.10 mm | licht klemmen | `NOK_GAT` |
| pennetjes Ø 3 van de arm in het mesblok | 0.05 mm | klemmen | `MES_GAT` |
| pennen Ø 4 van de voorplaat in de beugel (liggend druppelgat) | 0.05 mm | klemmen | `BEUGEL_GAT` |
| zeskant van de plank in de arm | 0.10 mm | schuiven zonder wiebelen | `ZESKANT_SPEL` |
| zijranden in de kokers van de voet | 0.30 mm | schuiven | `SPEL` |
| naaf van de plank in het gat van de voorplaat | 1.00 mm | vrij | `NAAF_R` |
| vin van de verdeler in de spleet | 0.60 mm | vrij |  |

## 6. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)

| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |
|---|---|---|---|---|---|
| 1 passtrip | Charcoal | 119 × 40 × 11 | ✅ | ✅ 0 | ✅ geen |
| 2 achterplaat | Charcoal | 218 × 230 × 10 | ✅ | ✅ 0 | ✅ geen |
| 3 voorplaat | Charcoal | 218 × 230 × 14 | ✅ | ✅ 0 | ✅ geen |
| 4 voet en beugel | Latte Brown | 226 × 106 × 52 | ✅ | ✅ 0 | ✅ geen |
| 5 slinger, mesblok en plank | Maroon Red | 110 × 184 × 17 | ✅ | ✅ 0 | ✅ geen |
| 6 munten | Marine Blue | 145 × 55 × 4 | ✅ | ✅ 0 | ✅ geen |

### Slicer (PrusaSlicer, A1-achtig profiel)

| Plaat | Filament | Tijd (indicatie) | Support nodig |
|---|---|---|---|
| 1 passtrip | 11.7 g | 31m 42s | ✅ nee |
| 2 achterplaat | 137.7 g | 3h 16m 2s | ✅ nee |
| 3 voorplaat | 108.5 g | 3h 1m 52s | ✅ nee |
| 4 voet en beugel | 74.2 g | 1h 36m 21s | ✅ nee |
| 5 slinger, mesblok en plank | 46.1 g | 54m 13s | ✅ nee |
| 6 munten | 16.5 g | 32m 29s | ✅ nee |
