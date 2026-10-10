# Controle B-34A Taakvreter

Automatisch gegenereerd door `cad/controle.py` (v0.1). Maten in mm. ✅ = in orde, ⚠️ = bekijken.

![Mechanisme](mechanisme.png)

## 1. Botsingen over een hele slag

✅ 24 standen van de kruk (om de 15°), 28 onderdelen inclusief O-ringen en paspennen, elk paar: nergens overlap. Bewust klemmend: de paspennen in de voorhelft en de hoorntjes, en de O-ringen tegen de bovenste rol (0,65 mm).

## 2. Spelingen over een hele slag

Vrije ruimte tussen wat beweegt en wat stilstaat (de lagers zelf staan in de tabel eronder):

| Paar | Kleinste speling | Bij krukhoek |
|---|---|---|
| onderste rol - romp (schouder, rol, tandwiel) | ✅ 0.30 | 0° |
| bovenste rol - beugel, rib, glijgoot | ✅ 0.30 | 0° |
| kruk - romp (kraag, arm) | ✅ 0.33 | 0° |
| knop (en klikpen) - zijwand | ✅ 0.90 | 0° |
| krukschijf - beugel | ✅ 0.70 | 0° |
| drijfstang - romp | ✅ 1.00 | 0° |
| drijfstang - onderste rol | ✅ 3.90 | 270° |
| tuimelaar - zijwand | ✅ 0.76 | 90° |
| oogtrommels - gezicht | ✅ 2.69 | 0° |
| pupil - gezicht | ✅ 4.90 | 240° |
| invoertunnel - onderste rol | ✅ 2.26 | 0° |
| invoertunnel - bovenste rol | ✅ 1.00 | 0° |
| invoertunnel - O-ring | ✅ 1.22 | 0° |
| lade - romp | ✅ 0.40 | 0° |

Lagers (speling rondom, ontwerpmaat; de pasproef laat zien of je printer dat haalt):

| Lager | As of pen | Gat | Speling |
|---|---|---|---|
| asstompen van de rollen en de kruk-as in de zijwanden | Ø 6.0 | Ø 6.30 | 0.15 |
| nokas in de beugel | Ø 6.0 | Ø 6.30 | 0.15 |
| oogas in de kliklagers | Ø 6.0 | Ø 6.30 | 0.15 |
| D-eind van de oogas in de tuimelaar | Ø 6.0 | Ø 6.30 | 0.15 |
| pennen van tuimelaar en drijfstang | Ø 4.0 | Ø 4.40 | 0.20 |
| klikpen van de knop in de krukarm | Ø 4.5 | Ø 4.90 | 0.20 |
| zeskant van kruk-as en nokas in de rollen (over de platte kanten) | Ø 5.2 | Ø 5.40 | 0.10 |

## 3. Tandwielen

Twee tandwielen m = 1.0, z = 17, hart-op-hart 17.0 mm: de rollen draaien 1:1 tegen elkaar in. Speling tussen de tanden over één steek: kleinste ✅ 0.076, grootste 0.076. De kale rollen staan 0.4 mm uit elkaar; de O-ringen overbruggen dat.

## 4. Oogmechaniek

Vierstangenmechanisme: kruk 5.0 mm op de bovenste rol, drijfstang 34.0 mm, tuimelaar 11.0 mm op de oogas, vaste stang 34.66 mm.

- ✅ Grashof met de kruk als kortste stang: de kruk draait rond, de ogen schommelen heen en weer.
- Uitslag van de ogen: 54.2°, de pupillen gaan 5.9 mm op en neer (één keer per omwenteling van de kruk).
- ✅ Overbrengingshoek 57.9° tot 90.0° (boven 40° loopt het licht, zonder klemmen).

## 5. Papierpad

- Sleuf 82 × 4 mm voor blaadjes van 76 mm: tot 4.7° scheef past het blaadje nog.
- ✅ De rollen lopen van -42 tot 42, breder dan de sleuf: de tandwielen zitten 1.3 mm buiten de rand van de sleuf, daar kan geen papier komen.
- ✅ De O-ringen drukken nominaal 0.65 mm in tegen de bovenste rol. De lagers hebben samen tot 0.4 mm speling (twee draaigaten en een zeskant per kant); die duwen de O-ringen eerst weg, dus er blijft minstens 0.25 mm over, plus de dikte van het blaadje.
- ✅ Tussen de O-ringen en het schap eronder: 0.85 mm.
- De tunnel stopt 10.5 mm voor het knijppunt; de rollen vormen daar een trechter naar het papiervlak.
- Achter de rollen buigt de glijgoot het blaadje omlaag: het plafond begint 0.77 mm boven het papiervlak. Achter het schap is een opening van 49.0 mm naar de lade.
- Lade binnenmaat 107 × 88 × 29 mm.

## 6. Vingerproef

- ✅ De invoer is 4 mm hoog: een vinger (ook van een kind, Ø 8 mm) past er niet in. Het knijppunt zit 35.2 mm achter de lip.
- Het schap onder de rollen loopt tot voorbij de rollen: van onderen (lade erin) kun je er niet bij.
- ⚠️ Met de lade eruit komt een vinger (Ø 8 mm) achter het schap tot 0.66 mm van de achterkant van de onderste rol. Daar lopen de rollen uit elkaar (geen intrekkende kant), maar draai de kruk toch alleen met de lade erin.

## 7. Lade

- ✅ Speling rondom in de baai: 0.4 mm.
- ✅ Er helemaal uit schuiven: haakt nergens.

## 8. Printbaarheid (Bambu Lab A1, 0,4 mm nozzle, 0,2 mm laag)

| Plaat | Kleur | Maat | Past | Overhang > 48° (mm²) | Te smal (< 0,4 mm) |
|---|---|---|---|---|---|
| 1 voorhelft, kruk en paspennen | Pumpkin Orange | 232 × 136 × 44 | ✅ | ✅ 0 | ✅ geen |
| 2 achterhelft | Pumpkin Orange | 124 × 136 × 62 | ✅ | ✅ 0 | ✅ geen |
| 3 lade, ogen en witte delen | Ivory White | 197 × 112 × 93 | ✅ | ✅ 0 | ✅ geen |
| 4 rollen en zwarte delen | Charcoal | 204 × 93 × 107 | ✅ | ✅ 0 | ✅ geen |

### Slicer (PrusaSlicer, A1-achtig profiel)

| Plaat | Filament | Tijd (indicatie) | Support nodig |
|---|---|---|---|
| 1 voorhelft, kruk en paspennen | 114.9 g | 3h 12m 3s | ✅ nee |
| 2 achterhelft | 198.6 g | 4h 28m 20s | ✅ nee |
| 3 lade, ogen en witte delen | 72.0 g | 1h 58m 42s | ✅ nee |
| 4 rollen en zwarte delen | 46.5 g | 1h 33m 32s | ✅ nee |
| pasproef 1 proefframe | 86.7 g | 2h 14m 15s | ✅ nee |
| pasproef 2 rollen, kruk en paspennen | 58.9 g | 2h 12m 32s | ✅ nee |
