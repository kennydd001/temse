# B-34B Repair Pod (v0.1)

*English: [README.en.md](README.en.md)*

Compacte capsule voor op de werkbank, volgens regel **B-34B** van de MakerWorld-bouwselectie. Trek je de
oranje knop omhoog, dan klapt de pod open tot een kleine werkplek. Links en rechts zakken twee
schroevenbakjes naar buiten. Ze blijven daarbij horizontaal, want elk bakje hangt aan een parallellogram
van vier armen. Op het dak staat een houder voor tien 1/4"-bits. Er is geen hardware nodig: alles klikt in
elkaar met geprinte pennen.

![Dicht, half open, open](renders/reeks.png)

> **Status:** CAD, STL's, het Bambu Studio-project en de montage- en printcontrole zijn klaar. Er is **nog
> niets echt geprint**. Druk eerst de pasproef af (`pasproef/pasproef.3mf`), zie hieronder.
>
> De 3MF-projecten zijn gecontroleerd tegen de regels van het Bambu-formaat
> (`gereedschap/bambu/valideer.py` in de root van de repo). Ze zijn hier niet in Bambu Studio zelf
> geopend: dat programma draait niet in deze omgeving.

| | Dicht | Open |
|---|---|---|
| Maat (b × d × h) | 114 × 102 × 96 mm | 212 × 102 × 96 mm |
| Bakjes | 2 × 2 vakken, binnenmaat ca. 41 × 39 × 23 mm per vak | bodem 23 mm boven de tafel |
| Knop | slag 11 mm | |

Kleuren (Bambu PLA, uit de kleurplanning): **Charcoal 11101** voor de voet en het dak, **Pumpkin Orange
10301** voor de armen, de schuif en de pennen, en **Ivory White 11100** voor de bakjes.

## Hoe het werkt

![Mechanisme](controle/mechanisme.png)

- Elk bakje hangt aan vier armen van 58 mm: twee vooraan en twee achteraan. Elk paar vormt een
  parallellogram, dus het bakje schuift en draait niet. Het blijft horizontaal van dicht (92°) tot open (28°).
- De twee binnenste voorarmen hebben een tandsector. Die grijpt in een heugel met tanden aan beide kanten:
  de **schuif**. De schuif loopt in een zwaluwstaartgroef vooraan in de middenrug en eindigt in de knop
  boven het dak. Eén knop beweegt dus beide kanten tegelijk en precies gelijk.
- **Open**: de zwaartekracht trekt de bakjes naar buiten. In de open stand rust de binnenarm op een nok in
  de voet.
- **Dicht**: de knop ligt op het dak en de pod blijft vanzelf dicht. Om hem te sluiten met 150 g in elk bakje
  duw je hooguit 18 N, ongeveer 1,8 kg, op de knop.
- **Stabiliteit**: de voet is 114 mm breed. Met 150 g in één bakje en het andere leeg blijft het zwaartepunt
  minstens 31 mm binnen de rand van de voet.

## Onderdelen

| Onderdeel | Aantal | Kleur | Maat (mm) | Bestand |
|---|---|---|---|---|
| voet (voetplaat, middenrug, flanken, draaipunten, nokken) | 1 | Charcoal | 114 × 97 × 86 | `print/voet.stl` |
| dak met bitkam (10 × 1/4") | 1 | Charcoal | 108 × 102 × 14 | `print/dak.stl` |
| bakje (2 vakken) | 2 | Ivory White | 46 × 86 × 28 | `print/bak.stl` |
| tandarm (binnenste voorarm met tandsector) | 2 | Pumpkin Orange | 16 × 74 × 4,5 | `print/tandarm.stl` |
| arm | 6 | Pumpkin Orange | 12 × 68 × 4,5 | `print/arm.stl` |
| schuif (heugel en knop) | 1 | Pumpkin Orange | 24 × 95 × 8 | `print/schuif.stl` |
| klikpen | 16 + 2 reserve | Pumpkin Orange | Ø 9 × 12 | `print/pen.stl` |

Dat zijn zeven verschillende onderdelen. Beide bakjes zijn gelijk, net als beide tandarmen en alle zes
gewone armen: aan de linkerkant monteer je ze omgedraaid.

## Printen met het Bambu Studio-project

Open `print/repair-pod.3mf` in Bambu Studio. Het project heeft drie platen, elk in één kleur:

| Plaat | Wat | Kleur | Filament* | Tijd* |
|---|---|---|---|---|
| 1 | voet en dak | Charcoal 11101 | 180 g | 3 u 43 |
| 2 | twee bakjes | Ivory White 11100 | 59 g | 1 u 12 |
| 3 | 2 tandarmen, 6 armen, schuif, 18 pennen | Pumpkin Orange 10301 | 29 g | 1 u 15 |

Samen is dat ongeveer **270 g filament en 6 uur 10 printen**.

\* Gewicht en tijd komen uit PrusaSlicer met een profiel dat de A1 benadert, zie
[controle/RAPPORT.md](controle/RAPPORT.md). Het gewicht is betrouwbaar; de tijd rekent Bambu Studio zelf uit.

- **Instellingen**: *0.20mm Standard @BBL A1* op de *Textured PEI Plate*, 0,4 mm nozzle, **geen support**.
  Per object staat dit al in het project:
  - voet, dak, bakjes: 3 wanden;
  - armen en schuif: 3 wanden, 40 % infill;
  - pennen: 4 wanden (dan zijn ze helemaal massief), geen brim.
- **Printligging**:
  - Voet, dak en bakjes staan rechtop.
  - Armen en schuif liggen plat, zodat de tandjes en de ogen het mooist worden.
  - Pennen staan op hun kop, zodat de schacht precies rond is.
- Liggende gaten zijn druppelvormig (punt naar boven) en ronde randen hebben aan de onderkant een
  45°-afschuining. Zo heeft niets support nodig. De slicer bevestigt dat voor alle platen.

## Eerst de pasproef

Print `pasproef/pasproef.3mf` (één kleur naar keuze). Het project heeft twee platen.

**Plaat 1: scharnierstrip, bitblokje en 4 pennen** (16 g, 40 min).

- **Wandstrip**: drie liggende gaten van 4,95, 5,05 en 5,15 mm, zoals in de voet en de bakjes. Een pen moet
  er met een duidelijke klik in gaan en mag er niet vanzelf uitvallen.
- **Armstrip**: drie staande gaten van 5,15, 5,25 en 5,35 mm, zoals in de armen. De arm moet vrij draaien
  op de pen, zonder te klapperen.
- **Bitblokje**: drie zeskanten van 6,4, 6,6 en 6,8 mm. Een bit moet erin blijven als je het blokje
  omkeert, en er toch makkelijk uit te halen zijn.

Pas de beste maten aan in `cad/params.py` (`GAT_WAND`, `GAT_ARM`, `BIT_SW`) en draai `python bouw.py` opnieuw.

**Plaat 2: één zijmechanisme** (140 g, 3 u 05). Dit is de rechterhelft van de voet met de hele middenrug,
één bakje, een tandarm, drie armen, tien pennen en de schuif. Monteer het zoals hieronder. Leg dan 150 g in
het bakje (zo'n dertig M3-schroefjes) en controleer:

- [ ] Het bakje blijft horizontaal over de hele slag (hooguit ~1,3° scheef door de speling in de scharnieren).
- [ ] De scharnieren lopen soepel, zonder te klemmen of te klapperen.
- [ ] Open: de binnenarm rust op de nok. Dicht: zonder dak rust het bakje tegen de rug (in de echte pod
      houdt de knop op het dak het 0,6 mm daarvóór tegen).
- [ ] Nergens schuurt iets: een arm langs de voet, het bakje langs de flank, de schuif in de groef.
- [ ] De knop heeft een slag van 11 mm, en de tanden springen niet over als je de knop met een vol bakje
      omlaag duwt. Steek de schuif erin zoals in stap 3 hieronder.

## In elkaar zetten

1. **Armen en bakjes**: elke pen gaat door het oog van een arm in een gat van de voet of het bakje en klikt
   erachter vast.
   - Vooraan, per kant:
     - de tandarm op het binnenste draaipunt, met de tandsector naar het midden;
     - een gewone arm op het buitenste draaipunt.
   - Achteraan: twee gewone armen, op dezelfde manier.
   - Links monteer je dezelfde delen omgedraaid.
   - De bovenste ogen gaan op de pennen in het bakje, de afgeschuinde onderrand van het bakje naar buiten.

   Zonder schuif zakken de bakjes vanzelf open, tot de binnenarmen op hun nok rusten. Laat ze zo staan.
2. **Dak**: leg het dak op de rug, met de twee nokjes in de gaten en de inkeping vooraan. Het dak ligt los en
   blijft op zijn plaats door zijn gewicht en de nokjes. Wil je de pod aan het dak optillen, zet het dan vast
   met een druppel lijm op de nokjes.
3. **Schuif**: steek de schuif met de pod open van boven door de inkeping in het dak, met de heugel vóór de
   rug en de zwaluwstaart in de groef. Laat hem zakken tot de heugel tussen de twee tandsectoren grijpt, en
   duw dan de knop omlaag: beide bakjes gaan gelijk dicht.
4. **Controle**: dicht moet de knop precies op het dak liggen.
   - Blijft de knop een paar mm boven het dak staan, dan zit de heugel een tand te hoog.
   - Gaat de pod niet helemaal dicht, dan zit hij een tand te laag.

   Open dan de pod, trek de schuif eruit en laat hem één tand lager (knop te hoog) of hoger (pod niet dicht)
   opnieuw grijpen.

## Mappen

```
print/        repair-pod.3mf (3 platen) en elk onderdeel als STL in printligging
pasproef/     pasproef.3mf (2 platen) en de STL's die niet in print/ staan
cad/          de bron: Python-scripts die alles hierboven maken
controle/     RAPPORT.md: botsingen, spelingen, tandwielen, eindstops, kracht, kantelen, printbaarheid, slicer
renders/      Blender-renders
onderdelen.json   volume en maat per STL
```

## Zelf bouwen

Vereist Python 3 met `manifold3d`, `shapely`, `trimesh`, `numpy`, `contourpy` en `matplotlib`. De
gedeelde code staat in [`gereedschap/`](../../gereedschap/) in de root van de repo.

```
cd cad
python bouw.py        # print/ en pasproef/: STL's en 3MF-projecten
python controle.py    # controle/RAPPORT.md (voor de slicertabel eerst python slice.py, vereist prusa-slicer)
python render.py alle # renders/ (vereist de pip-module bpy)
```

Alle maten staan in `cad/params.py`. Het mechanisme wordt bij elke run opnieuw doorgerekend:

- botsingen over 33 standen;
- spelingen;
- tandspeling;
- eindstops.

Pas je iets aan, kijk dan eerst in `controle/RAPPORT.md` of alles nog groen is.

![Detail tandsector](renders/detail.png)
![Explosietekening](renders/explosie.png)
