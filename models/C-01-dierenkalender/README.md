# C-01 Eeuwigdurende dierenkalender: vos (v0.1)

Functioneel prototype volgens het tabblad **Dierenkalender** van de MakerWorld-productielijst.
Printer: Bambu Lab A1 met AMS Lite, maximaal vier kleuren per plaat.

![Opgebouwde kalender](renders/geheel.png)

Vier posities: **DAG | tiental | eenheid | MAAND**, en daaronder vier opbergbakken.
De zes andere dieren (koe, uil, draak, eenhoorn, leeuw, hert) gebruiken straks dezelfde
voet, dezelfde kaartgeleiders en dezelfde kaartjes. Alleen het dierenfront verandert.

> **Status:** CAD en STL gemaakt, **nog niet geprint**. Druk eerst de pasproef af (stap 1)
> voordat je de hele set print.

## Printvolgorde

| Stap | Map | Onderdeel | Kleuren | Maat (mm) |
|---|---|---|---|---|
| 1 | `stl/1-pasproef` | `proef_kanalen`: drie cijferkanalen 2,0 / 2,2 / 2,4 mm met tong | Oranje | 84 × 60 × 8 |
| 1 | `stl/1-pasproef` | `proef_sleuven`: drie sleuven 3,6 / 3,8 / 4,0 mm onder 8° | Oranje | 108 × 24 × 23 |
| 1 | `stl/1-pasproef` | `proef_kaartjes`: 0, 1, 2 en WOENSDAG | Wit + zwart | 134 × 28 × 1,6 |
| 2 | `stl/4-kaartjes` | `kaartjes_maanden` (12), `kaartjes_dagen` (7), `kaartjes_cijfers` (4 + 10) | Wit + zwart | max. 236 × 124 |
| 3 | `stl/3-voet` | `voet`: vier bakken, schuine sleuf, open achtervak | Oranje | 206 × 72 × 36 |
| 4 | `stl/2-vos` | `front_vos`: bord, vos, geleiders, labels, blaadjes | Oranje, wit, zwart, groen | 200 × 210 × 8 |

Kleuren: Pumpkin Orange 10301, Ivory White 11100, Charcoal 11101 en Mistletoe Green 10502.
Elk bestand eindigt op `__oranje`, `__wit`, `__zwart` of `__groen`. Het volume per onderdeel
staat in `stl/onderdelen.json`; dat is massief volume, het echte filamentgebruik komt uit de slicer.

### Meerkleurig laden in Bambu Studio

1. Sleep **alle** STL-bestanden van één onderdeel tegelijk in Bambu Studio (bijvoorbeeld de vier `front_vos__*.stl`).
2. Kies **Ja** bij *"Load these files as a single object with multiple parts?"*. De delen staan al op de juiste plek.
3. Wijs per deel het filament toe in de objectenlijst (oranje, wit, zwart, groen).
4. Leg het front en de kaartjes **plat**, met de achterkant op de plaat. De kleuren zitten alleen in de bovenste
   millimeters, dus er zijn maar een paar lagen met kleurwissels.
5. Startinstellingen: 0,2 mm laag, 0,4 mm nozzle, Textured PEI Plate, 15 % infill. Probeer het eerst zonder supports
   en controleer de lippen in de laagvoorvertoning (zie hieronder).

## Pasproef: wat je test

![Pasproef](renders/proef.png)

Steek de kanaalstrook met zijn tong in elke sleuf van het sleuvenblok en schuif de cijferkaartjes in elk kanaal.

- [ ] **Kanaal** (2,0 / 2,2 / 2,4): het kaartje glijdt er makkelijk in en uit, maar valt er niet uit als je de strook voorover kantelt.
- [ ] **Uitnemen**: het kaartje steekt 9 mm boven de geleiders uit, zodat je het aan de zijkanten kunt vastpakken. In de opening
      in de draagrand (met nagelkuiltje) kun je het van onderen omhoog duwen.
- [ ] **Sleuf** (3,6 / 3,8 / 4,0): de tong zakt erin zonder te klemmen en de strook staat niet te wiebelen.
- [ ] **Kanteling**: de strook leunt ongeveer 8° achterover en blijft staan als je een kaartje wisselt.
- [ ] **Lippen**: de 45°-lippen zijn netjes geprint en niet doorgezakt.
- [ ] **Kaartjes**: WOENSDAG en de cijfers zijn goed leesbaar en de letters zijn heel.

Pas daarna de winnende maten aan in `cad/params.py` (`CHANNEL`, `SLOT_W`) en genereer opnieuw.

## Opnieuw genereren

```bash
cd cad
pip install -r requirements.txt
python kalender.py          # alle STL's (pasproef + set), ongeveer 3 seconden
python kalender.py proef    # alleen de pasproef
python render.py geheel     # renders/geheel.png  (ook: vos, proef; --snel voor een snelle preview)
```

- `cad/params.py`: alle maten, met de startwaarden uit de productielijst
- `cad/kalender.py`: geleiders, front, voet, kaartjes, pasproef en export
- `cad/fox.py`: de eigen getekende vossenkop (eenvoudige vormen, geen overgenomen geometrie)
- `cad/render.py`: Blender/Cycles-render, werkt zonder scherm

## Ontwerpkeuzes en afwijkingen van de startmaten

| Onderwerp | Keuze | Reden |
|---|---|---|
| Kaarttekst | 0,4 mm Charcoal **verzonken** in de bovenkant, kaartje overal 1,6 mm | Vlak kaartje schuift beter onder de lippen en stapelt recht; losse letters kunnen niet afbreken |
| Lettertype | Fredoka SemiBold (cijfers, labels) en Fredoka SemiBold smal (dagen, maanden), 8 mm hoog | DONDERDAG past daarmee in 56 mm; alleen dat woord is nog ~5 % extra versmald |
| Zijlippen | Lip van 2 mm met een onderkant onder 45°; kanaal 2,2 mm bij de rib | Printbaar zonder support terwijl het front plat ligt |
| Kaartje uitnemen | Geleiders 19 mm hoog, kaartje steekt 9 mm boven uit; opening van 8 mm in de draagrand | Vastpakken aan de zijkanten of van onderen omhoog duwen |
| Sleufdiepte | 16,5 mm in plaats van 16 mm | De tong raakt de bodem niet; de schouders van het bord rusten op de voet |
| Onderhoeken bord | Recht (bovenhoeken R8) | Vlakke schouders naast de tong |
| Voorkant bakken | 10 mm hoog, zijwanden lopen schuin op naar 36 mm | Tekst van het voorste kaartje blijft zichtbaar, vingers kunnen erbij |
| Buitenwanden voet | 7,4 mm | Vier bakken van 58/24/24/78 mm passen precies in 206 mm, met ruimte voor de R6-hoeken |
| Achtervak | Open vak achter de sleuf | Minder filament dan een massieve voet; extra opbergruimte |

## Licentie lettertype

Fredoka valt onder de SIL Open Font License (`fonts/OFL.txt`). De smalle variant is een statische instantie
(wdth 75, wght 600) van de variabele Fredoka uit Google Fonts.
