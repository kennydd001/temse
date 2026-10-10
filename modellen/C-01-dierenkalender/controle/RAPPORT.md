# Controle C-01 dierenkalender

Automatisch gegenereerd door `cad/controle.py` uit de STL's, de Bambu-projecten en de montage. Maten in mm. ✅ = in orde, ⚠️ = bekijken.

## 1. Botsingen in gemonteerde stand (vos; voet, geleiders en kaartjes zijn voor elk dier gelijk)

✅ 69 combinaties gecontroleerd; grootste overlap 0,0000 mm³.


## 2. Spelingen

| Kanaal | Kaart ↔ rib | Kaart ↔ lip | Lip over kaartrand (slechtst) |
|---|---|---|---|
| kanaal dag (ZATERDAG) | 0,50 | 1,31 | 0,99 |
| kanaal cijfer (1) | 0,50 | 1,31 | 0,99 |
| kanaal cijfer (0) | 0,50 | 1,31 | 0,99 |
| kanaal maand (OKTOBER) | 0,50 | 1,31 | 0,99 |

| Tong in sleuf | Speling |
|---|---|
| voorzijde | 0,40 |
| achterzijde | 0,40 |
| onderkant tong ↔ sleufbodem | 0,98 |
| uiteinden links / rechts | 0,40 / 0,40 |
| schouder boven voet (0 = rust erop) | 0,000 |

| Bak, alle kaartjes erin | Zijspeling | Kleinste afstand in de achterhoek | Ruimte vóór de stapel |
|---|---|---|---|
| bak dag (7 st.) | 1,00 | 0,30 | 14,2 |
| bak tiental (4 st.) | 1,00 | 0,30 | 19,1 |
| bak eenheid (10 st.) | 1,00 | 0,30 | 9,2 |
| bak maand (12 st.) | 1,00 | 0,30 | 6,0 |

![Doorsneden](../renders/doorsneden.png)

## 3. Uitneemroute per variant

Elk kaartje 0,05 mm van de plaat en vanaf 1 mm boven de draagrand omhoog geschoven tot de onderrand boven de geleiders is. Onderweg is de kleinste afstand de zijspeling langs de ribben; bovenaan telt de afstand tot pootjes, hoefjes of kin.

| Variant | Kleinste afstand onderweg | Kleinste afstand tot het dier bovenaan |
|---|---|---|
| 01 vos | ✅ 0,50 | ✅ 3,73 |
| 02 koe | ✅ 0,50 | ✅ 3,00 |
| 03 uil | ✅ 0,50 | ✅ 4,50 |
| 04 draak | ✅ 0,50 | ✅ 3,00 |
| 05 eenhoorn | ✅ 0,50 | ✅ 3,17 |
| 06 leeuw | ✅ 0,50 | ✅ 3,82 |
| 07 hert | ✅ 0,50 | ✅ 3,00 |
| 08 mario | ✅ 0,50 | ✅ 4,00 |
| 09 sonic | ✅ 0,50 | ✅ 4,00 |
| 10 spiderman | ✅ 0,50 | ✅ 4,01 |
| 11 steve | ✅ 0,50 | ✅ 5,00 |
| 12 creeper | ✅ 0,50 | ✅ 5,00 |

## 4. Printbaarheid (Bambu Lab A1, 256 × 256 mm, 0,4 mm nozzle, 0,2 mm laag)

| Onderdeel | Maat | Past | Overhang > 48° (mm²) | 45°-vlakken (mm²) | Korte bruggen < 3 mm (mm²) | Te smal (< 0,4 mm) | Lagen met kleurwissel |
|---|---|---|---|---|---|---|---|
| fanart/08-mario/en/front_mario | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 122 | ✅ geen | 27 |
| fanart/08-mario/nl/front_mario | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 122 | ✅ geen | 27 |
| fanart/09-sonic/en/front_sonic | 200 × 210 × 8 | ✅ | ✅ 0 | 801 | 118 | ✅ geen | 26 |
| fanart/09-sonic/nl/front_sonic | 200 × 210 × 8 | ✅ | ✅ 0 | 801 | 118 | ✅ geen | 26 |
| fanart/10-spiderman/en/front_spiderman | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 153 | ✅ geen | 26 |
| fanart/10-spiderman/nl/front_spiderman | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 153 | ✅ geen | 26 |
| fanart/11-steve/en/front_steve | 200 × 208 × 8 | ✅ | ✅ 0 | 821 | 121 | ✅ geen | 25 |
| fanart/11-steve/nl/front_steve | 200 × 208 × 8 | ✅ | ✅ 0 | 821 | 121 | ✅ geen | 25 |
| fanart/12-creeper/en/front_creeper | 200 × 208 × 8 | ✅ | ✅ 0 | 821 | 134 | ✅ geen | 25 |
| fanart/12-creeper/nl/front_creeper | 200 × 208 × 8 | ✅ | ✅ 0 | 821 | 134 | ✅ geen | 25 |
| gedeeld/kaartjes/cijfers | 178 × 60 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/kaartjes/en/days | 236 × 60 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/kaartjes/en/months | 236 × 124 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/kaartjes/nl/dagen | 236 × 60 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/kaartjes/nl/maanden | 236 × 124 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/pasproef/en/test-cards | 134 × 28 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/pasproef/kanalen | 85 × 60 × 8 | ✅ | ✅ 0 | 365 | 0 | ✅ geen | 0 |
| gedeeld/pasproef/nl/proefkaartjes | 134 × 28 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 |
| gedeeld/pasproef/sleuven | 111 × 24 × 22 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 0 |
| gedeeld/voet/voet | 206 × 72 × 36 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 0 |
| varianten/01-vos/en/front_fox | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 69 | ✅ geen | 12 |
| varianten/01-vos/nl/front_vos | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 68 | ✅ geen | 12 |
| varianten/02-koe/en/front_cow | 200 × 209 × 8 | ✅ | ✅ 0 | 821 | 72 | ✅ geen | 26 |
| varianten/02-koe/nl/front_koe | 200 × 209 × 8 | ✅ | ✅ 0 | 821 | 72 | ✅ geen | 26 |
| varianten/03-uil/en/front_owl | 200 × 206 × 8 | ✅ | ✅ 0 | 821 | 71 | ✅ geen | 25 |
| varianten/03-uil/nl/front_uil | 200 × 206 × 8 | ✅ | ✅ 0 | 821 | 64 | ✅ geen | 25 |
| varianten/04-draak/en/front_dragon | 200 × 209 × 8 | ✅ | ✅ 0 | 821 | 94 | ✅ geen | 26 |
| varianten/04-draak/nl/front_draak | 200 × 209 × 8 | ✅ | ✅ 0 | 821 | 86 | ✅ geen | 26 |
| varianten/05-eenhoorn/en/front_unicorn | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 90 | ✅ geen | 26 |
| varianten/05-eenhoorn/nl/front_eenhoorn | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 104 | ✅ geen | 26 |
| varianten/06-leeuw/en/front_lion | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 71 | ✅ geen | 26 |
| varianten/06-leeuw/nl/front_leeuw | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 85 | ✅ geen | 26 |
| varianten/07-hert/en/front_deer | 200 × 197 × 8 | ✅ | ✅ 0 | 821 | 77 | ✅ geen | 26 |
| varianten/07-hert/nl/front_hert | 200 × 197 × 8 | ✅ | ✅ 0 | 821 | 75 | ✅ geen | 26 |

45°-vlakken zijn de bewust schuine lippen, afschuiningen en afloop van de geleiders: printbaar zonder support. Korte bruggen zijn de verzonken letters van het versielabel aan de achterkant (op het bed).

### Slicer (PrusaSlicer 2.7, A1-achtig profiel)

Bambu Studio draait hier niet; het gewicht is betrouwbaar, de tijd en het aantal wissels zijn een indicatie. Engelse fronten verschillen alleen in de labels en zijn niet apart geslicet.

| Onderdeel | Filament (g) | Tijd (indicatie) | Support nodig | Wissels |
|---|---|---|---|---|
| fanart/08-mario/nl/front_mario | 95,1 | 2h 53m 58s | ✅ nee | 53 |
| fanart/09-sonic/nl/front_sonic | 97,9 | 2h 55m 25s | ✅ nee | 48 |
| fanart/10-spiderman/nl/front_spiderman | 89,5 | 2h 51m 5s | ✅ nee | 46 |
| fanart/11-steve/nl/front_steve | 91,3 | 2h 28m 3s | ✅ nee | 38 |
| fanart/12-creeper/nl/front_creeper | 92,3 | 2h 35m 37s | ✅ nee | 38 |
| gedeeld/kaartjes/cijfers | 18,4 | 36m 41s | ✅ nee | 4 |
| gedeeld/kaartjes/en/days | 23,0 | 41m 39s | ✅ nee | 4 |
| gedeeld/kaartjes/en/months | 52,1 | 1h 24m 50s | ✅ nee | 4 |
| gedeeld/kaartjes/nl/dagen | 23,0 | 42m 9s | ✅ nee | 4 |
| gedeeld/kaartjes/nl/maanden | 52,1 | 1h 24m 46s | ✅ nee | 4 |
| gedeeld/pasproef/en/test-cards | 8,0 | 16m 22s | ✅ nee | 4 |
| gedeeld/pasproef/kanalen | 13,2 | 23m 34s | ✅ nee | 0 |
| gedeeld/pasproef/nl/proefkaartjes | 8,0 | 16m 17s | ✅ nee | 4 |
| gedeeld/pasproef/sleuven | 23,5 | 33m 42s | ✅ nee | 0 |
| gedeeld/voet/voet | 107,8 | 2h 17m 44s | ✅ nee | 0 |
| varianten/01-vos/nl/front_vos | 90,5 | 2h 22m 20s | ✅ nee | 21 |
| varianten/02-koe/nl/front_koe | 92,1 | 2h 34m 34s | ✅ nee | 41 |
| varianten/03-uil/nl/front_uil | 91,7 | 2h 32m 34s | ✅ nee | 40 |
| varianten/04-draak/nl/front_draak | 91,3 | 2h 34m 46s | ✅ nee | 42 |
| varianten/05-eenhoorn/nl/front_eenhoorn | 94,3 | 2h 46m 50s | ✅ nee | 60 |
| varianten/06-leeuw/nl/front_leeuw | 92,6 | 2h 34m 41s | ✅ nee | 42 |
| varianten/07-hert/nl/front_hert | 89,2 | 2h 31m 18s | ✅ nee | 44 |

Support is gecontroleerd met automatische support bij een drempel van 30° (Bambu-standaard).

![Printplaten](../renders/printplaten.png)

![Fronten](../renders/fronten.png)

## 5. Bambu Studio-projecten

Elk .3mf-bestand is gecontroleerd op de regels van de Bambu Studio-lezer (bbs_3mf.cpp): verplichte bestanden, id's van objecten en delen, filamentslots, lijsten per filament, spoelmatrix, en of elk object volledig op zijn eigen plaat ligt zonder de spoeltoren te raken. Niet getest in Bambu Studio zelf.

- ✅ `fanart/08-mario/en/animal-calendar_mario_en.3mf`
- ✅ `fanart/08-mario/nl/dierenkalender_mario_nl.3mf`
- ✅ `fanart/09-sonic/en/animal-calendar_sonic_en.3mf`
- ✅ `fanart/09-sonic/nl/dierenkalender_sonic_nl.3mf`
- ✅ `fanart/10-spiderman/en/animal-calendar_spiderman_en.3mf`
- ✅ `fanart/10-spiderman/nl/dierenkalender_spiderman_nl.3mf`
- ✅ `fanart/11-steve/en/animal-calendar_steve_en.3mf`
- ✅ `fanart/11-steve/nl/dierenkalender_steve_nl.3mf`
- ✅ `fanart/12-creeper/en/animal-calendar_creeper_en.3mf`
- ✅ `fanart/12-creeper/nl/dierenkalender_creeper_nl.3mf`
- ✅ `gedeeld/kaartjes/en/cards_en.3mf`
- ✅ `gedeeld/kaartjes/nl/kaartjes_nl.3mf`
- ✅ `gedeeld/pasproef/en/test-fit_en.3mf`
- ✅ `gedeeld/pasproef/nl/pasproef_nl.3mf`
- ✅ `gedeeld/voet/voet.3mf`
- ✅ `varianten/01-vos/en/animal-calendar_fox_en.3mf`
- ✅ `varianten/01-vos/nl/dierenkalender_vos_nl.3mf`
- ✅ `varianten/02-koe/en/animal-calendar_cow_en.3mf`
- ✅ `varianten/02-koe/nl/dierenkalender_koe_nl.3mf`
- ✅ `varianten/03-uil/en/animal-calendar_owl_en.3mf`
- ✅ `varianten/03-uil/nl/dierenkalender_uil_nl.3mf`
- ✅ `varianten/04-draak/en/animal-calendar_dragon_en.3mf`
- ✅ `varianten/04-draak/nl/dierenkalender_draak_nl.3mf`
- ✅ `varianten/05-eenhoorn/en/animal-calendar_unicorn_en.3mf`
- ✅ `varianten/05-eenhoorn/nl/dierenkalender_eenhoorn_nl.3mf`
- ✅ `varianten/06-leeuw/en/animal-calendar_lion_en.3mf`
- ✅ `varianten/06-leeuw/nl/dierenkalender_leeuw_nl.3mf`
- ✅ `varianten/07-hert/en/animal-calendar_deer_en.3mf`
- ✅ `varianten/07-hert/nl/dierenkalender_hert_nl.3mf`

## 6. Stabiliteit

Vulgraad: slicer {'voet': 0.49, 'front': 0.64, 'kaart': 1.03}.

| Variant | Gewicht (g) | Zwaartepunt hoogte (mm) | Kantelt voorover bij | Kantelt achterover bij |
|---|---|---|---|---|
| 01 vos | 290 | 48 | 37° | 36° |
| 02 koe | 288 | 47 | 38° | 37° |
| 03 uil | 287 | 47 | 38° | 37° |
| 04 draak | 286 | 46 | 38° | 38° |
| 05 eenhoorn | 284 | 45 | 39° | 38° |
| 06 leeuw | 288 | 47 | 38° | 37° |
| 07 hert | 282 | 45 | 39° | 39° |
| 08 mario | 287 | 47 | 38° | 37° |
| 09 sonic | 292 | 49 | 37° | 36° |
| 10 spiderman | 281 | 44 | 39° | 39° |
| 11 steve | 287 | 48 | 38° | 37° |
| 12 creeper | 289 | 48 | 37° | 37° |
