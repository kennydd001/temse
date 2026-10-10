# Controle C-01 dierenkalender (vos)

Automatisch gegenereerd door `cad/controle.py` uit de STL's en de montage. Maten in mm. ✅ = in orde, ⚠️ = bekijken.

## 1. Botsingen in gemonteerde stand

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

## 3. Uitneemroute

Elk kaartje 0,05 mm van de plaat en vanaf 1 mm boven de draagrand omhoog geschoven tot de onderrand boven de geleiders is. De kleinste afstand onderweg is meestal de zijspeling langs de ribben.

| Kaartje | Slag | Kleinste afstand onderweg | Afstand tot het dier bovenaan |
|---|---|---|---|
| dag (ZATERDAG) | 19,5 | 0,50 | 4,22 |
| cijfer (1) | 19,5 | 0,50 | 3,80 |
| cijfer (0) | 19,5 | 0,50 | 8,86 |
| maand (OKTOBER) | 19,5 | 0,50 | 3,73 |

## 4. Printbaarheid (Bambu Lab A1, 256 × 256 mm, 0,4 mm nozzle, 0,2 mm laag)

| Plaat | Maat | Past | Overhang > 48° (mm²) | 45°-vlakken (mm²) | Korte bruggen < 3 mm (mm²) | Te smal (< 0,4 mm) | Lagen met kleurwissel | Wissels (schatting) |
|---|---|---|---|---|---|---|---|---|
| proef_kaartjes | 134 × 28 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 | 2 |
| proef_kanalen | 85 × 60 × 8 | ✅ | ✅ 0 | 365 | 0 | ✅ geen | 0 | 0 |
| proef_sleuven | 111 × 24 × 22 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 0 | 0 |
| front_vos | 200 × 210 × 8 | ✅ | ✅ 0 | 821 | 69 | ✅ geen | 12 | 17 |
| voet | 206 × 72 × 36 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 0 | 0 |
| kaartjes_cijfers | 178 × 60 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 | 2 |
| kaartjes_dagen | 236 × 60 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 | 2 |
| kaartjes_maanden | 236 × 124 × 2 | ✅ | ✅ 0 | 0 | 0 | ✅ geen | 2 | 2 |

45°-vlakken zijn de bewust schuine lippen, afschuiningen en afloop van de geleiders: printbaar zonder support. Korte bruggen zijn de verzonken letters van het versielabel aan de achterkant (op het bed).

### Slicer (PrusaSlicer 2.7, A1-achtig profiel)

Bambu Studio is hier niet beschikbaar; de tijden zijn een indicatie, het gewicht is betrouwbaarder.

| Plaat | Filament (g) | Tijd (indicatie) | Support nodig | Wissels |
|---|---|---|---|---|
| proef_kaartjes | 8,0 | 16m 17s | ✅ nee | 4 |
| proef_kanalen | 13,2 | 23m 34s | ✅ nee | 0 |
| proef_sleuven | 23,5 | 33m 42s | ✅ nee | 0 |
| front_vos | 90,5 | 2h 22m 29s | ✅ nee | 21 |
| voet | 107,8 | 2h 17m 44s | ✅ nee | 0 |
| kaartjes_cijfers | 18,4 | 36m 41s | ✅ nee | 4 |
| kaartjes_dagen | 23,0 | 42m 9s | ✅ nee | 4 |
| kaartjes_maanden | 52,1 | 1h 24m 47s | ✅ nee | 4 |

Support is gecontroleerd met automatische support bij een drempel van 30° (Bambu-standaard): de slicer vond nergens iets om te ondersteunen.

![Printplaten](../renders/printplaten.png)

## 5. Stabiliteit

- Totaalgewicht ongeveer 290 g (slicer).
- Zwaartepunt (x, y, z) = [4.7, 36.6, 48.0] mm; voet is 72 mm diep.
- Kantelt pas voorover bij 37° en achterover bij 36° scheefstand.

