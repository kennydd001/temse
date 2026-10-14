# E-01 Splitsmijn: pasproef van het valbord (v0.1)

*English: [README.en.md](README.en.md)*

Rekenmachine om te leren **splitsen tot 10**, voor het 1ste leerjaar (5 tot 6 jaar). Het kind gooit een
handvol diamantmuntjes in de mijn. Ze tikken langs de pinnen naar beneden en vallen links of rechts in een
bakje. Dan zegt het kind de splitsing: *8 is 3 en 5*. In het eindmodel is één bakje een dichte kist:
*8 = 3 + ?*. Eerst raden, dan de kist openen en tellen.

Dit is de **pasproef**: het valbord op ware grootte, met de slinger, een eenvoudige voet met twee open bakjes
en 10 muntjes. Ze test de twee dingen die het concept kunnen breken:

1. **Blijft er ooit een muntje hangen?**
2. **Zijn de splitsingen echt willekeurig?** Anders weet het kind na een paar keer al waar de muntjes heen gaan.

![Splitsmijn pasproef](renders/voor.png)

> **Status:** CAD, STL's, het Bambu Studio-project en de controle zijn klaar. Er is **nog niets echt
> geprint**. Begin met de passtrip (plaat 1), dan de rest, en volg *Testen* hieronder.
>
> Het 3MF-project is gecontroleerd tegen de regels van het Bambu-formaat (`gereedschap/bambu/valideer.py` in
> de root van de repo). Het is hier niet in Bambu Studio zelf geopend: dat programma draait niet in deze
> omgeving.

| | |
|---|---|
| Maat (b × d × h) | 226 × 48 × 262 mm, met de voet |
| Muntjes | 10 stuks Ø 25 × 4,5 mm (zo groot als 2 euro), met een diamant erin gegraveerd |
| Nodig | niets: geen schroeven, lijm of metalen onderdelen |

Kleuren (Bambu PLA, uit de kleurplanning):

- **Charcoal 11101**: achter- en voorplaat. Op donker zie je de muntjes goed.
- **Latte Brown 11800**: de voet en de beugel.
- **Maroon Red 10205**: de slinger, het mesblok en de kantelplank: alles wat beweegt.
- **Marine Blue 11600**: de muntjes.

## Hoe het werkt

![Zonder voorplaat](renders/binnen.png)

- **Valbord**: de muntjes vallen rechtop in een spleet van 6 mm tussen de achter- en voorplaat.
  - Er staan vier rijen van 3 of 4 pinnen in een driehoeksraster, 44 mm uit elkaar. Tussen twee pinnen, of
    tussen een pin en de wand, is het overal minstens 29 mm, voor een muntje van 25 mm.
  - Naast de rijen met een middenpin zit een halve pin in de wand, zodat geen muntje langs de wand naar
    beneden glijdt.
  - Onderaan staat de verdeler recht onder de middelste pin. Wat links ervan valt, komt in het linkerbakje.
- **Waarom een slinger**: een muntje van 25 mm is groot naast pinnen van 44 mm. Het botst niet willekeurig
  maar rolt in een vaste baan over de pinnen. In de simulatie ging een gewoon valbord 65 tot 95 % van de keren
  naar de kant waar het kind mikte. Dan leert een kind mikken, geen splitsen.
  - Daarom valt elk muntje eerst op een **kantelplank** onder de inworp. Die plank zit vast aan een zware
    **slinger** voor het bord.
  - Het kind geeft de slinger één duw. Wat er dan beslist, is welke kant van de plank op dat moment lager
    ligt. Het muntje is veel te licht om de slinger te storen.
  - De inworp is maar 2 mm breder dan een muntje: hoe minder ruimte om te mikken, hoe minder het mikpunt
    meetelt.
- **Mes in plaats van as**: een geprinte as van 4 mm remt de slinger zo hard af dat hij na 10 tot 20 seconden
  stilhangt.
  - Daarom hangt de slinger op een **mes** in een V-groef, zoals een ouderwetse balans. Een halve minuut na
    een duw van 35° slingert hij nog met 27 tot 31°.
  - De groef is stomp (150°): het mes kan 45° kantelen voor zijn flanken de groef raken, en de slinger slaat
    hooguit 35° uit.
  - De groef loopt niet tot de voorkant door. Dat randje houdt het mes tegen, zodat de slinger niet van zijn
    zeskant kan schuiven.

![Het mes in de groef](renders/mes.png)

### Wat de controle zegt

De controle bouwt het valbord na in een 2D-fysicamodel. Een ronde gaat zoals in de klas: één duw, dan
8 muntjes met 0,8 tot 2 seconden ertussen. Dat is doorgerekend voor 12 combinaties van stuit, wrijving en
demping, want PLA op PLA is niet precies bekend. Alles staat in [controle/RAPPORT.md](controle/RAPPORT.md).

- **Vastlopen**: 0 van 4608 muntjes bleven hangen. Er is ook nergens een plek waar een muntje op twee
  steunpunten kan blijven liggen.
- **Mikken**: links mikken geeft hooguit 4 procentpunten meer links dan rechts mikken. Zonder slinger is dat
  30 tot 90 procentpunten.
- **Splitsingen** van 8: ze volgen de kansen van een eerlijk muntstuk. 4 + 4 komt het vaakst voor, 0 + 8 haast
  nooit.
- **Slinger**: na het laatste muntje van een ronde slingert hij nog met zo'n 29°. Eén duw is genoeg voor een
  hele ronde.

![Controle](controle/valbord.png)

## Onderdelen en printen

Open `pasproef/pasproef.3mf` in Bambu Studio. Het project heeft zes platen, elk in één kleur:

| Plaat | Wat | Kleur | Filament* | Tijd* |
|---|---|---|---|---|
| 1 | **passtrip: eerst printen**, zie hieronder | Charcoal 11101 | 12 g | 0 u 32 |
| 2 | achterplaat met de pinnen | Charcoal 11101 | 138 g | 3 u 16 |
| 3 | voorplaat met kijkspleten | Charcoal 11101 | 109 g | 3 u 02 |
| 4 | voet met de bakjes, beugel | Latte Brown 11800 | 74 g | 1 u 36 |
| 5 | slinger, mesblok, kantelplank | Maroon Red 10205 | 46 g | 0 u 54 |
| 6 | 10 muntjes | Marine Blue 11600 | 17 g | 0 u 32 |

Samen is dat ongeveer **395 g filament en bijna 10 uur printen**. Dat is geen proefstukje voor één avond, maar
het valbord kan alleen op ware grootte getest worden: de muntjes, de pinnen en de slinger moeten echt zijn.

\* Gewicht en tijd komen uit PrusaSlicer met een profiel dat de A1 benadert, zie
[controle/RAPPORT.md](controle/RAPPORT.md). Het gewicht is betrouwbaar; de tijd rekent Bambu Studio zelf uit.

- **Instellingen**: *0.20mm Standard @BBL A1* op de *Textured PEI Plate*, 0,4 mm nozzle, **geen support**,
  gyroid-infill. Per object staat dit al in het project:
  - achterplaat en voet: 3 wanden; voorplaat: 2 wanden;
  - slinger: 3 wanden en **100 % infill**, rechtlijnig (gyroid kan niet op 100 %): een zwaar gewicht slingert
    langer;
  - kantelplank: 4 wanden; muntjes: 3 wanden, zonder brim.
- **Printligging**:
  - Achter- en voorplaat liggen plat. De achterplaat ligt met de achterkant op het bed, de pinnen omhoog. De
    voorplaat ligt met de binnenkant op het bed.
  - De slinger ligt met de achterkant van de arm op het bed: de pennetjes en het gewicht wijzen omhoog.
  - Het mesblok ligt op zijn achterkant (het mes staat rechtop). De beugel staat op zijn poten. De plank ligt
    met de naaf omhoog.
- `pasproef/voorplaat_glad.stl` is dezelfde voorplaat zonder kijkspleten. Die is voor doorzichtig filament
  (PETG of transparante PLA), als je de muntjes helemaal wilt zien vallen.

| Onderdeel | Aantal | Maat (mm) | Bestand |
|---|---|---|---|
| achterplaat met pinnen en zijranden | 1 | 218 × 230 × 10 | `pasproef/achterplaat.stl` |
| voorplaat | 1 | 218 × 230 × 2,5 (met pennen 13,5) | `pasproef/voorplaat.stl` |
| voet met twee bakjes en verdeler | 1 | 226 × 48 × 52 | `pasproef/voet.stl` |
| beugel met V-groef | 1 | 44 × 14 × 13 | `pasproef/beugel.stl` |
| slinger (arm en gewicht) | 1 | 44 × 184 × 17 | `pasproef/slinger.stl` |
| mesblok | 1 | 20 × 16 × 6,5 | `pasproef/mesblok.stl` |
| kantelplank met naaf en zeskant | 1 | 36 × 10 × 13 | `pasproef/plank.stl` |
| diamantmuntje | 10 | Ø 25 × 4,5 | `pasproef/munt.stl` |
| passtrip: gatenstrip, pennenblok, beugelproef | 3 | 58 × 34 × 2,5; 50 × 12 × 10,5; 42 × 12 × 11 | `pasproef/gatenstrip.stl` enz. |

## Eerst de passtrip

![Passtrip](renders/passtrip.png)

Er zitten geen klikhaakjes in: alles **klemt** of **schuift** in elkaar. Hoe strak dat wordt, hangt af van je
printer. Een gaatje komt meestal 0,1 tot 0,2 mm kleiner uit dan in het CAD, en bij de 26 nokjes van de
voorplaat maakt dat veel uit: te strak, en je krijgt de voorplaat er niet op (of weer af).

Print daarom eerst plaat 1 (een halfuurtje) en probeer de stukjes in elkaar:

| Probeer | In | Kies het gat waar het | Zet in `cad/params.py` | Nu |
|---|---|---|---|---|
| nokje van het pennenblok | ronde gaten 3,0 tot 3,4 | met lichte duimdruk in gaat en blijft zitten, zonder forceren | `NOK_GAT` | 3,2 |
| hetzelfde nokje | ronde gaten 3,0 tot 3,4 | stevig klemt (een maat strakker mag) | `MES_GAT` | 3,1 |
| zeskant van het pennenblok | zeskantgaten 8,1 tot 8,4 | vlot schuift zonder te wiebelen | `ZESKANT_SPEL` = maat − 8 | 0,2 |
| pen Ø 4 van het pennenblok | gaten 4,0 tot 4,3 van de beugelproef | stevig klemt | `BEUGEL_GAT` | 4,1 |

Klopt alles met wat er nu staat, dan hoef je niets te doen. Anders: pas de waarden aan en draai
`python bouw.py` in `cad/` opnieuw. Dan pas print je platen 2 tot 6.

Wat waar klemt en hoeveel speling elke verbinding in het CAD heeft, staat in
[controle/RAPPORT.md](controle/RAPPORT.md), onder *Verbindingen*.

## In elkaar zetten

![Explosietekening](renders/explosie.png)

1. **Achterplaat**: leg ze plat op tafel, met de pinnen omhoog.
2. **Kantelplank**: leg de plank in het midden onder de inworp, met de naaf omhoog. Ze ligt tussen de
   bovenste pinnen en het plafond.
3. **Voorplaat**: leg ze erop, met de kijkspleten naar je toe. Steek de naaf door het ronde gat.
   - Druk de voorplaat rondom aan: de 26 nokjes op de pinnen en de randen klemmen licht in de gaatjes.
   - De plank moet nu vrij kunnen kantelen, de naaf raakt het gat nergens.
4. **Slinger**: druk het mesblok op de twee pennetjes bovenaan de arm (dat klemt), met het mes naar beneden.
   Schuif dan de arm met zijn zeskantgat op de zeskant van de plank.
5. **Beugel**: schuif de beugel met zijn twee gaten op de pennen onder de naaf (dat klemt), met de poten naar
   de voorplaat.
   - Til het gewicht een klein beetje op terwijl je schuift. De naaf heeft 1 mm speling in het gat, en zo
     glijdt het mes van achteren in de groef.
   - Het randje vooraan de groef houdt het mes nu tegen.
6. **Voet**: zet het bord rechtop in de voet. De zijranden zakken in de kokers, de vin van de verdeler steekt
   onderaan in de spleet.

## Testen

Geef de slinger altijd eerst een duw. Staat hij stil, dan ligt de plank vlak en beslist het mikpunt weer.

- [ ] **Slingeren**: duw de slinger tot zo'n 30° en tel hoe lang hij beweegt. Na 30 seconden moet hij nog
      duidelijk slingeren.
  - Valt hij snel stil, kijk dan of het mes mooi in de groef zit en of de naaf of de plank nergens schuurt.
- [ ] **Vastlopen**: gooi 10 rondes van 10 muntjes. Er mag er geen enkel blijven hangen.
  - Hangt er toch een, noteer waar. Klemt het muntje tussen de platen, maak dan `SPLEET` in `cad/params.py`
    0,5 mm groter.
- [ ] **Inworp**: past elk muntje vlot door de inworp? Zo niet, zet `INWORP_HALF` 0,5 mm groter.
- [ ] **Eerlijk**: gooi 40 muntjes en mik ze allemaal zo ver mogelijk links in de trechter. Doe daarna
      hetzelfde rechts.
  - Tel telkens hoeveel er links uitkomen. Bij een eerlijk bord ligt dat beide keren tussen 14 en 26.
  - Komt het bij links mikken veel vaker links uit, dan telt het mikpunt nog mee.
- [ ] **Met een kind**: laat een kleuter of een kind uit het 1ste leerjaar spelen.
  - Snapt het de duw en de inworp zonder uitleg?
  - Kan het de muntjes uit de bakjes halen?
  - Hoe lang blijft het spelen?

Draai na elke aanpassing `python bouw.py` opnieuw. Werkt de pasproef, dan komt het eindmodel: een
mijnschacht met een dichte kist voor het tweede bakje en een tienveld om de splitsing op te leggen.

## In de klas (ijsbergmodel)

Het valbord is het concrete niveau van de ijsberg: het kind ziet en voelt de muntjes zich splitsen.

1. **Concreet**: gooi bijvoorbeeld 8 muntjes en tel ze in elk bakje. *8 is 3 en 5.*
2. **Schematisch**: leg de muntjes op een tienveld, of vul een splitsbeen in.
3. **Abstract**: schrijf *8 = 3 + 5*. Wat is dan *5 + 3*?
4. **Kist dicht**: houd één bakje bedekt, met een bekertje of je hand. Hoeveel zitten er links? Dan
   *8 = 3 + ?*. Eerst raden, dan kijken.

Omdat de splitsing elke keer anders is, oefent het kind op den duur alle splitsingen van hetzelfde getal.

## Mappen

```
pasproef/     pasproef.3mf (6 platen, plaat 1 is de passtrip) en elk onderdeel als STL in printligging
cad/          de bron: Python-scripts die alles hierboven maken (params.py heeft alle maten)
controle/     RAPPORT.md: muntdoorgang, valsimulatie, slinger, spelingen, verbindingen, printbaarheid, slicer
renders/      Blender-renders
onderdelen.json   volume en maat per STL
```

```
cd cad
python bouw.py            # pasproef/
python slice.py           # controle/slicer.json (vereist PrusaSlicer)
python controle.py        # controle/ (--snel voor een kortere simulatie)
python simulatie.py       # alleen de valsimulatie
python render.py alle     # renders/ (vereist de pip-module bpy)
```

## MakerWorld

Het concept leent bij een bekend blokjesspel en bij het valbord uit een tv-spelletje. Gebruik die namen
niet in de titel, de tags of de beschrijving. Het model heet **De Splitsmijn**, en de diamantjes zijn een
eigen tekening.
