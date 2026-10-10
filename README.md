# 3D-printmodellen · 3D print models

Eigen ontwerpen voor de Bambu Lab A1 met AMS Lite, gemaakt als code (parametrisch) en klaar voor MakerWorld.
*Original designs for the Bambu Lab A1 with AMS Lite, built as code (parametric) and ready for MakerWorld.*

| Model | Varianten · variants | Talen · languages |
|---|---|---|
| [C-01 Dierenkalender · animal calendar](modellen/C-01-dierenkalender/) | vos, koe, uil, draak, eenhoorn, leeuw, hert · fox, cow, owl, dragon, unicorn, lion, deer | NL, EN |
| [B-34B Repair Pod](modellen/B-34B-repair-pod/) | werkplek die openklapt met twee horizontale schroevenbakjes · workspace that unfolds with two level screw trays | NL, EN |
| [B-34A Taakvreter · Task Eater](modellen/B-34A-taakvreter/) | bureaumonster dat je afgewerkte taakbriefjes opeet · desk monster that eats your finished task notes | NL, EN |
| [E-01 Splitsmijn · Split Mine](modellen/E-01-splitsmijn/) | pasproef: valbord om te leren splitsen tot 10 (1ste leerjaar) · test print: drop board for number bonds up to 10 (first grade) | NL, EN |

![C-01 familie](modellen/C-01-dierenkalender/renders/familie.png)

![B-34B dicht, half open, open](modellen/B-34B-repair-pod/renders/reeks.png)

<img src="modellen/B-34A-taakvreter/renders/voor.png" width="420" alt="B-34A Taakvreter"> <img src="modellen/E-01-splitsmijn/renders/voor.png" width="340" alt="E-01 Splitsmijn">

## Structuur · layout

```
gereedschap/                    gedeelde code · shared code: geometrie, tandwielen, Bambu-3MF, slicer, printcontrole, Blender
modellen/
  B-34A-taakvreter/             zelfde opbouw als B-34B · same layout as B-34B
  B-34B-repair-pod/
    README.md, README.en.md     uitleg · documentation
    print/                      Bambu Studio-project (.3mf) en STL's · project and STLs
    pasproef/                   eerst printen · print this first
    cad/, controle/, renders/   bron, controle, foto's · source, checks, renders
  C-01-dierenkalender/
    README.md, README.en.md     uitleg · documentation
    varianten/NN-dier/nl|en/    Bambu Studio-project (.3mf), front-STL's per kleur, productfoto
    gedeeld/                    herbruikbaar · reusable: voet/base, kaartjes/cards, pasproef/test fit
    fanart/                     bestaande personages, alleen voor eigen gebruik · existing characters, personal use only
    cad/                        de bron: Python-scripts die alles genereren · the source
    controle/                   montage- en printcontrole · assembly and print check
    renders/                    Blender-renders
  E-01-splitsmijn/              voorlopig alleen pasproef/ (valbord), met valsimulatie · for now only pasproef/ (drop board), with drop simulation
```

Nieuwe modellen komen als eigen map onder `modellen/`. Een nieuw dier voor C-01 is één bestand in
`modellen/C-01-dierenkalender/cad/dieren/`, en daarna `python bouw.py` draaien.
*New models get their own folder under `modellen/`. A new C-01 animal is one file in `cad/dieren/` plus `python bouw.py`.*
