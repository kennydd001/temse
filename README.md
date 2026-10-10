# 3D-printmodellen · 3D print models

Eigen ontwerpen voor de Bambu Lab A1 met AMS Lite, gemaakt als code (parametrisch) en klaar voor MakerWorld.
*Original designs for the Bambu Lab A1 with AMS Lite, built as code (parametric) and ready for MakerWorld.*

| Model | Varianten · variants | Talen · languages |
|---|---|---|
| [C-01 Dierenkalender · animal calendar](modellen/C-01-dierenkalender/) | vos, koe, uil, draak, eenhoorn, leeuw, hert · fox, cow, owl, dragon, unicorn, lion, deer | NL, EN |

![C-01 familie](modellen/C-01-dierenkalender/renders/familie.png)

## Structuur · layout

```
modellen/
  C-01-dierenkalender/
    README.md, README.en.md     uitleg · documentation
    varianten/NN-dier/nl|en/    Bambu Studio-project (.3mf), front-STL's per kleur, productfoto
    gedeeld/                    herbruikbaar · reusable: voet/base, kaartjes/cards, pasproef/test fit
    fanart/                     bestaande personages, alleen voor eigen gebruik · existing characters, personal use only
    cad/                        de bron: Python-scripts die alles genereren · the source
    controle/                   montage- en printcontrole · assembly and print check
    renders/                    Blender-renders
```

Nieuwe modellen komen als eigen map onder `modellen/`. Een nieuw dier voor C-01 is één bestand in
`modellen/C-01-dierenkalender/cad/dieren/`, en daarna `python bouw.py` draaien.
*New models get their own folder under `modellen/`. A new C-01 animal is one file in `cad/dieren/` plus `python bouw.py`.*
