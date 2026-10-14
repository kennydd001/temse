# H-03 Poop Monster (v0.1)

*Nederlands: [README.md](README.md)*

A pumpkin monster that **eats your A1's poop**: the blobs of filament the printer spits out at every colour
change. Put it where the poop falls out of the printer. It catches them with its teeth, and through the stitches
in its belly you see how much rainbow it has eaten already.

![Poop Monster](renders/voor.png)

> **Status:** CAD, STLs, the Bambu Studio project and the checks are done. **Nothing has been printed yet**,
> and it has not stood next to a real A1. The pictures are renders with simulated poop.
>
> The 3MF project has been checked against the rules of the Bambu format (`gereedschap/bambu/valideer.py` in
> the root of the repo). It has not been opened in Bambu Studio itself: that program does not run in this
> environment.

| | |
|---|---|
| Variants | **low** (laag): rim at the back 40 mm (tips 48), at the front 75 mm · **high** (hoog): back 65 mm (tips 73), front 100 mm · or a **custom height** |
| Size | Ø 116 mm, 83 or 108 mm tall. Fits on the A1 mini |
| Capacity | ~300 cm³ (low) or ~530 cm³ (high), up to the back rim |
| Filament | about 48 g (low) or 68 g (high), estimated. One colour, no AMS needed |
| You need | nothing: one piece, no supports, screws or glue |

## How it works

- A ribbed bin, like a pumpkin, with a 2 mm wall.
- The top is **cut at a slant**: low at the back, high at the front.
  - Put the low side towards the printer. That is where the poop falls in.
  - Whatever flies too far bounces off the high front and still drops in.
- **24 teeth** of 8 mm run all the way round.
- At the front, two **eyes**, a big one and a small one. At the bottom of each eye sits a pupil, cross-eyed
  towards the nose. It rests on the edge of the eye, so it prints without supports too.
- The **belly** is a slit sewn shut with vertical **stitches**.
  - The gaps between them are 6.6 mm wide: narrower than a poop blob, so it stays inside.
  - The stitches also carry the flat ceiling of each gap: a short bridge.

![From above: the mouth with the teeth](renders/boven.png)

## Placing it

1. After a print with a few colour changes, look **where the poop lands on the table**: that is where the mouth
   goes.
2. Measure how high the opening is where the poop comes out of the printer.
   - The **tips of the teeth at the back** must be at least 1 cm lower: 48 mm for *low*, 73 mm for *high*.
   - If neither fits, make a custom height (below).
3. Put the monster with its **back to the printer** and its face towards you.
4. Full? Lift it and empty it.

## Printing

Open `print/poepmonster.3mf` in Bambu Studio. There is one plate per variant:

| Plate | Variant | Colour in the project | Filament (estimated) |
|---|---|---|---|
| 1 | low | Pumpkin Orange 10301 | ~48 g |
| 2 | high | Pumpkin Orange 10301 | ~68 g |

- **Settings**: *0.20mm Standard @BBL A1* on the *Textured PEI Plate*, 0.4 mm nozzle. **No supports.** The
  project sets 3 walls and gyroid infill.
- It **stands upright**, bottom on the bed.
- **Colour**: any colour works. Orange makes a pumpkin, purple or green makes a monster.

| File | Size (mm) |
|---|---|
| `print/poepmonster_laag.stl` (low) | 116 × 116 × 83 |
| `print/poepmonster_hoog.stl` (high) | 116 × 116 × 108 |

## Custom height

```
cd cad
python bouw.py eigen 50            # rim at the back 50 mm (tooth valleys), front 35 mm higher
python bouw.py eigen 50 90         # back 50, front 90
```

- Give the height of the **tooth valleys**; the tips are 8 mm higher.
- At least 25 mm at the back. The front has to be high enough for the eyes and the belly; the script tells
  you if it is not.
- The STL goes to `print/poepmonster_eigen_<back>.stl`. The script also tells you how much it holds.

## Checks

See [controle/RAPPORT.md](controle/RAPPORT.md) (in Dutch). In short:

- **Face**: making the holes self-supporting takes about 31 mm² off the drawn holes (the pointed tops of the
  eyes). The gaps in the belly are 6.6 mm wide, the stitches 2.4 mm.
- **Printability**: one piece and watertight, also as an STL. There are no overhangs steeper than 48° and no
  details thinner than the nozzle.

![Face](controle/gezicht.png)

## Folders

```
print/        poepmonster.3mf (2 plates) and one STL per variant
cad/          the source: params.py (variants and sizes), gezicht.py (eyes and belly), monster.py (the bin with
              the teeth), bouw.py, controle.py, render.py
controle/     RAPPORT.md, controle.json, gezicht.png
renders/      Blender renders: voor.png, boven.png, varianten.png
onderdelen.json   volume, size, rim heights and capacity per STL
```

```
cd cad
python bouw.py            # print/
python controle.py        # controle/
python render.py alle     # renders/ (needs the pip module bpy)
```

## MakerWorld

- **Original design**: the monster is our own drawing.
- The strongest picture is a **timelapse**: the monster next to the printer, and at every colour change a blob
  flies into its mouth. Film it with the real print; MakerWorld does not allow AI images as model photos.
