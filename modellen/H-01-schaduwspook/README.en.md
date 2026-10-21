# H-01 Shadow Spook (v0.1)

*Nederlands: [README.md](README.md)*

One figure, four shadows. Put it on the table, shine your phone's flashlight on it in the dark, and a
**pumpkin** appears on the wall. Turn it an eighth of a turn: a **ghost**. Another eighth: a **skull**. One
more: a **tombstone**. The figure itself looks like none of the four.

![Four shadows](renders/schaduwen.png)

> **Status:** CAD, STLs, the Bambu Studio project and the checks are done. **Nothing has been printed yet**,
> and the shadow has not been tried with a real flashlight. The pictures above are renders with a simulated
> point light.
>
> The 3MF project has been checked against the rules of the Bambu format (`gereedschap/bambu/valideer.py` in
> the root of the repo). It has not been opened in Bambu Studio itself: that program does not run in this
> environment.

| | |
|---|---|
| Variants | **four shadows**: pumpkin, ghost, skull and tombstone, every 45° · **BOO and skull** and **TRICK and TREAT**: two shadows, every 90° |
| Size | four shadows 56 × 52 × 60 mm · BOO 120 × 48 × 55 mm · TRICK 112 × 124 × 36 mm. Everything fits on the A1 mini |
| Filament | about 24 g · 63 g · 102 g (estimated) |
| You need | a flashlight (the one on your phone) and a light wall. No supports, screws or glue |

## How it works

- Every shadow is a **silhouette**. That silhouette is pushed straight through in the direction the light
  comes from, like a cookie cutter through a block of dough. The figure is the part that lies inside **all**
  of those pushed-through silhouettes at once.
- When the light shines along one silhouette's direction, you see exactly that silhouette. Every point inside
  it has something in the way, and next to the edge there is nothing.
- **Two shadows** at 90° always fit together exactly.
- **Four shadows** at 45° only fit if no shape is too wide for the diagonal directions.
  - That is why the pumpkin is drawn 15 % narrower.
  - What still gets lost is at most 2 %: a sliver at the top of the ghost and the tombstone.
  - The numbers per shadow are in [controle/RAPPORT.md](controle/RAPPORT.md).
- **Printing without supports**: every silhouette is made self-supporting first (the report lists every
  change). You can see a little of that in the shadow:
  - holes (eyes, mouth, teeth) get a pointed top, like a gothic arch;
  - an overhang gets a 40° wedge underneath;
  - anything hanging over a hole gets a thin pillar down to the floor (the skull's teeth).
- All silhouettes stand on a **base bar**. That makes the figure one piece, and it stands firmly.

![Silhouettes: as drawn and as printed](controle/silhouetten.png)

## Making the shadow

1. Darken the room. The wall should be light.
2. Put the figure **10 to 30 cm in front of the wall**.
3. Lay your phone with the flashlight on **flat on the table, 40 to 60 cm in front of the figure**. Aim the
   light straight at the figure.
   - The further away the light, the sharper the shadow.
   - A normal lamp or a big flashlight gives a blurry shadow: the light has to come from one small point.
4. Turn the figure **every 45°** (four shadows) or **every 90°** (BOO, TRICK).
   - Half a turn further on, the same sequence comes back, mirrored.
5. **For a video**: put the figure on a turntable or a spinning cake stand. The shadow then changes shape by
   itself.

## Printing

Open `print/schaduwspook.3mf` in Bambu Studio. There is one plate per variant:

| Plate | Variant | Colour in the project | Filament (estimated) |
|---|---|---|---|
| 1 | four shadows | Pumpkin Orange 10301 | ~24 g |
| 2 | BOO and skull | Charcoal 11101 | ~63 g |
| 3 | TRICK and TREAT | Pumpkin Orange 10301 | ~102 g |

- **Settings**: *0.20mm Standard @BBL A1* on the *Textured PEI Plate*, 0.4 mm nozzle. **No supports, no
  brim.** The project sets 3 walls and gyroid infill per figure.
- The figure **stands upright**, with the base bar on the bed.
- **Colour doesn't matter**: the shadow is always black. Orange or black looks best in daylight.

| File | Size (mm) |
|---|---|
| `print/schaduwspook_vier-schaduwen.stl` | 56 × 52 × 60 |
| `print/schaduwspook_boo-schedel.stl` | 120 × 48 × 55 |
| `print/schaduwspook_trick-treat.stl` | 112 × 124 × 36 |

## Making your own figure

```
cd cad
python bouw.py eigen smal:EMMA schedel 40         # your name from the front, a skull from the side, 40 mm tall
python bouw.py eigen kat spook pompoen*0.85 schedel 60
```

- Give 2 to 4 shapes. They are spread evenly over half a turn: 2 every 90°, 3 every 60°, 4 every 45°.
- **Shapes**: `schedel` (skull), `spook` (ghost), `pompoen` (pumpkin), `kat` (cat), `grafsteen` (tombstone),
  `tekst:WORD`, or `smal:WORD` (condensed letters, for longer words).
  - `*0.85` after a shape makes it narrower.
  - A number at the end is the height in mm (default 55). A 4-letter word at 40 mm is already 130 mm
    wide.
- The script reports how much of each shadow gets lost.
  - Words are wide: they work best with two shapes.
  - With four shapes, they should all be about the same width.
- The STL goes to `print/eigen_<shapes>.stl`.

Drawing a new shape happens in `cad/silhouetten.py`: a function that returns a shapely shape (bottom at y = 0),
plus a line in `VORMEN`.

## Checks

See [controle/RAPPORT.md](controle/RAPPORT.md) (in Dutch). In short:

- **Shadow**: measured row by row (0.2 mm) in every viewing direction. With two shadows nothing is missing.
  With four shadows at most 2 % is missing.
- **Printability**: every figure is one piece and watertight, also as an STL. There are no overhangs steeper
  than 48° and no details thinner than the nozzle.

## Folders

```
print/        schaduwspook.3mf (3 plates) and one STL per variant
cad/          the source: params.py (variants and sizes), silhouetten.py (the drawings), schaduw.py (the
              intersection and the self-supporting step), beeldje.py, bouw.py, controle.py, render.py
controle/     RAPPORT.md, controle.json, silhouetten.png
renders/      Blender renders with a point light and a wall
onderdelen.json   volume, size and shadow check per STL
```

```
cd cad
python bouw.py            # print/
python controle.py        # controle/
python render.py alle     # renders/ (needs the pip module bpy)
```

## MakerWorld

- **Original design**: the shapes are our own drawings; there is no one else's character in it.
- Shoot the cover photo and the video with the **real print and a real shadow**. MakerWorld does not allow
  AI images as model photos, and a real shadow on a wall is more convincing than a render.
