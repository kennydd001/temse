# H-02 Moodkin (v0.1)

*Nederlands: [README.md](README.md)*

A pumpkin lantern with four moods. Turn the stem: *click*, and the pumpkin looks **happy**, **angry**,
**scared**, or gives you a **wink**. Put an LED tealight inside and only the face at the front lights up.

![Four faces in the dark](renders/gezichten.png)

> **Status:** CAD, STLs, the Bambu Studio project and the checks are done. **Nothing has been printed yet**:
> the click, the pins and how bright the light comes through the holes have not been tested. The pictures are
> renders.
>
> The 3MF project has been checked against the rules of the Bambu format (`gereedschap/bambu/valideer.py` in
> the root of the repo). It has not been opened in Bambu Studio itself: that program does not run in this
> environment.

> ⚠️ **Only an LED tealight (Ø 38 mm), never a real candle.** PLA softens from about 55 °C: a candle melts the
> pumpkin and can start a fire.

| | |
|---|---|
| Faces | happy, angry, scared, wink (blij, boos, bang, knipoog): every 90°, with a click |
| Size | Ø 121 mm (lid), 116 mm tall with the stem. Everything fits on the A1 mini |
| Filament | about 166 g: shell 85 g and lid 48 g orange, drum 31 g yellow, stem 2 g green (estimated) |
| You need | one LED tealight (Ø 38 mm, up to 45 mm tall). No supports, screws, glue or AMS |

## How it works

- The **shell** has one fixed carved face: two pointed eye sockets, a triangular nose socket and a grin of
  five tooth columns.
- Behind it turns a yellow **drum** with four faces, every 90°. Through the sockets you see the yellow "flesh"
  with the holes of one face in it.
  - In daylight you see the face in yellow.
  - At night the light only comes through the holes of the face at the front. The other three are behind the
    closed wall of the shell (the checks confirm you can't see them, not even at an angle).
- The drum **hangs under the lid**: four pins press into four holes.
- The lid rests on the rim of the shell. The rim has **four bumps**, and the lid has four grooves underneath.
  - When you turn the lid, it rides up 0.8 mm on the bumps.
  - After a quarter turn it drops back with a **click**: the next face is at the front.
- The lid has the same ribs as the shell. One click is exactly two ribs, so the ribs line up in every
  position.

![Daylight](renders/dag.png)

## Assembly

1. Press the **drum** with its four pins into the holes under the **lid**. It goes on four ways; each one is
   fine.
2. Push the **stem** with its square pin into the lid.
3. Switch on the **LED tealight** and put it in the ring at the bottom of the shell.
4. Lower the lid and drum into the shell and turn until it clicks.

Turning clockwise (seen from above), the faces come in the order happy, angry, scared, wink.

![Open: this is how the light goes in](renders/open.png)

## Printing

Open `print/moodkin.3mf` in Bambu Studio. Every part has its own plate, in one colour:

| Plate | Part | Colour in the project | Filament (estimated) | On the bed |
|---|---|---|---|---|
| 1 | shell | Pumpkin Orange 10301 | ~85 g | upright |
| 2 | drum with the four faces | Sunflower Yellow 10402 | ~31 g | upright, pins on top |
| 3 | lid | Pumpkin Orange 10301 | ~48 g | flat underside on the bed |
| 4 | stem | Mistletoe Green 10502 | ~2 g | upside down, with a brim |

- **Settings**: *0.20mm Standard @BBL A1* on the *Textured PEI Plate*, 0.4 mm nozzle. **No supports.**
- The project sets the number of walls per part:
  - the drum 4, so its 1.6 mm wall is completely solid (no light through the infill);
  - shell, lid and stem 3.
- **No supports**: every hole is drawn so it prints upright.
  - Eyes and nose have a pointed top.
  - The tooth columns and the "scared" eyes have a flat ceiling of at most 12 mm. The printer bridges it.
- **Colour**: the drum can be a different colour. Lighter (yellow, white) shows the face best in daylight.
  Darker gives more contrast at night.

| File | Size (mm) |
|---|---|
| `print/moodkin_schil.stl` (shell) | 116 × 116 × 73 |
| `print/moodkin_trommel.stl` (drum) | 100 × 100 × 46 |
| `print/moodkin_deksel.stl` (lid) | 121 × 121 × 18 |
| `print/moodkin_steel.stl` (stem) | 18 × 13 × 31 |

## Making changes

- **Faces**: `cad/gezichten.py`. A face is a drawing in the plane (u = width, v = height), as seen from the
  front. It has to stay inside the sockets of the shell; the checks tell you whether it does.
- **Sizes**: `cad/params.py` (shell, drum, clearance, click, LED ring).

```
cd cad
python bouw.py            # print/
python controle.py        # controle/
python render.py alle     # renders/ (needs the pip module bpy)
```

## Checks

See [controle/RAPPORT.md](controle/RAPPORT.md) (in Dutch). In short:

- **Faces**: every face lies completely inside the sockets, with 0.6 mm to spare. The thinnest bit of wall
  between two holes is 3.3 mm in the drum and 1.5 mm in the shell.
- **Light**: through the sockets you see the drum up to 41° to the side at most. The nearest hole of another
  face is at 59°.
- **Fit**:
  - at least 1 mm clearance between drum and shell;
  - no overlap at all in 3D, in every position and halfway through a click;
  - 0.1 mm per side for the pins and 0.15 mm for the stem.
- **Printability**: every part is one piece and watertight. There are no overhangs steeper than 48° and no
  details thinner than the nozzle.

![Cross-section](controle/doorsnede.png)

## Folders

```
print/        moodkin.3mf (4 plates) and one STL per part
cad/          the source: params.py (sizes), gezichten.py (sockets and faces), lantaarn.py (the parts),
              bouw.py, controle.py, render.py
controle/     RAPPORT.md, controle.json, gezichten.png, doorsnede.png
renders/      Blender renders: nacht_<face>.png, gezichten.png, dag.png, open.png
onderdelen.json   volume and size per STL
```

## MakerWorld

- **Original design**: the faces and the mechanism are our own drawings.
- The strongest picture is a **video in the dark**: a hand turns the stem, *click*, and the face changes. Film
  it with the real print; MakerWorld does not allow AI images as model photos.
