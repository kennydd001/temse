# B-34A Task Eater (v0.1)

*Nederlands: [README.md](README.md)*

A desk monster following line **B-34A** of the MakerWorld build selection. Done with a task? Put the note in
its mouth and turn the crank: two rollers pull the note in, whole, and it drops into the drawer in its belly.
Meanwhile its eyes roll up and down. Turn back, and the note comes out again.

![Task Eater](renders/voor.png)

> **Status:** the CAD, STLs, Bambu Studio project and the assembly and print check are done. **Nothing has been
> printed yet.** Print the test fit first (`pasproef/pasproef.3mf`), see below.
>
> The 3MF projects are checked against the rules of the Bambu format (`gereedschap/bambu/valideer.py` at
> the root of the repo). They have not been opened in Bambu Studio itself: it does not run in this
> environment.

| | |
|---|---|
| Body (w × d × h) | 124 × 96 × 136 mm; 147 wide with the crank, 156 tall with the horns |
| Notes | 76 × 76 mm sticky notes (like Post-it), through an 82 × 4 mm slot |
| Drawer | 107 × 88 × 29 mm inside |
| You need | **3 O-rings 14 × 2 mm (NBR)**, or a rubber band wound a few times into each groove |

Colours (Bambu PLA, from the colour plan):

- **Pumpkin Orange 10301**: the body, the crank and the dowels.
- **Ivory White 11100**: the drawer, the eyes, the teeth, the knob and the horns.
- **Charcoal 11101**: the rollers, the eye mechanism, the mouthpiece and the pupils.

## How it works

![Mechanism](controle/mechanisme.png)

- **Crank and rollers**: the crank is on the right and drives the lower roller directly, through a hexagon.
  - The crank and its shaft are a single part.
  - Gears on the right side make the upper roller turn along, 1:1.
  - The lower roller carries three O-rings. They press 0.65 mm into the upper roller, which gives the rollers
    their grip on the paper. Even when the bearings take up their play (up to 0.4 mm together), at least
    0.25 mm is left.
- **Roller covering and shafts**: the covering is O-rings 14 × 2 mm (NBR). The shafts are printed: each roller
  has its own stub axle, and the crank has its own shaft. You don't need metal shafts.
- **Paper path**:
  - The note goes through a 4 mm high slot in the mouthpiece. It still fits up to 4.7° skewed.
  - The rollers are wider than the slot, so paper can't reach the gears.
  - Behind the rollers a chute bends the note downwards, through a 49 mm opening into the drawer.
- **Eyes**: a small 5 mm crank sits on the left end of the upper roller. Through a connecting rod and a
  rocker it turns the eye shaft back and forth.
  - The eye drums swing 54° and the pupils move about 6 mm up and down, once per turn.
  - The linkage turns all the way round without binding: the transmission angle stays at 58° or more.
- **Finger safe**:
  - No finger fits in the 4 mm slot, not even a child's. The pinch point is 35 mm deep.
  - A shelf under the rollers shields them from below.
  - **Only turn the crank with the drawer in.** Without the drawer a finger could reach the back of the
    rollers from below. They run apart there, so they don't pull anything in.

![Cross-section](renders/doorsnede.png)

## Parts

| Part | Qty | Colour | Size (mm) | File |
|---|---|---|---|---|
| front half (face, eye bearings, shelf, bracket) | 1 | Pumpkin Orange | 124 × 136 × 44 | `print/romp_voor.stl` |
| back half (chute, bracket) | 1 | Pumpkin Orange | 124 × 136 × 62 | `print/romp_achter.stl` |
| crank with shaft | 1 | Pumpkin Orange | 46 × 15 × 32 | `print/kruk.stl` |
| dowel Ø 5 × 16 | 8 + 2 spare | Pumpkin Orange | | `print/paspen.stl` |
| drawer | 1 | Ivory White | 111 × 93 × 31 | `print/lade.stl` |
| eyes (eye shaft with two drums) | 1 | Ivory White | Ø 26 × 93 | `print/ogen.stl` |
| teeth | 1 | Ivory White | 80 × 5 | `print/tanden.stl` |
| knob | 1 | Ivory White | Ø 13 × 22 | `print/knop.stl` |
| horn | 2 | Ivory White | Ø 16 × 20 | `print/hoorn.stl` |
| lower roller (with gear and O-ring grooves) | 1 | Charcoal | Ø 19 × 107 | `print/rol_onder.stl` |
| upper roller (with gear) | 1 | Charcoal | Ø 19 × 105 | `print/rol_boven.stl` |
| eye crank (crank disc for the eyes) | 1 | Charcoal | Ø 17 × 17 | `print/nokas.stl` |
| connecting rod | 1 | Charcoal | 41 × 7 | `print/drijfstang.stl` |
| rocker | 1 | Charcoal | 19 × 11 | `print/tuimelaar.stl` |
| mouthpiece (lip, mouth, feed tunnel) | 1 | Charcoal | 105 × 24 × 25 | `print/mondstuk.stl` |
| pupil | 2 | Charcoal | Ø 9 | `print/pupil.stl` |

That makes sixteen different parts: 25 printed pieces in the assembly, plus 3 O-rings.

## Printing with the Bambu Studio project

Open `print/taakvreter.3mf` in Bambu Studio. The project has four plates, one colour each:

| Plate | What | Colour | Filament* | Time* |
|---|---|---|---|---|
| 1 | front half, crank, 10 dowels | Pumpkin Orange 10301 | 115 g | 3 h 12 |
| 2 | back half | Pumpkin Orange 10301 | 199 g | 4 h 28 |
| 3 | drawer, eyes, teeth, knob, horns | Ivory White 11100 | 72 g | 1 h 58 |
| 4 | rollers, eye crank, rod, rocker, mouthpiece, pupils | Charcoal 11101 | 46 g | 1 h 33 |

In total about **430 g of filament and 11 h 11 of printing**.

\* Weight and time come from PrusaSlicer with a profile that approximates the A1, see
[controle/RAPPORT.md](controle/RAPPORT.md). The weight is reliable. Bambu Studio works out the time itself.

- **Settings**: *0.20mm Standard @BBL A1* on the *Textured PEI Plate*, 0.4 mm nozzle, **no supports**. The
  project already stores these per object:
  - body, drawer, mouthpiece, knob: 3 walls;
  - rollers: 4 walls and a 5 mm brim, because they stand upright and are 105 mm tall;
  - eyes: 3 walls and a brim;
  - crank: 4 walls, 40 % infill;
  - eye crank, rod, rocker, dowels: 4 walls (fully solid), dowels without brim.
- **Print orientation**:
  - The front half lies face down, the back half on its back. Everything inside grows up from that side. In
    print, the chute is a sloped ceiling that faces upwards.
  - The rollers stand upright, so they come out perfectly round: the lower one on its gear, the upper one on
    its left end. Each gear sits on a 45° collar.
  - The eyes stand on the right drum, with a 45° cone under the left drum.
  - The mouthpiece lies on its lip. The teeth and pupils lie on their front faces.
- **No supports**: small horizontal holes are teardrop-shaped, and blind holes have a 45° point. The slicer
  confirms that no plate needs supports.

## Test fit first

This is the first test from the build selection: a shielded test piece with the feed rollers. Print
`pasproef/pasproef.3mf` in one colour of your choice. The project has two plates:

| Plate | What | Filament* | Time* |
|---|---|---|---|
| 1 | test frame: the body around the rollers (mouth, bracket, ribs) with a closed lid and an exit slot at the back | 87 g | 2 h 14 |
| 2 | both rollers, crank, knob, eye crank, mouthpiece, 6 dowels, dowel strip | 59 g | 2 h 12 |

Step by step:

1. **Dowel strip**: a dowel should be a tight fit in one of the small holes (4.85 to 4.95 mm). In one of the
   large holes (5.00 to 5.10 mm) it should slide with some friction.
   - Set the sizes in `cad/params.py`: `GAT_PASPEN_VAST` and `GAT_PASPEN_LOS`.
2. **Assemble the test frame** as in *Assembly* below, steps 3 to 6 and step 8. The eyes, drawer, teeth and
   horns are not part of it.
   - The rollers should turn easily with the O-rings on.
   - If it turns too stiffly, make `GROEF_DIEP` 0.1 mm deeper.
   - If the paper slips, use thicker O-rings or wind a rubber band in as well.
3. **Test** with the crank:
   - [ ] **Paper types**: a sticky note, a strip of 80 g printer paper, receipt paper and a 160 g card. Each
         should come through whole and come out of the slot at the back.
   - [ ] **Skew**: push a note into the slot at an angle. It must not jam and must not reach the gears.
   - [ ] **Slip**: the note follows the crank without stalling.
   - [ ] **Reverse**: turn back halfway. The note comes back out.
   - [ ] **Fingers**: neither a finger nor a pencil can reach the rollers or gears anywhere, including
         through the slot.
   - [ ] **Bearings**: the stub axles and the crank shaft turn smoothly in the half holes. If not, adjust
         `GAT_DRAAI`.
   - [ ] **Hexagon**: the crank and the eye crank fit into the hex sockets of the rollers. If not, adjust
         `ZESKANT_GAT`.

Run `python bouw.py` again after every change.

## Assembly

1. **Mouthpiece**: press the teeth, with their three little pins, into the front of the mouthpiece. Then push
   the mouthpiece into the mouth from the front until the lip rests on the face.
2. **Eyes**:
   - Press the pupils into the drums by their pins.
   - Slide the rocker onto the D-shaped end of the eye shaft, with its pin facing inwards.
3. **Front half**: lay the front half face down on the table. Press six dowels into the holes in the split
   plane (they are a tight fit).
4. **Eye shaft**: click the eye shaft into its two snap bearings: one on the left next to the bracket, one
   between the drums. The rocker then sits in the left compartment, against the side wall.
5. **Lower roller**:
   - Put the three O-rings into the grooves.
   - Push the crank's hexagon into the right end of the roller.
   - Lay roller and crank together into the half holes. The stub goes into the left wall, the crank shaft
     through the right wall; the crank arm stays outside.
6. **Upper roller**:
   - Push the eye crank's hexagon into its left end.
   - Lay the roller into the half holes: the stub into the right wall, the eye crank into the bracket. The
     crank disc then sits in the left compartment.
   - Let the gears mesh.
7. **Connecting rod**: put its pin into the hole in the crank disc. The hole at its top goes onto the
   rocker's pin.
8. **Back half**: put the back half onto the dowels and press the two halves together.
9. **Finish**:
   - Click the knob into the crank arm.
   - Put a dowel into each horn and press them into the holes on top.
   - Slide the drawer into the belly.

**Paper jam?** Pull off the back half. The dowels slide out of the back half, and everything lies open in the
front half.

## Using it

Write your task on a sticky note. When it's done, put the note in the mouth with the sticky edge **last**.
Turn the crank away from you: its top moves backwards. Drawer full? Pull it out and look back at everything
you've done.

![Eyes up and down](renders/ogen.png)

## Folders

```
print/        taakvreter.3mf (4 plates) and every part as an STL in print orientation
pasproef/     pasproef.3mf (2 plates, test fit) and the STLs not already in print/
cad/          the source: Python scripts that generate everything above
controle/     RAPPORT.md: collisions, clearances, gears, eye linkage, paper path, finger probe, drawer, printability, slicer
renders/      Blender renders
onderdelen.json   volume and size per STL
```

## Building it yourself

You need Python 3 with `manifold3d`, `shapely`, `trimesh`, `numpy`, `contourpy` and `matplotlib`. The shared
code lives in [`gereedschap/`](../../gereedschap/) at the root of the repo.

```
cd cad
python bouw.py        # print/ and pasproef/: STLs and 3MF projects
python controle.py    # controle/RAPPORT.md (for the slicer table run python slice.py first, needs prusa-slicer)
python render.py alle # renders/ (needs the pip module bpy)
```

All dimensions are in `cad/params.py`. Every run recomputes the mechanism:

- collisions over a full turn of the crank;
- clearances and bearings;
- gear backlash;
- the eye linkage;
- the paper path.

If you change something, check `controle/RAPPORT.md` first to see that everything is still green.

![Side view](renders/zij.png)
![Exploded view](renders/explosie.png)
