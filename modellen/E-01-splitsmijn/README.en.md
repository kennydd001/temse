# E-01 The Split Mine (De Splitsmijn) (final model v1.0, test print v0.1)

*Nederlands: [README.md](README.md)*

A math toy for learning **number bonds up to 10**, for first grade (ages 5 to 6). The child counts out a
handful of diamond coins, throws them into the mine one by one and gives the pickaxe a push. The coins tick
down past the pins and disappear into two **mine carts**, deep inside the mountain. Nobody sees where they went.

Pull out one cart: 3 coins. How many are in the other one? *8 = 3 + ?* Guess first, then open the second
cart. Lay the coins on the ten-frame: the ones from the first cart blue side up, the ones from the second
yellow side up.

![The Split Mine: 8 = 3 + 5](renders/splitsmijn.png)

> **Status:** the CAD, the STLs, the Bambu Studio project and the checks are done, for the final model and for
> the test print. **Nothing has been printed yet.**
>
> The 3MF projects have been checked against the rules of the Bambu format (`gereedschap/bambu/valideer.py` in
> the root of the repo). They have not been opened in Bambu Studio itself: that program does not run in this
> environment.

| | |
|---|---|
| Size (w × d × h) | 232 × 108 × 303 mm |
| Coins | 10 coins Ø 25 × 4.5 mm (the size of a 2-euro coin), half blue with a diamond, half yellow |
| Filament | about 670 g in 6 colours (slicer), some 19 hours of printing over 9 plates |
| Hardware | none: no screws, glue or metal parts. An AMS is only needed for the name and the coins |

Colours (Bambu PLA):

- **Charcoal 11101**: the back and front plate (the coins stand out against the dark board) and the letters of
  the name.
- **Latte Brown 11800**: the mine timber, the portal, the pickaxe handle, the bracket.
- **Terracotta 11203**: the mine mountain and the ten-frame.
- **Sunflower Yellow 10402**: the pickaxe head, the tilting plank, the divider fin, the yellow side of the coins.
- **Maroon Red 10205**: the two mine carts.
- **Marine Blue 11600**: the blue side of the coins.

## How it works

![Without the front plate and the mine timber](renders/splitsmijn_binnen.png)

- **Drop board**: the coins fall upright in a 6 mm slot between the back and the front plate.
  - There are four rows of 3 or 4 pins in a triangular grid, 44 mm apart. Between two pins, or between a pin
    and the wall, there is always at least 29 mm, for a 25 mm coin.
  - Next to the rows with a middle pin, a half pin sits in the wall, so no coin slides down along the wall.
  - At the bottom, the **divider fin** stands right under the middle pin. Whatever falls to the left of it goes
    into the left cart.
- **Why a pickaxe**: a 25 mm coin is big next to pins 44 mm apart. It does not bounce randomly; it rolls along
  a fixed path over the pins. On an ordinary drop board it goes to the side the child aims for 65 to 95 % of
  the time. Then the child learns to aim, not to split.
  - That is why every coin first lands on a **tilting plank** under the slot at the top. The plank is fixed to
    the pickaxe, a heavy pendulum in front of the board.
  - The child gives the pickaxe one push. What decides is which side of the plank is lower at that moment. The
    coin is far too light to disturb the pendulum.
- **A knife edge instead of an axle**: a printed axle brakes a pendulum so much that it stops after 10 to 20
  seconds.
  - So the pickaxe hangs on a **knife edge** in a V-groove, like an old-fashioned balance. Half a minute after a
    push it still swings 26 to 31°: one push lasts a whole round.
- **Mine mountain**: under the board, two funnels in the deck guide every coin into a **cart** in a tunnel.
  - Between the top of a cart and the deck there is 1 mm: no coin fits in there.
  - The carts slide out of the front of the mountain like drawers.
- **Mine timber and portal**: the timber frame and the mine entrance with the name hold without glue.
  - Left and right, a **socket foot** clamps into the deck with its plug. The side rails of the board and the
    uprights of the frame stand together in its U-channel, 30 mm tall.
  - At the top, each corner gets a **socket cap**. Its lower slot clamps over the board and the frame, its upper
    slot holds a tongue of the portal.
  - Pegs on the back keep the frame in place.

![Exploded view](renders/splitsmijn_explosie.png)

### What the checks say

The checks rebuild the drop board in a 2D physics model, from the slot at the top into the carts.

- A round goes like in class: one push, then 8 coins 0.8 to 2 seconds apart.
- It runs for 12 combinations of bounce, friction and damping, because PLA on PLA is not known exactly.
- The pickaxe is in it with its real mass, calculated from the geometry: 44 g, period 0.77 s.

Everything is in [controle/RAPPORT.md](controle/RAPPORT.md) (in Dutch). In short:

- **Jamming**: 0 of 4608 coins got stuck. On the board, there is no spot where a coin can rest on two points.
  The funnels are 56° and 65° steep: a coin always slides on.
- **Aiming**: aiming left gives at most 7 percentage points more left than aiming right. Without the pickaxe
  that is 30 to 90 percentage points.
- **Splits** of 8 follow a fair coin: 4 + 4 comes up most often, 0 + 8 hardly ever.
- **Fit**: over the pickaxe's whole swing nothing touches, with at least 0.5 mm clearance everywhere.
- **Stability**: pushing the pickaxe backwards takes 3.8 N (about 390 g) before the board tips.

![Checks](controle/valbord.png)

![The mountain in section](controle/mijnberg.png)

## In class

The Split Mine is the concrete level: the child sees and feels the coins split.

1. **Concrete**: count out 8 coins and throw them into the mine. Pull out both carts and count. *8 is 3 and 5.*
2. **Hidden cart**: pull out only one cart. There are 3 in it. How many are in the other one? *8 = 3 + ?*
   Guess first, then look.
3. **Pictorial**: lay the coins on the ten-frame, first the ones from one cart blue side up, then the ones from
   the other yellow side up. Or fill in a number bond diagram.
4. **Abstract**: write *8 = 3 + 5*. What is *5 + 3* then?

Because the split is different every time, the child eventually practises every split of the same number.

## Printing

Open `print/splitsmijn.3mf` in Bambu Studio. The project has nine plates:

| Plate | What | Colour | Filament* | Time* |
|---|---|---|---|---|
| 1 | **fit strip: print this first**, see below | Charcoal | 11 g | 0 h 31 |
| 2 | back plate with the pins (the same as for the test print) | Charcoal | 138 g | 3 h 37 |
| 3 | front plate with viewing slots | Charcoal | 107 g | 3 h 10 |
| 4 | mine timber | Latte Brown | 33 g | 1 h 06 |
| 5 | mine mountain, ten-frame | Terracotta | 195 g | 5 h 41 |
| 6 | portal with the name, handle, bracket, knife block, two socket feet, two socket caps | Latte Brown + Charcoal (letters) | 60 g | 1 h 52 |
| 7 | pickaxe head, tilting plank, divider fin | Sunflower Yellow | 33 g | 0 h 48 |
| 8 | two carts | Maroon Red | 78 g | 1 h 25 |
| 9 | 10 coins | Sunflower Yellow + Marine Blue | 17 g | 0 h 36 |

That is about **670 g of filament** and 19 hours of printing.

\* From PrusaSlicer with a profile that approximates the A1, each object with the walls and infill from the
project (`cad/slice.py`). The weight is reliable; Bambu Studio calculates its own print time. The purge for the
colour changes on plates 6 and 9 comes on top.

- **Settings**: *0.20mm Standard @BBL A1* on the *Textured PEI Plate*, 0.4 mm nozzle, **no supports**, gyroid
  infill. The project already sets this per object:
  - the handle, the pickaxe head and the knife block: **100 % infill** (rectilinear). A heavy pendulum swings
    longer.
  - everything else: 2 to 4 walls.
- **Two colours on one plate**, only on plates 6 and 9.
  - On plate 6 the letters lie 0.6 mm deep in the crossbeam of the portal: a few filament changes.
  - On plate 9 every coin changes once, halfway: the bottom half yellow, the top half blue.
  - No AMS? Print the letters in the same brown, or change filament by hand at the right height.
- **Print orientation** (already set in the project):
  - The back and front plate lie flat; the mine timber lies front face down.
  - The **mine mountain lies upside down**: the tunnels are then open at the top and nothing hangs in the air.
  - The handle, the pickaxe head and the knife block lie on their backs. The portal is a flat plate and lies on
    its back too: the letters are in the top layers.
  - The socket feet stand upright, plug down on the bed. The socket caps lie on their outer side.
- Everything prints without supports. Even PrusaSlicer with automatic supports finds nothing, apart from a few
  (0.5 g) under the bridge in the slot of the pickaxe head. Without supports it prints that as a bridge.
  - The carts stand upright, open at the top.

| Part | Qty | Size (mm) | File |
|---|---|---|---|
| back plate with pins and side rails | 1 | 218 × 230 × 10 | `print/achterplaat.stl` |
| front plate | 1 | 218 × 230 × 2.5 (16.5 with pins) | `print/voorplaat.stl` |
| mine timber (frame of beams) | 1 | 218 × 230 × 2.4 | `print/mijnhout.stl` |
| mine mountain | 1 | 232 × 108 × 40 | `print/mijnberg.stl` |
| ten-frame | 1 | 157 × 70 × 4.5 | `print/tienveld.stl` |
| portal with the name | 1 | 224 × 31 × 10.6 | `print/portaal.stl` |
| pickaxe handle | 1 | 20 × 174 × 10.5 | `print/steel.stl` |
| pickaxe head | 1 | 89 × 25 × 20 | `print/kop.stl` |
| socket foot | 2 | 13 × 18 × 44.5 | `print/kokervoet_L.stl`, `print/kokervoet_R.stl` |
| socket cap | 2 | 24 × 20.5 × 13 | `print/kokerkap_L.stl`, `print/kokerkap_R.stl` |
| bracket with V-groove, knife block | 1 + 1 | 44 × 17 × 13; 20 × 16 × 6.5 | `print/beugel.stl`, `print/mesblok.stl` |
| tilting plank, divider fin | 1 + 1 | 36 × 10 × 16; 3 × 25 × 5 | `print/plank.stl`, `print/verdelervin.stl` |
| mine cart | 2 | 92 × 105 × 31 | `print/karretje_L.stl`, `print/karretje_R.stl` |
| coin | 10 | Ø 25 × 4.5 | `print/munt.stl` |

Each STL is one piece (the portal and the coin too), for printing without an AMS.

## The fit strip first

![Fit strip](renders/passtrip.png)

There are no snap hooks: everything **clamps** or **slides** together. How tight that is depends on your
printer.

- A hole usually comes out 0.1 to 0.2 mm smaller than in the CAD.
- With the 26 pegs of the front plate that matters a lot: too tight, and you cannot get the front plate on
  (or off again).

So print plate 1 first (about half an hour) and try the pieces together:

| Try | In | Choose the hole where it | Set in `cad/params.py` | Now |
|---|---|---|---|---|
| peg of the peg block | round holes 3.0 to 3.4 | goes in with light thumb pressure and stays, without forcing | `NOK_GAT` | 3.2 |
| the same peg | round holes 3.0 to 3.4 | clamps firmly (one size tighter is fine) | `MES_GAT` | 3.1 |
| hexagon of the peg block | hex holes 8.1 to 8.4 | slides easily without wobbling | `ZESKANT_SPEL` = size − 8 | 0.2 |
| Ø 4 pin of the peg block | holes 4.0 to 4.3 of the bracket test | clamps firmly | `BEUGEL_GAT` | 4.1 |

`MES_GAT` also applies to the pins of the handle in the pickaxe head.

- Does everything match the current values? Then there is nothing to do.
- If not: change the values and run `python bouw.py` in `cad/` again. Then print the other plates.

What clamps where and how much clearance each joint has is in [controle/RAPPORT.md](controle/RAPPORT.md),
under *Verbindingen*.

## Assembly

1. **Back plate**: lay it flat on the table, pins up.
2. **Tilting plank**: lay the plank in the middle under the slot, hub up.
3. **Front plate**: lay it on top, viewing slots towards you. Push the hub through the round hole. Press the
   front plate down all round: the 26 pegs clamp lightly in the holes. The plank must tilt freely.
4. **Mine timber**: lay the frame on the front plate. The pegs on its back drop into the holes of the front
   plate.
5. **Pickaxe**:
   - press the knife block onto the two pins at the top of the handle, knife edge down;
   - press the pickaxe head onto the two pins at the bottom;
   - then slide the handle with its hex hole onto the hexagon of the plank.
6. **Bracket**: slide the bracket with its two holes onto the pins under the hub, legs towards the front
   plate. Lift the pickaxe slightly while sliding: that way the knife edge slides into the groove from behind.
7. **Mine mountain**:
   - press the tongue of the divider fin into the slot in the middle of the deck;
   - press the plugs of the two socket feet into the recesses left and right in the deck, open side towards the
     middle. They clamp lightly;
   - slide the two carts into the tunnels.
8. **Board into the mountain**: stand the board upright between the socket feet. The side rails and the
   uprights of the mine timber sink into the sockets together, and the divider fin slides into the slot at the
   bottom.
9. **Portal**: slide the two socket caps over the top corners of the board, open side towards the middle. They
   go over the board and the mine timber together. Then put the two tongues of the portal into the caps.

The ten-frame lies loose on the table, in front of the mine: there is no room for it on the deck, and the pickaxe
swings there.

Moving it? Lift the Splitsmijn by the mine mountain, not by the portal: the board stands loose in the socket feet.

## Testing

Always give the pickaxe a push first. If it hangs still, the plank is level and the aim decides again.

- [ ] **Swinging**: push the pickaxe to about 30° and count how long it moves. After 30 seconds it should still
      be clearly swinging.
  - Does it stop quickly? Check that the knife edge sits nicely in the groove and that the hub and the plank do
    not rub anywhere.
- [ ] **Jamming**: throw 10 rounds of 10 coins. Not a single one may get stuck, not in the funnels and not
      between a cart and the deck.
  - Does one get stuck? If it is pinched between the plates, make `SPLEET` in `cad/params.py` 0.5 mm larger.
- [ ] **Slot**: does every coin pass easily through the slot at the top? If not, make `INWORP_HALF` 0.5 mm
      larger.
- [ ] **Carts**: do they slide out and in easily, also when full? If not, make `KAR_SPEL` larger.
- [ ] **Socket feet**: do they clamp in the deck without a hammer? Too tight: make `KOKER_SPEL` 0.1 larger. Too
      loose: one step smaller, or a strip of tape round the plug.
- [ ] **Fair**: throw 40 coins and aim all of them as far left as possible in the funnel. Then do the same on
      the right.
  - Count how many come out left each time. On a fair board that is between 14 and 26 both times.
  - Do far more come out left when aiming left? Then the aim still counts.
- [ ] **With a child**: let a preschooler or a first grader play.
  - Do they understand the push, the slot and the carts without explanation?
  - How long do they keep playing?

![A cart up close](renders/splitsmijn_kar.png)

## Test print (v0.1)

The test print is the same drop board with a simple base, two open trays and a red pendulum. It tests the two
things that could break the concept:

1. Does a coin ever get stuck?
2. Are the splits really random?

If you want to test only those two things first, print `pasproef/pasproef.3mf` (6 plates, about 395 g and
almost 10 hours). The **back plate is the same** as in the final model. The pendulum and the bracket hang 3 mm
closer to the front plate, because there is no mine timber in between.

![The test print](renders/voor.png)

| Plate | What | Colour | Filament* | Time* |
|---|---|---|---|---|
| 1 | fit strip | Charcoal 11101 | 12 g | 0 h 32 |
| 2 | back plate with the pins | Charcoal 11101 | 138 g | 3 h 16 |
| 3 | front plate with viewing slots | Charcoal 11101 | 109 g | 3 h 02 |
| 4 | base with the trays, bracket | Latte Brown 11800 | 74 g | 1 h 36 |
| 5 | pendulum, knife block, tilting plank | Maroon Red 10205 | 46 g | 0 h 54 |
| 6 | 10 coins | Marine Blue 11600 | 17 g | 0 h 32 |

\* From PrusaSlicer with a profile that approximates the A1. The weight is reliable; Bambu Studio calculates
the time itself.

- `pasproef/voorplaat_glad.stl` is the same front plate without viewing slots. It is for transparent filament
  (PETG or clear PLA), if you want to see the coins fall all the way.
- Assembly goes as above, without the mine timber, the portal and the carts. The board sinks with its side
  rails into the sockets of the base.

![The knife edge in the groove](renders/mes.png)

## Folders

```
print/        the final model: splitsmijn.3mf (9 plates, plate 1 is the fit strip) and every part as an STL in print orientation
pasproef/     the test print: pasproef.3mf (6 plates) and every part as an STL
cad/          the source: Python scripts that make everything above
              params.py (all sizes), valbord.py (the board and the test print), mijn.py (the final model),
              simulatie.py, platen.py, bouw.py, controle.py, render.py
controle/     RAPPORT.md: coin passage, drop simulation, pendulum, clearances, joints, stability, printability;
              slicer.json and slicer_eind.json (PrusaSlicer)
renders/      Blender renders: splitsmijn*.png (final model), voor, binnen, mes, explosie, passtrip (test print)
onderdelen.json   volume and size per STL
```

```
cd cad
python bouw.py            # print/ and pasproef/
python slice.py           # controle/slicer*.json: weight, time and supports per plate (needs prusa-slicer)
python controle.py        # controle/ (--snel for a shorter simulation)
python simulatie.py       # just the drop simulation (test print)
python render.py eind     # renders/ of the final model (proef: the test print; needs the pip module bpy)
```

## MakerWorld

The concept borrows from a well-known block game and from the drop board of a TV game show. Do not use those
names in the title, the tags or the description. The model is called **The Split Mine** (De Splitsmijn), and the
mine, the pickaxe, the carts and the diamonds are our own drawings.
