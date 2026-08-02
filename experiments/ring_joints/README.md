# Interlocked ring joints

Second linkage for the flexi fish. **It is wired in**: `joint_style` selects
it in `flexifish.py`, `flexifish_nurbs.py` and both browser designers, and
the ball joint stays the default. The solid it subtracts lives in
`/joint_tool.py`; this folder is the working-out and the checks.

```bash
python3 flexifish.py --config <(echo '{"joint_style":"tool"}') --coupon
python3 flexifish_nurbs.py --config <(echo '{"joint_style":"tool"}')
```
or the **Linkage: Ball / Rings** toggle in either designer.

**Status: it builds, and it is linked.** The fusion that blocked an earlier
session is fixed, the fish meshes as separate parts at print resolution, both
rings are real and visible, and every joint passes the slide test. The linkage
is the *only* thing holding a joint together — there is no ball, no socket, no
dome. Nothing has been printed yet.

What the wiring costs, and it is not small: a ring joint eats about 26 mm of
body against a ball joint's 10, so the default fish drops from five segments
to two. Sizing reduces the count and says so rather than building a fish whose
segments come out fused. That trade is the reason the ball joint is still the
default.

---

## Why rings at all

The ball joint captures by **interference** — the socket mouth is narrower
than the ball. But that same mouth also has to open wide enough for the neck
to swing, so:

```
grip = R − (R + clearance)·sin(θ)        θ = swing + asin(neck/R)
```

Grip strictly *decreases* as swing increases. Measured on the shipping fish:
the front joint at 16° swing gets 0.56 mm of grip, the rear at 12° gets
0.90 mm. Every degree of motion is paid for in holding power, and no amount
of tolerance tuning escapes it because one aperture is doing both jobs.

Two closed rings threaded through each other are **topologically linked**.
They cannot separate at any tolerance — only by breaking plastic. Capture
becomes completely independent of both clearance and range of motion. Also
forgiving of printer variation (a sloppy fit still can't come apart, it just
rattles) and load spreads around a closed loop instead of cantilevering on a
thin socket lip.

Trade-off accepted deliberately: the ball joint *could* pop apart under
force and be pushed back together. Rings can't — they snap instead. The
owner chose rings knowing this.

---

## The geometry (confirmed with the owner)

Coordinates as in the generator: **x** nose→tail, **y** left/right, **z** up,
build plate at z = 0. "Axis" below means the torus's axis of revolution.

- **Tilted ring** — axis lies in the **XZ plane**, 30° from vertical
  (equivalently: the ring's *plane* is 60° from vertical / 30° off the
  floor). Fused into the front segment and protruding rearward past the seam.
  Roughly a third of it stands proud — the arc where `sin t < −1/2`, which is
  where the seam plane cuts it; the rest is buried in the body. The axis
  tilts toward **−x**, which puts the proud arc on the ring's *high* side;
  see "which way the ring leans" below, because the other sign does not print.
- **Vertical ring** — axis along **y** (plane XZ), fused into the rear
  segment and protruding forward. Vertical, so it prints unsupported.
- **The seam is a plain wedge** and nothing else. Each segment ends in a flat
  face that opens into a V with |y|, `face_gap` apart, and the two rings
  cross in the gap between them. See "no dome" below.

### The design rule (derived, then verified numerically to 3 decimals)

With the tilted ring's axis at angle `a` from vertical, both rings radius
`R`, centres offset by `off` along x:

```
links when   off < 2·R·cos(a)
best at      off = R·cos(a)
max centreline separation = R·(1 − sin a)
therefore    tube r ≤ (R·(1 − sin a) − clearance) / 2
```

`(1 − sin a)` is punishing, and it is what pinned down the 60°:

| a (axis from vertical) | ring plane | max tube @ R=5 |
|---|---|---|
| 0° | flat on the floor | 2.23 mm |
| **30°** | **30° off the floor** | **0.97 mm** ← usable |
| 45° | 45° | 0.46 mm |
| 60° | 60° off the floor | 0.06 mm — impossible |

If "60°" had meant the *axis* at 60° from vertical, the tube maxes out at
0.06 mm. So the only physically viable reading is the ring **plane** at 60°
from vertical, i.e. `ring_axis_deg = 30`.

Verification of the formula (`verify_rule.py`):

```
   a  off=Rcos a  measured sep  predicted  linked
   0        4.00         4.000      4.000    True
  10        3.94         3.305      3.305    True
  30        3.46         2.000      2.000    True
  60        2.00         0.536      0.536    True
```

### No dome

There used to be one: a ball on the front segment nesting into a cup on the
rear, meant to hide the linkage and act as a second capture. It is gone. Two
linked rings cannot come apart at any tolerance, so a second capture was
never needed, and the ball was actively in the way.

It was sized `R + rt + 1`, to clear the ring's envelope. But what matters is
how close the tilted ring comes to the ball's *centre*, and with the ring
offset `off/2 = R·cos(a)/2` forward of it:

```
|p(t)|² = R²·[1 + cos²(a)·(sin t + 1/4)]

nearest  = R·√(1 − ¾cos²a)  = 0.661·R at a = 30°
farthest = R·√(1 + 1¼cos²a) = 1.392·R at a = 30°
```

(linear in `sin t`, so the extremes are at `sin t = ∓1`; verified against
brute force to 3 decimals for R ∈ {4, 5.42, 7}, a ∈ {0, 20, 30, 45}). At
R = 7 the ball was 9.47 against a nearest approach of 4.63 — twice what it
could be — so it swallowed the tilted ring whole. The front segment came out
as a solid ball with a tunnel bored through it, the exact negative of the
rear segment, with no second ring anywhere on the model. The link was real;
it just was not the design.

Shrinking it to 4.63 made the ring visible, but it was still a sphere sitting
in the middle of the ring doing a job the ring already does. Removing it
outright is simpler and strictly better: the seam is a plain wedge, the
largest print island drops from 0.58 mm³ to 0.06 mm³ (the worst one *was* the
ball's undercut lower tip), the marching-cubes specks disappear entirely, and
about 70 s comes off the build.

Consequence for sizing: nothing bounds the rings but the body, so `fits()`
checks the **ring envelopes** directly — the vertical ring is the tall one
(`zc ± (R + rt)`), the tilted ring is the wide one (`± (R + rt)` in y, at
`z = zc`). Both get clipped to the skin when the plate is assembled, and a
clipped ring is a broken ring.

### Which way the ring leans

The tilted ring's axis can lean toward +x or −x. Mirroring x swaps the two
rings' positions, and the vertical ring is symmetric about its own centre, so
**both signs give an identical link and an identical centreline separation**
— `check_joints.py` prints the same numbers either way. Only one of them
prints.

The seam cuts the tilted ring at `sin t = −1/2`, so the protruding arc is the
`sin t < −1/2` third. With the axis toward **+x** that arc is the ring's low
half: it springs from the body face at `z = zc − R·sin(a)/2` and dips to
`zc − R·sin(a) − rt`, out past the seam with nothing beneath it. The printer
would have to start it in mid-air — `check_print.py` finds it as a 0.34 mm³
island at z = 4.70.

With the axis toward **−x** the same arc is the high half: it springs from
the body at `z = zc + R·sin(a)/2` and arches upward. The island is gone, and
the shallowest part of it steps about 0.8 mm horizontally per 0.2 mm layer
against a 2.9 mm tube — roughly 70% overlap layer to layer, which prints.

Same ring, same joint, same everything except supported instead of floating.
`ring_axis_deg` is still 30; it is the sign of the axis's x component that
matters, and it lives in `_size_joints`.

### `ring_fillet` — a root fillet, off by default

`_weld` can smooth-union each ring into its own segment instead of hard-
unioning it, filleting the root. A plain union leaves a sharp crease at the
one place the ring is a cantilever, so a fillet is right on stress grounds
whatever it looks like. It is **off** (`ring_fillet = 0.0`) because it was
written from a guess at what the owner's "fused / consumed edges" sketch was
asking for, and that guess is unconfirmed. At 0 the build is identical to the
plain-union version.

Both settings build clean:

```
ring_fillet = 0.0   614946 tris, manifold, shells = 8 (expected 8)
ring_fillet = 0.8   614694 tris, manifold, shells = 8 (expected 8)
```

and at 0.8 the joints pass, the coupon is 2 shells at 0.30 / 0.20 / 0.16 mm,
swing first contact is 16-20 deg and the largest print island is 0.06 mm3.

Two things it turned up that are worth keeping either way:

- **The relief has to be carved after the ring is welded, not before.**
  `smin(a, b, k)` bulges by up to k/4 where two surfaces meet, and the ring
  root sits squarely inside the channel the neighbouring segment carves for
  it. Carving first, a 0.8 mm fillet took the measured joint gap from 0.55 mm
  down to 0.40. Carving last restores it exactly, and costs nothing: the two
  bare rings are `clearance` apart by construction and stay that way through
  the swing, so the carve has none of the ring to take.
- **`smin` is not `min` in float32** outside the blend band — `b + (a - b)` is
  not bitwise `a`. Blending across the whole grid flipped one voxel on the
  belly rim, 25 mm from the nearest ring, and left a two-triangle hole in an
  otherwise manifold plate. `_weld` blends only inside `|seg - ring| < k`.

### Sizing consequence — read this before changing segment counts

A 0.70 mm minimum tube radius forces `R ≥ 3.9`, which makes the linkage
span `off + 2(R+r) ≈ 13 mm`, which needs **segments ≥ ~15 mm**. The
shipping fish has 11–12.5 mm segments — *no* ring pair fits them at any
tilt. This is inherent to ring linkages, not an implementation limit, and
it is why ring-based flexi models on the internet have chunky bodies.

Sizes the generator now produces on the blob fish at `n_segments = 2`:

| joint x | R | tube | offset | zc | skin over the rings (z, y) |
|---|---|---|---|---|---|
| 34.0 | 7.00 | 1.48 | 6.06 | 9.62 | 19.50, 10.48 |
| 76.0 | 7.00 | 1.47 | 6.06 | 9.62 | 7.61, 3.63 |
| 101.0 | 5.28 | 1.05 | 4.57 | 7.48 | 0.80, 1.62 |

Those tubes (1.0–1.5 mm radius) are genuinely strong — better than expected.
The front two joints are capped by `ring_max`; the rear one is capped by the
body, through the ring-envelope checks in `fits()` — 0.80 mm of skin over the
top of its vertical ring — so it is the first thing that gives if the body
gets any slimmer.

---

## Files here

| file | what it is |
|---|---|
| `proto.py` | torus SDF, ring sampling, **`linked()`** — the Hopf-link test (counts how many times ring B's centreline crosses ring A's disk; odd = truly linked) |
| `verify_rule.py` | **reproduces the verification table above** — predicted vs brute-force measured separation, plus the usable-tube table that pins down `a = 30` |
| `v2.py` | search for a viable (R, tube, offset) at `r = 0.34R`. Prints "no fit" at every tilt — that negative result is what forced the closed-form derivation, since a 34%-of-R tube is far too thick for this configuration |
| `check_joints.py` | **the joints the generator actually sizes**: linked, clearances, floor and lip thicknesses, and the pierce test on the built field |
| `check_swing.py` | **range of motion**: yaws the rear segment and reports where it first touches, and whether the stop is the rings or the seam |
| `check_print.py` | **island check**: walks the coupon layer by layer looking for material that starts in mid-air. This is the long-deferred overhang audit, and it is what caught the ring leaning the wrong way |
| `sdf_json.py` | reads a scene from the repo's `sdf_editor.html` and evaluates it as an SDF, so the owner's own model can be measured directly instead of paraphrased |
| `check_against_editor.py` | **proves `sdf_json.py` agrees with the editor** — lifts the editor's own `PRIMS` / `smin` / `invRot` / `sceneSDF` out of the HTML, runs them under node, and compares. Currently 1.9e-7 max error, 100% sign agreement over 4000 points |
| `owner_joint.json` | the owner's two-segment reference joint, and the thing the generator is meant to reproduce |
| `tool.json` | the owner's joint **tool**: one solid that, subtracted from a body, leaves two interlocking pieces. Body 1 of this file is what `/joint_tool.py` embeds, so this is the source of truth for it |
| `tool_cut.py` | **the ring joints the shipping generator builds** — cuts the whole default fish with `joint_style="tool"` and checks pieces, linkage and gap. Runs against `flexifish.py` and `joint_tool.py`, not the WIP copy, so it is a regression test rather than a study |
| `describe_scene.py` | **what a scene is actually made of**: node order per body, how far the segments interleave, and how the seam cavity sits against the rings it passes through |
| `flexifish_rings_WIP.py` | full `flexifish.py` with rings swapped in |
| `flexifish_rings.diff` | that same work as a unified diff against this branch's `flexifish.py` |
| `ring_pair.stl` / `.png` | a standalone linked pair, meshed and verified as 2 manifold shells. Printable on its own as a feel test |

```bash
cd experiments/ring_joints
python3 proto.py            # linkage across tilts and offsets (all True)
python3 verify_rule.py      # closed form vs measurement, + the tube table
python3 v2.py               # the "no fit" negative result, for context
python3 check_joints.py     # every joint of the default fish  (<1 s)
python3 check_swing.py 1    # range of motion at the middle joint  (~11 s)
python3 check_print.py 1    # islands at the middle joint  (~10 s)
python3 check_against_editor.py          # reader vs the editor  (~2 s, needs node)
python3 describe_scene.py                # structure of the reference joint  (~1 min)
python3 tool_cut.py                      # the shipping ring joints  (~8 min)
python3 flexifish_rings_WIP.py --out rings.stl --coupon   # ~3.5 min
```

Current results:

```
check_joints.py   all joints pass (linked, sep matches rule to 3 dp,
                  surface gap 0.550 = clearance, front-segment material
                  through the rear ring's hole)
check_swing.py    clear through 14 deg at every joint, first contact at
                  16-20 deg; design asks +/-6 to 8
check_print.py    largest island 0.06 mm3 -- a couple of voxels
coupon shells     2 at every joint, at 0.30 / 0.20 / 0.16 mm, no specks
rings.stl         614946 tris, manifold, shells = 8 (expected 8)
rings_joint_test  106344 tris, manifold, shells = 2 (expected 2)
```

and from the shipping generators, `joint_style="tool"` at `res = 0.30`:

```
tool_cut.py       4 body pieces, expected 4; every consecutive pair linked;
                  gaps 1.26 / 0.99 / 0.49 mm at joints 34 / 76 / 101
flexifish.py      622084 tris, manifold, shells = 8 (expected 8)
   --coupon       113820 tris, manifold, shells = 2 (expected 2)
flexifish_nurbs   585240 tris, manifold, shells = 8 (expected 8)
ball joints       unchanged: 11 shells (blob), 10 (NURBS), both manifold
```

---

## The fusion, and what it actually was

Historical, and kept because the *method* is the reusable part. All of this
happened while the dome still existed.

Last session's coupon came out as one welded blob and the note here blamed
the ring relief channel through the dome. **That was wrong.** Two segments'
fields never overlapped anywhere — the closest approach was a clean
+0.275 = clearance/2. The weld was somewhere the overlap test could not see,
because two solids can be disjoint and still be joined:

The face was `(X − xa) − sqrt(dome² − rho²)`, an offset **along x** rather
than a distance. At the dome rim the surface is parallel to x, so an x-offset
of `face_gap` bought a perpendicular gap of *zero*. The dome lip and the cup
lip met in a knife edge around the entire rim circle. Voxel-adjacency found
it immediately: touching voxels at x 76.6–78.25, spanning y ±9.5 and z 0–18.5
— exactly the circle of radius `dome` = 9.47 centred on the joint axis.

The fix was to make the face a real distance field — `min(wedge, ball)` at
the time, just `wedge` now that the ball is gone — so that offsetting it by
±`face_gap/2` gives a gap measured perpendicular to the surface, the same
everywhere. Shells went from one to two at every resolution tested, 0.30 down
to 0.12 mm. The general lesson outlives the dome: **a field you offset has to
be a distance, or the gap you think you asked for is not the gap you get.**

Three more things were wrong underneath it, all found the same way:

1. **`zc = R + rt + 0.6`** left 0.05 mm of floor under the neighbouring
   ring's relief carve — a wafer, not a floor. It is now
   `R + rt + clearance + 0.6`.
2. **The cup lip ran tangent to the skin.** `fits()` checked the ring
   envelope against the body but not the *dome*, so the cup could surface at
   a glancing angle and end in a knife edge. Moot now — no cup — but it is
   why `fits()` grew, and shrank again, and ended up back on the ring
   envelopes it started with.
3. **The seam wedge had the same tilt on both sides**, which is a parallel
   gap, not a V. The rear swept straight into the front on whichever side it
   turned toward. The two faces need opposite tilts — `faceF` and `faceR`.

Sub-voxel specks were produced where the relief tunnel left the dome at a
glancing angle: the two surfaces crossed in a feather edge thinner than a
voxel and marching cubes turned stray samples into lone octahedra. Removing
the dome removed them — the coupon is clean at 0.30, 0.20 and 0.16 mm.
`drop_debris` is still there and still worth keeping, because the shell count
is only a useful check if a stray octahedron cannot break it.

## Range of motion, and the relief sweep

The other open question was whether the linkage would fight a yaw about the
seam — the motion the fish actually makes. It does not, and it is the nicest
property the design has. Measured centreline separation under that yaw:

```
yaw     0     4     8    12    16    20 deg
sep  3.50  3.50  3.50  3.51  3.52  3.53 mm
```

0.03 mm of variation over 20°. Whatever else the pair does, turning is free.
(This was originally the argument that the dome and the rings were compatible
constraints. With the dome gone it is simply why the swept relief below can be
a pure yaw: nothing else needs modelling.)

What was actually stopping it: the relief carved for the neighbouring ring
was a constant `clearance` offset around where that ring **sits**. The ring
could therefore move `clearance` — about 4.5° at R = 7 — and then hit the
wall of its own channel. Carving the **swept volume** instead (`_swept`,
sampling the yaw at 4° steps; the scallop left between samples is under
0.01 mm) gives the range back. Measured, first contact moved 6° → 14°, and
moved from the body face to the rings, which is the end stop the design
wants.

---

## Subtracting a tool, instead of building a seam

The approach above -- derive a seam cavity from the body -- is superseded.
The owner's `tool.json` is a single solid that, cut out of a body, leaves two
interlocking pieces. It works, and it scales, which the fitted cavity would
not have.

It is deliberately **larger than the body it cuts**: the tool is 54.5 x 47 x
41.8 mm and overshoots the skin on every side. Only the interlock in the
middle has to sit inside the material, so getting the outer dimensions right
does not matter -- which is exactly why it survives being scaled.

Measured over the default fish, tool scaled to each joint's own section
(`tool_cut.py`, 0.35 mm voxels):

| joint | section | scale | pieces | linked | gap | thinnest |
|---|---|---|---|---|---|---|
| 34 | 20.1 x 37.6 | 0.98 | 2 | yes | 1.26 mm | 10.4 mm |
| 76 | 12.2 x 25.7 | 0.60 / 0.66 | 2 | yes | 0.99 mm | 10.8 mm |
| 101 | 8.1 x 14.6 | 0.40 / 0.38 | 2 | yes | 0.70 mm | 11.9 mm |

Two linked pieces at every joint, over a 2.5x range of scale, and it does not
much care whether the along-body scale follows the height or stays at 1.0.

Linkage is not assumed, it is tested: slide one piece along the body and watch
the overlap. Pieces that merely nest come apart at once and never overlap;
pieces that are linked have to pass through each other, so overlap goes
positive over a middle range before they clear. Every joint shows the second
signature.

### The belly plane is degenerate at level 0.0

Worth knowing before trusting any shell count from this generator. The plate
cut is `max(d, -Z)`, which is **exactly** zero wherever the body crosses
z = 0. If the meshing grid puts a lattice plane there -- and the default
`z0 = -1.2` with `res = 0.30` does, giving 3201 exact zeros -- then marching
cubes at `level = 0.0` stitches across that sheet and welds together every
part that touches the plate:

```
marching cubes at level  0.000:  6 shells
                        -0.001:  8 shells   correct
                        -0.010:  8 shells   correct
                        +0.001:  5 shells
```

Nothing was wrong with the geometry: the field labels into 8 disjoint pieces
under both 6- and 26-connectivity, so no grid cube can even contain material
from two of them. `mesh()` now meshes a hair below zero, which removes the
degeneracy and moves the surface by a thousandth of a millimetre.

This is why the first tool-cut build came out at 6 shells and looked like a
failure of the tool. It was not.

**The gap scales with the tool.** 1.26 -> 0.99 -> 0.49 mm as the tool shrinks,
so a joint much smaller than the tail one would close it up. `joint_gap` pins
it instead: the tool is eroded (or dilated) by half the difference *after*
scaling, so every joint gets the same clearance whatever its section. Left at
0 the tool keeps its own, which is what the table above measures.

## Wiring it into the generators

`joint_tool.py` at the repo root holds the solid — the eight nodes of
`tool.json`'s body 1, verbatim, plus an evaluator that agrees with
`sdf_json.py` to 0.0 — and the numbers derived from it. `flexifish.py` places
one per joint; `flexifish_nurbs.py` inherits that unchanged.

Two parameters: `joint_style` (`"ball"` | `"tool"`) and `joint_gap`.

**How much body a joint costs.** The tool's bounding box reaches 11 mm ahead
of the joint and 35.75 mm behind, but most of that tail is the wall of a cup
that wraps the rear piece from *outside* the skin. Cutting a real section,
removal starts at the tool's very front and has fallen to a few percent by 15
behind. So the joint costs `FOOTPRINT_AHEAD + FOOTPRINT_BEHIND` = 26 tool
units of body, and the layouts space the joints at least that far apart,
scaled.

That figure is a quality floor, not a hard one. The hard one was measured, by
cutting one fish with two tools at a range of spacings and labelling the
result:

| spacing (tool units) | pieces | linked |
|---|---|---|
| 10 | 5 — two of them debris under 2 mm³ | — |
| 12 | 3 | yes |
| 14 … 34 | 3 | yes |

Below 12 the second tool cuts through the first one's interlock; that is
`SHATTER_SPACING`, and sizing raises `SystemExit` there rather than warn.
Between 12 and 26 the joint works but the segment is nothing but shroud, with
none of its own body showing — a warning, since the dorsal fin can force it.

**What each generator does now.** The default blob fish: `n_segments` 5 → 2,
joints at 34 / 76 / 101, gaps 1.26 / 0.99 / 0.49 mm, four linked body pieces
at 0.35 mm voxels, 8 manifold shells at print resolution (2 segments + head +
tail + 4 fins), coupon 2. The default NURBS fish: `tail_segments` 3 → 1,
joints at 45 / 67 / 101, 8 manifold shells; its dorsal segment is 21.8 mm
against the 22.7 mm the joints reach, so it warns.

**The browser designers.** `fish_designer.html` never built joint geometry —
it grooves the cut lines and leaves the real plate to `flexifish.py` — so it
takes the layout rule and the notes only. `fish_designer_nurbs.html` does
build the printable plate, so it carries a full JS port of the solid;
`toolRaw` is exported from its core and agrees with `joint_tool.raw` to
3e-14, and `plateSDF` agrees with `NurbsFishBuilder.plate` in sign at every
sampled point. Note that `buildMesh` inlines the body build and calls the
segmenter itself — patching `plateSDF` alone changes the one-off queries and
not a single exported triangle.

Two bugs fell out of the port, both pre-existing and both in the designers:
their `lin()` divided by zero for a single-segment layout, so every cut came
out `NaN`; and the NURBS region summary quoted the `tail_segments` slider
rather than the cuts the layout made.

## The seam-cavity approach this replaced

Kept for the reasoning, not as a plan. `owner_joint.json` was the target, and
three things about it are structural, not
cosmetic, and each one breaks an assumption the generator is built on.

**1. Node order is load-bearing.** Each segment is built in this sequence:

```
ADD  vertical ring   k = 0        <- first, so both cuts apply to it
ADD  body blob       k = 11
CUT  ring relief     k = 1.5
CUT  seam cavity     k = 2.75
ADD  tilted ring     k = 3.5      <- last, so neither cut touches it
CUT  build plate     k = 1.75     (global)
```

The seam cavity's surface passes 12.4 mm *inside* the tilted ring's
centreline — it would swallow the ring whole — and does no damage, because the
ring is unioned after it. That is the whole trick, and it is what makes a
body-scale dish possible at all. The vertical ring is added first and so wears
the relief cut, which is the scallop measured earlier. The asymmetry is the
design, not an oversight.

**2. The joint is a socket, not a butt joint.** The two segments interleave
over 18.5 mm of body length: body 2 spans y −32.5…21, body 3 spans 2.5…56.
Neither ends in a face at a station. The generator cuts both segments at a
plane through the joint, which cannot express this.

**3. The cavity is a clearance envelope, not an offset of the mating nose.**
Measured over the facing region, the distance from the cavity surface to the
next blob's surface ranges over 18 mm — so "dilate the nose by the gap" is the
wrong rule. What it has to satisfy is weaker and local: clear the nose by at
least the gap, terminate the segment before the next joint, and stay off the
ring that was added before it. The dish is also *tighter* than the nose it
receives (pole radius 18.4 mm against 22.1 mm in z), so they meet near a point
rather than conforming — which is what stops the seam binding as it yaws.

Reference proportions, for whatever the generator ends up deriving them from:

| | value | as a fraction of the blob |
|---|---|---|
| cavity semi-axes | 18.5, 31.25, 24.0 | 0.90 x, 1.23 y, 1.01 z |
| cavity centre vs blob centre | (0, +24, −5) | +0.94 semi-y, −0.21 semi-z |
| cavity front pole | 7.25 mm ahead of the blob centre | |

The open question is what those derive from once the body tapers. The blob
does not exist in the generator — the body is one continuous blended field —
so the cavity has to come from the local section instead (`halfwidth_at` and
`top_at` already measure it). Copying the ratios above is not enough on its
own, because segment length and body section vary independently along the
fish, and the three constraints above are what actually have to hold.

## What is left

- **Nothing has been printed.** Print `rings_joint_test.stl` before trusting
  any of this. The whole design rests on 0.55 mm gaps resolving on a real
  printer, and no amount of field probing substitutes for that.
- The overhang audit is now `check_print.py`, and it passes, but it only
  looks at the joint coupon. The fins, the eyes and the caudal fan have never
  been through it.
- The proud arc of the tilted ring is a shallow arch. `check_print.py` says
  it is anchored, not floating, but "anchored" is not "prints cleanly" —
  around 70% layer-to-layer overlap at the shallowest point. This is the most
  likely thing to look bad on the first print.
- The seam is a visible V-groove and the rings show through it. Nothing hides
  the linkage any more; whether that reads as mechanical or as part of the toy
  is a call to make on a print, not on a render.
- The joint has slop. Only the rings hold the segments together, so a segment
  can be pulled back until the rings tighten and pushed forward until the
  faces meet — roughly `face_gap` of travel. Normal for a linked-ring flexi,
  and the trade the ball joint used to hide, but it will rattle.
- `joint_gap` is implemented and its arithmetic is checked, but no fish has
  been cut end-to-end with it pinned. The default (0, the tool's own) is what
  every number above measures.
- The NURBS default warns: its dorsal segment is 21.8 mm and the two joints
  around it reach 22.7 mm. It builds and it is linked, but that segment shows
  none of its own body. Widening the dorsal region would fix it.
- The ball-joint knobs (`joint_capture`, `joint_ball_max`, `joint_neck`) stay
  live — the ball joint is still the default, and it is the one that fits five
  segments in a 113 mm fish.
- Build time is 3.5 min at `res = 0.30`, up from 2.5 min before the swept
  relief, which costs ~25 extra full-grid torus evaluations. Restricting the
  sweep to a box around each joint would get most of that back.
