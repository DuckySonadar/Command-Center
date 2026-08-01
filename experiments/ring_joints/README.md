# Interlocked ring joints — work in progress

Replacement for the flexi fish segment linkage. **Nothing here is wired into
the shipping generator.** `flexifish.py` on this branch is the working
plate-cut ball joint (from PR #8); the ring code lives only in this folder.

Branch: `claude/ring-joints-wip`, based on `9302e59` (ball joints).
Not for merging into `main` as-is — see "what is left" at the bottom.

**Status: it builds.** The fusion that blocked the last session is fixed, the
joint meshes as separate parts at print resolution, both rings are real and
visible, and it swings past its design range. The linkage is the *only* thing
holding a joint together — there is no ball, no socket, no dome. Nothing has
been printed yet.

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
- Segment-count defaults: the blob fish needs `n_segments` 5 → 2 (done in
  this file); the NURBS fish needs `tail_segments` 3 → 2 (its 34.3 mm tail
  region gives 17.2 mm segments, which fits) — not done.
- Port to `nurbscore.js` for the browser designer, and extend
  `parity_test.js` (the Python↔JS harness lives in the session scratchpad
  and is **not** preserved — it will need rebuilding from `dump_ref.py`
  patterns described in the main README).
- The ball-joint knobs (`joint_capture`, `joint_ball_max`, `joint_neck`)
  become dead once rings land; remove them then, not before.
- Build time is 3.5 min at `res = 0.30`, up from 2.5 min before the swept
  relief, which costs ~25 extra full-grid torus evaluations. Restricting the
  sweep to a box around each joint would get most of that back.
