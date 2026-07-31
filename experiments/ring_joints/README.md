# Interlocked ring joints — work in progress

Unfinished replacement for the flexi fish segment linkage. **Nothing here is
wired into the shipping generator.** `flexifish.py` on this branch is the
working plate-cut ball joint (from PR #8); the ring code lives only in this
folder.

Branch: `claude/ring-joints-wip`, based on `9302e59` (ball joints).
Not for merging into `main` as-is.

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
  floor). Fused into a **convex dome** on the front segment, protruding
  rearward. Being nearly flat it cannot print floating — the dome is what
  supports it. Only a thin rim on its underside sits at the steep angle;
  the rest of the torus curves away and is shallower.
- **Vertical ring** — axis along **y** (plane XZ), recessed in a **concave
  cup** on the rear segment. Vertical, so it prints unsupported.
- Dome nests into cup so the linkage is hidden and only a body seam shows.

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

### Sizing consequence — read this before changing segment counts

A 0.70 mm minimum tube radius forces `R ≥ 3.9`, which makes the linkage
span `off + 2(R+r) ≈ 13 mm`, which needs **segments ≥ ~15 mm**. The
shipping fish has 11–12.5 mm segments — *no* ring pair fits them at any
tilt. This is inherent to ring linkages, not an implementation limit, and
it is why ring-based flexi models on the internet have chunky bodies.

Working sizes the WIP produced on the blob fish at `n_segments = 2`:

| joint x | R | tube | offset | zc | dome |
|---|---|---|---|---|---|
| 34.0 | 7.00 | 1.48 | 6.06 | 9.07 | 9.47 |
| 76.0 | 7.00 | 1.47 | 6.06 | 9.07 | 9.47 |
| 101.0 | 5.42 | 1.08 | 4.69 | 7.10 | 7.50 |

Those tubes (1.1–1.5 mm radius) are genuinely strong — better than expected.

---

## Files here

| file | what it is |
|---|---|
| `proto.py` | torus SDF, ring sampling, **`linked()`** — the Hopf-link test (counts how many times ring B's centreline crosses ring A's disk; odd = truly linked) |
| `verify_rule.py` | **reproduces the verification table above** — predicted vs brute-force measured separation, plus the usable-tube table that pins down `a = 30` |
| `v2.py` | search for a viable (R, tube, offset) at `r = 0.34R`. Prints "no fit" at every tilt — that negative result is what forced the closed-form derivation, since a 34%-of-R tube is far too thick for this configuration |
| `flexifish_rings_WIP.py` | full `flexifish.py` with rings swapped in. **Broken — see below** |
| `flexifish_rings.diff` | that same work as a unified diff against this branch's `flexifish.py` |
| `ring_pair.stl` / `.png` | a standalone linked pair, meshed and verified as 2 manifold shells. Printable on its own as a feel test |

Reproduce the linkage check:

```bash
cd experiments/ring_joints
python3 proto.py        # linkage across tilts and offsets (all True)
python3 verify_rule.py  # closed form vs measurement, + the tube table
python3 v2.py           # the "no fit" negative result, for context
```

---

## Where it broke

`flexifish_rings_WIP.py` sizes rings correctly and meshes, but **the two
segments come out as one fused shell instead of two**. A coupon around the
joint at x=76 gave 5 shells: 2 pelvic-fin fragments (coupon-window
artifacts, ignorable), 2 single-point marching-cubes specks, and one
584k-triangle blob spanning x 62→90 — i.e. both segments welded together.

Each piece checks out in isolation, which is why this is confusing:

- face gap is 1.0 mm
- inter-ring gap is `R(1−sin a) − 2r` = 3.5 − 2.94 = **0.56 mm**, about
  3.5 voxels at the 0.16 mm coupon resolution — should resolve fine
- both ring reliefs are applied (`max(seg, −(otherRing − clearance))`)

### Leading suspicion

The dome bulges rearward by its **full radius** — 9.47 mm at R=7. So the
front segment's convex face reaches to `xa + 9.47`, which puts the rear
segment's vertical ring (centre at `xa + 3.03`) *deep inside the front
segment's dome*. That is geometrically intended — it is a pin through a
hole — but it means ring B threads a long channel through front-segment
material, and if the relief carve does not fully span that channel the two
weld together somewhere along it.

### Suggested next steps, cheapest first

1. **Probe the field along the channel.** Walk `plate()` along a line at
   `y=0, z=zc` from `xa − 12` to `xa + 12` and print the values; find the x
   where the gap fails to open. Ten lines of code, answers it directly.
2. **Shrink the dome** so it barely clears the rings (`ring_dome_pad` →
   0.2, or cap `dome` at `R + r`). If the fusion disappears, the channel
   theory is confirmed.
3. **Check relief ordering.** In `plate()` the relief is applied before the
   segment's own ring is unioned in. Confirm the relief for ring B actually
   covers ring B's full extent inside the dome, not just near the joint
   plane.
4. **Render cross-sections** the way the ball joint was debugged — slice
   `plate()` at `z = zc` and at `y = 0` and contour it. That visual made
   the ball-joint problems obvious immediately and would here too.

### Also still to do once it meshes

- Dome/cup rotation: the dome's sphere is centred at `(xa, 0, zc)` but the
  mechanical pivot is the ring pair. They are not co-located, so the seam
  gap will open as the joint flexes. May need the dome centred on the
  linkage, or accepting a wider `face_gap`.
- Overhang audit on the tilted ring's underside rim.
- Segment-count defaults: blob fish needs `n_segments` 5 → 2; the NURBS
  fish needs `tail_segments` 3 → 2 (its 34.3 mm tail region gives 17.2 mm
  segments, which fits).
- Port to `nurbscore.js` for the browser designer, and extend
  `parity_test.js` (the Python↔JS harness lives in the session scratchpad
  and is **not** preserved — it will need rebuilding from `dump_ref.py`
  patterns described in the main README).
- The ball-joint knobs (`joint_capture`, `joint_ball_max`, `joint_neck`)
  become dead once rings land; remove them then, not before.
