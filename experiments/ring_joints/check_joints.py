"""Check the rings the *generator* actually sizes, not the abstract rule.

verify_rule.py confirms the closed form. This confirms that what
`FishBuilder._size_joints` produces for a real fish is genuinely linked, has
the clearance it thinks it has, and leaves printable material around itself.

    python3 check_joints.py [config.json]
"""
import importlib.util
import json
import os
import sys

import numpy as np

from proto import linked

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "flexifish_rings_WIP", os.path.join(HERE, "flexifish_rings_WIP.py"))
ff = importlib.util.module_from_spec(_spec)
sys.modules["flexifish_rings_WIP"] = ff
_spec.loader.exec_module(ff)


def pierces(b, ji, n=140):
    """Does the front segment's *solid* thread the rear segment's ring?

    The centreline test below is about two ideal tori. What actually gets
    printed is not that: `dome` is bigger than R + rt, so the tilted ring ends
    up buried inside a solid dome, and the linkage is carried by a tunnel
    bored through that dome rather than by a visible second ring. Which is
    fine -- more material, not less -- but it means the centreline test would
    still pass if the dome had swallowed the linkage whole. So check the
    field: front-segment material inside the rear ring's hole, front-segment
    material outside the rear ring entirely, and one connected piece joining
    them, is a link that cannot be pulled apart.
    """
    j = b.joints[ji]
    R, rt = j["R"], j["rt"]
    c = np.array(j["cB"])
    nrm = np.array(j["axB"], dtype=float)
    a = np.array([0.0, 0.0, 1.0])
    u = np.cross(nrm, a); u /= np.linalg.norm(u)
    w = np.cross(nrm, u)
    # a disk spanning the ring, stopping short of the tube itself
    rr = np.linspace(0.0, R - rt - 0.05, n)
    tt = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    RR, TT = np.meshgrid(rr, tt, indexing="ij")
    P = (c + RR[..., None] * (np.cos(TT)[..., None] * u
                              + np.sin(TT)[..., None] * w))
    X = P[..., 0].astype(ff.F32)[..., None]
    Y = P[..., 1].astype(ff.F32)[..., None]
    Z = P[..., 2].astype(ff.F32)[..., None]
    through = (b.segment(ji, X, Y, Z) < 0).sum()
    # and the same segment well outside the ring, so the material that
    # threads the hole has to come back around the outside
    far = np.array([j["xa"] - R - rt - 3.0, 0.0, j["zc"]], dtype=ff.F32)
    outside = float(b.segment(ji, *[np.array([[[v]]], dtype=ff.F32) for v in far])[0, 0, 0])
    return int(through), outside < 0


def centreline_gap(j, n=2000):
    """Smallest distance between the two ring centrelines, brute force."""
    from proto import ring_pts
    A = ring_pts(np.array(j["cA"]), np.array(j["axA"]), j["R"], n)
    B = ring_pts(np.array(j["cB"]), np.array(j["axB"]), j["R"], n)
    return float(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).min())


def main(argv):
    p = ff.FishParams()
    if len(argv) > 1:
        p = ff.replace(p, **json.load(open(argv[1])))
    b = ff.FishBuilder(p)
    for w in b.warnings:
        print("note:", w)

    print(f"{'x':>7} {'R':>6} {'tube':>6} {'offset':>7} {'zc':>6} {'dome':>6} "
          f"{'sep':>7} {'surf gap':>9} {'floor':>6} {'lip z':>6} {'lip y':>6} "
          f"linked")
    bad = 0
    for ji, j in enumerate(b.joints):
        sep = centreline_gap(j)
        surf = sep - 2 * j["rt"]                       # metal-to-metal gap
        # material left under the *other* ring's relief carve, and body wall
        # left outside the cup, which is the dome ball grown by face_gap/2
        floor = j["zc"] - j["R"] - j["rt"] - p.clearance
        top = b.top_at(j["xa"])
        lip_z = top - (j["zc"] + j["dome"] + p.face_gap / 2)
        lip_y = b.halfwidth_at(j["xa"], j["zc"]) - (j["dome"] + p.face_gap / 2)
        ok1, n1 = linked(np.array(j["cA"]), np.array(j["axA"]), j["R"],
                         np.array(j["cB"]), np.array(j["axB"]), j["R"])
        ok2, n2 = linked(np.array(j["cB"]), np.array(j["axB"]), j["R"],
                         np.array(j["cA"]), np.array(j["axA"]), j["R"])
        ok = ok1 and ok2
        thru, outside = pierces(b, ji)
        print(f"{j['xa']:7.1f} {j['R']:6.2f} {j['rt']:6.2f} {j['off']:7.2f} "
              f"{j['zc']:6.2f} {j['dome']:6.2f} {sep:7.3f} {surf:9.3f} "
              f"{floor:6.2f} {lip_z:6.2f} {lip_y:6.2f}  {ok} ({n1},{n2})"
              f"  through={thru}")
        # the rule predicts sep = R*(1 - sin a) at off = R*cos a
        pred = j["R"] * (1 - np.sin(np.deg2rad(p.ring_axis_deg)))
        for name, got, want in (("linked", ok, True),
                                ("sep matches rule", abs(sep - pred) < 5e-3, True),
                                ("surface gap >= clearance",
                                 surf > p.clearance - 1e-3, True),
                                ("floor under relief >= 0.5", floor > 0.5, True),
                                ("cup lip in z >= 0.9", lip_z > 0.9, True),
                                ("cup lip in y >= 0.9", lip_y > 0.9, True),
                                ("tube >= ring_tube_min",
                                 j["rt"] >= p.ring_tube_min, True),
                                ("front segment threads the rear ring",
                                 thru > 0, True),
                                ("...and reaches outside it", outside, True)):
            if got != want:
                print(f"    FAIL at x={j['xa']:.1f}: {name}")
                bad += 1
    print("\nall joints pass" if not bad else f"\n{bad} check(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
