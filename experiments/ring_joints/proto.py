"""Ring-pair prototype: verify topological linkage + fit before integrating."""
import numpy as np

def torus_sdf(P, C, N, R, r):
    """P:(...,3) points. Torus: centre C, plane normal N, major R, tube r."""
    v = P - C
    h = v @ N
    q = np.linalg.norm(v - h[..., None] * N, axis=-1)
    return np.sqrt((q - R) ** 2 + h ** 2) - r

def ring_pts(C, N, R, n=400):
    """Sample the ring centreline."""
    a = np.array([0.0, 0.0, 1.0])
    if abs(N @ a) > 0.9: a = np.array([1.0, 0.0, 0.0])
    u = np.cross(N, a); u /= np.linalg.norm(u)
    w = np.cross(N, u)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return C + R*(np.cos(t)[:, None]*u + np.sin(t)[:, None]*w)

def linked(C1, N1, R1, C2, N2, R2):
    """Does ring 2's centreline cross ring 1's disk an odd number of times?"""
    P = ring_pts(C2, N2, R2, 2000)
    h = (P - C1) @ N1                      # signed height above ring 1's plane
    cross = 0
    for i in range(len(P)):
        j = (i + 1) % len(P)
        if h[i] == 0 or (h[i] > 0) != (h[j] > 0):
            t = h[i] / (h[i] - h[j])
            X = P[i] + t * (P[j] - P[i])   # plane crossing point
            if np.linalg.norm(X - C1) < R1:   # inside the disk -> a real link
                cross += 1
    return cross % 2 == 1, cross

def tilt_normal(deg):
    """Ring normal tilted `deg` from vertical: Y axis rotating toward Z."""
    a = np.deg2rad(deg)
    return np.array([0.0, np.cos(a), np.sin(a)])

if __name__ == "__main__":
    R, r = 2.8, 1.0
    zc = 4.0
    NB = np.array([0.0, 1.0, 0.0])          # vertical ring: plane XZ
    print(f"{'tilt':>5} {'offset':>7}  linked  crossings   hole-vs-tube")
    for tilt in (45, 60, 75, 90):
        NA = tilt_normal(tilt)
        for off in (R*0.7, R, R*1.3):
            CA = np.array([-off/2, 0.0, zc])
            CB = np.array([ off/2, 0.0, zc])
            ok, n = linked(CA, NA, R, CB, NB, R)
            ok2, n2 = linked(CB, NB, R, CA, NA, R)
            print(f"{tilt:5.0f} {off:7.2f}  {str(ok and ok2):>6}  {n},{n2}"
                  f"        hole {R-r:.2f} vs tube+clr {r+0.55:.2f}"
                  f" {'OK' if R-r > r+0.55 else 'TIGHT'}")
