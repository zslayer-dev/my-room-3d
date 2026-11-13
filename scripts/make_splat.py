"""points.npz → 스플랫 PLY (K 만 바꿔 다시 만들 때).

사용:  python make_splat.py out48/points.npz out48/scene_k1.0.ply --K 1.0 [--n-keep 0]
"""
import argparse, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("ply")
ap.add_argument("--K", type=float, default=1.4)
ap.add_argument("--n-keep", type=int, default=0)
a = ap.parse_args()

d = np.load(a.npz); xyz, rgb = d["xyz"], d["rgb"]
if a.n_keep and a.n_keep < len(xyz):
    idx = np.linspace(0, len(xyz) - 1, a.n_keep).astype(np.int64)
    xyz, rgb = xyz[idx], rgb[idx]
SH_C0 = 0.28209479177387814
n = len(xyz)
extent = float((xyz.max(0) - xyz.min(0)).max())
r = a.K * extent / np.sqrt(n)
o = np.zeros((n, 17), np.float32)
o[:, 0:3] = xyz; o[:, 1] *= -1; o[:, 2] *= -1          # 모델 Y-아래 → three.js Y-위
o[:, 6:9] = (rgb.astype(np.float32) / 255 - 0.5) / SH_C0
o[:, 9] = np.log(0.9 / 0.1)
o[:, 10:13] = np.log(r)
o[:, 13] = 1.0
props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
hdr = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
       + "".join(f"property float {q}\n" for q in props) + "end_header\n").encode()
t = time.time()
with open(a.ply, "wb") as f:
    f.write(hdr); f.write(o.tobytes())
s = os.path.getsize(a.ply)
print(f"{a.ply}: 알갱이 {n:,}, K {a.K}, 반지름 {r:.5f}, 긴 변 {extent:.3f}, "
      f"{s:,}B, 알갱이당 {(s-len(hdr))/n:.0f}B, {time.time()-t:.1f}초")
