"""MapAnything 결과(points.npz + frames) → Brush 가 읽는 Nerfstudio 형식 데이터셋.

사용:  python export_brush.py out24s_c10 [--init 300000] [--holdout 4]
결과:  out24s_c10/brush/{images/, transforms.json, transforms_val.json, init.ply}

좌표 약속
- MapAnything camera_poses: 카메라→월드, OpenCV 축(x 오른쪽, y 아래, z 앞)
- Nerfstudio transform_matrix: 카메라→월드, OpenGL 축(x 오른쪽, y 위, z 뒤)
  → 회전의 y, z 열 부호를 뒤집는다. 월드 좌표는 그대로 두므로 init.ply 도 그대로.
- 내부값: 모델은 사진을 (sx, sy) 로 줄인 뒤 가운데를 잘라 518x294 로 쓴다.
  원본 프레임 기준으로 되돌린다:  f_o = f_m / s,  c_o = (c_m + 자른 여백) / s
"""
import argparse, json, os, shutil
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--init", type=int, default=300000, help="초기 점 개수 (고르게 뽑음)")
ap.add_argument("--holdout", type=int, default=0, help="N장마다 1장을 평가용으로 떼어 둠 (0=안 뗌)")
a = ap.parse_args()

d = np.load(os.path.join(a.run, "points.npz"))
cams, K, frames = d["cams"], d["intrinsics"], d["frames"]
Wm, Hm = (int(x) for x in d["model_wh"])
OUT = os.path.join(a.run, "brush")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "images"))

W, H = Image.open(os.path.join(a.run, "frames", str(frames[0]))).size
s = max(Wm / W, Hm / H) + 1e-8                      # mapanything 과 같은 계산
Ws, Hs = int(np.floor(W * s)), int(np.floor(H * s))
sx, sy = Ws / W, Hs / H                              # 실제로 적용된 축별 배율
ox, oy = 0.5 * (Ws - Wm), 0.5 * (Hs - Hm)            # 가운데 자르기 여백
print(f"원본 {W}x{H} → 축소 {Ws}x{Hs} → 자름 {Wm}x{Hm} (여백 {ox:.1f}, {oy:.1f})")

flip = np.diag([1.0, -1.0, -1.0, 1.0])
train, val = [], []
for i, (fn, c2w, k) in enumerate(zip(frames, cams, K)):
    fn = str(fn)
    shutil.copy(os.path.join(a.run, "frames", fn), os.path.join(OUT, "images", fn))
    fr = dict(file_path=f"images/{fn}",
              transform_matrix=(c2w.astype(np.float64) @ flip).tolist(),
              fl_x=float(k[0, 0] / sx), fl_y=float(k[1, 1] / sy),
              cx=float((k[0, 2] + ox) / sx), cy=float((k[1, 2] + oy) / sy),
              w=W, h=H)
    (val if a.holdout and i % a.holdout == a.holdout // 2 else train).append(fr)

# 초기 점: 위치 + 색 (점 구름 PLY)
xyz, rgb = d["xyz"], d["rgb"]
idx = np.linspace(0, len(xyz) - 1, min(a.init, len(xyz))).astype(np.int64)
xyz, rgb = xyz[idx].astype(np.float32), rgb[idx].astype(np.uint8)
v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
v["x"], v["y"], v["z"] = xyz.T
v["red"], v["green"], v["blue"] = rgb.T
with open(os.path.join(OUT, "init.ply"), "wb") as f:
    f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
    f.write(v.tobytes())

json.dump(dict(ply_file_path="init.ply", frames=train),
          open(os.path.join(OUT, "transforms.json"), "w"), indent=1)
if val:
    json.dump(dict(frames=val), open(os.path.join(OUT, "transforms_val.json"), "w"), indent=1)
k0 = train[0]
print(f"학습 {len(train)}장, 평가 {len(val)}장, 초기 점 {len(v):,}개")
print(f"예시 내부값 fx {k0['fl_x']:.1f} fy {k0['fl_y']:.1f} cx {k0['cx']:.1f} cy {k0['cy']:.1f}")
