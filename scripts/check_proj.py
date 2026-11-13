import json, sys, numpy as np
from PIL import Image
run = sys.argv[1]; fi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
T = json.load(open(f"{run}/brush/transforms.json")); fr = T["frames"][fi]
d = np.load(f"{run}/points.npz"); xyz, rgb = d["xyz"], d["rgb"]
c2w = np.array(fr["transform_matrix"]) @ np.diag([1, -1, -1, 1])   # GL → CV
w2c = np.linalg.inv(c2w)
pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
ok = pc[:, 2] > 0.05
pc, col = pc[ok], rgb[ok]
u = fr["fl_x"] * pc[:, 0] / pc[:, 2] + fr["cx"]; v = fr["fl_y"] * pc[:, 1] / pc[:, 2] + fr["cy"]
W, H = fr["w"], fr["h"]
m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, z, col = u[m].astype(int), v[m].astype(int), pc[m, 2], col[m]
order = np.argsort(-z)                       # 먼 것부터 → 가까운 것이 덮어씀
img = np.zeros((H, W, 3), np.uint8); img[v[order], u[order]] = col[order]
real = np.asarray(Image.open(f"{run}/brush/{fr['file_path']}").convert("RGB"))
Image.fromarray(np.concatenate([real, img], 1)).resize((W, H // 2)).save(f"{run}/proj_check_{fi}.jpg")
print(fr["file_path"], "투영된 점", m.sum())
