"""코랩 T4 에서 out24s_c10 과 같은 24장으로 MapAnything 복원 → A770 결과와 비교용 stats/points 저장.

전제: /content/frames24s.zip 업로드됨, 런타임 T4 GPU.
A770 쪽 설정과 맞춤: fp16, memory_efficient, minibatch 1, 신뢰도 하위 10% 버림, 튄 점 0.5~99.5%.
"""
import glob, json, os, subprocess, sys, time, zipfile
import numpy as np, torch

assert torch.cuda.is_available(), "GPU 런타임이 아닙니다 (런타임 유형을 T4 로)"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

if not os.path.isdir("/content/map-anything"):
    subprocess.run("git clone -q --depth 1 https://github.com/facebookresearch/map-anything.git /content/map-anything"
                   " && pip install -q -e /content/map-anything", shell=True, check=True)
sys.path.insert(0, "/content/map-anything")

with zipfile.ZipFile("/content/frames24s.zip") as z:
    z.extractall("/content/c10")
paths = sorted(glob.glob("/content/c10/frames/f_*.jpg"))
print("프레임", len(paths))

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

t = time.time()
model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
t_load = time.time() - t
print(f"모델 준비 {t_load:.0f}초")

views = load_images(paths)
print("모델 입력 크기", tuple(views[0]["img"].shape[-2:]))

def run():
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        pred = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                           use_amp=True, amp_dtype="fp16",
                           apply_mask=True, mask_edges=True, apply_confidence_mask=True,
                           confidence_percentile=10)
    torch.cuda.synchronize()
    return pred, time.time() - t

_, t_warm = run()                      # 첫 실행: 커널 준비 포함
torch.cuda.reset_peak_memory_stats()
pred, t_rec = run()                    # 두 번째 실행을 측정값으로
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"복원 첫 실행 {t_warm:.1f}초, 측정 {t_rec:.1f}초, 최대 GPU 메모리 {peak:.2f}GB")

pts, cols, cams = [], [], []
for v in pred:
    m = v["mask"][0, ..., 0].cpu().numpy().astype(bool)
    pts.append(v["pts3d"][0].float().cpu().numpy()[m])
    cols.append((v["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[m])
    cams.append(v["camera_poses"][0].float().cpu().numpy())
xyz = np.concatenate(pts); rgb = np.concatenate(cols); cams = np.stack(cams)
n_raw = len(xyz)
lo = np.percentile(xyz, 0.5, axis=0); hi = np.percentile(xyz, 99.5, axis=0)
keep = np.all((xyz > lo) & (xyz < hi), axis=1)
xyz, rgb = xyz[keep], rgb[keep]
cpos = cams[:, :3, 3]
spread = float(np.linalg.norm(cpos.max(0) - cpos.min(0)))
diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
print(f"점 {n_raw:,} → 튄 점 제외 {len(xyz):,} | 퍼짐 비 {spread/diag:.4f}")

np.savez_compressed("/content/t4_points.npz", xyz=xyz, rgb=rgb, cams=cams,
                    frames=np.array([os.path.basename(p) for p in paths]))
stats = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, frames=len(paths),
             load_sec=t_load, recon_warm_sec=t_warm, recon_sec=t_rec, peak_gb=peak,
             points_raw=n_raw, splats=len(xyz), spread_ratio=spread / diag)
json.dump(stats, open("/content/t4_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
