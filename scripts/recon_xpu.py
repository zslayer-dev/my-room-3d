"""영상 하나 → 프레임 24장 → MapAnything(Intel Arc XPU) → 점 + 카메라 → 스플랫 PLY.

사용:  .venv-xpu/Scripts/python recon_xpu.py room.mp4 [--frames 24] [--K 1.4] [--n-keep 0]
결과는 out/ 아래에 저장됩니다.
"""
import argparse, glob, json, os, subprocess, sys, time

import numpy as np
import torch

HERE = os.getcwd()  # 프로젝트 루트(map-anything 이 있는 곳)에서 실행
sys.path.insert(0, os.path.join(HERE, "map-anything"))

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--pick", type=int, default=1, help="후보를 이만큼씩 묶어 가장 선명한 장을 고름")
ap.add_argument("--K", type=float, default=1.4)
ap.add_argument("--conf-pct", type=float, default=0, help="신뢰도 하위 몇 퍼센트 점을 버릴지 (0=안 버림)")
ap.add_argument("--n-keep", type=int, default=0)
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--device", default="xpu" if torch.xpu.is_available() else "cpu")
args = ap.parse_args()

OUT = args.out
FR = os.path.join(OUT, "frames")
os.makedirs(FR, exist_ok=True)
for f in glob.glob(os.path.join(FR, "*.jpg")):  # noqa: 이전 결과 정리
    os.remove(f)
dev = torch.device(args.device)

# Arc(XPU)에서 torch.linalg.solve 등이 드라이버 assert로 프로세스째 죽는다.
# 작은 선형대수만 CPU에서 풀고 결과를 원래 장치로 돌려보낸다.
def _cpu_linalg(fn):
    def wrap(*a, **kw):
        devs = [x.device for x in a if isinstance(x, torch.Tensor)]
        if not devs or devs[0].type != "xpu":
            return fn(*a, **kw)
        d = devs[0]
        out = fn(*[x.cpu() if isinstance(x, torch.Tensor) else x for x in a], **kw)
        if isinstance(out, torch.Tensor):
            return out.to(d)
        if isinstance(out, tuple):
            return type(out)(*(o.to(d) if isinstance(o, torch.Tensor) else o for o in out)) \
                if hasattr(out, "_fields") else tuple(o.to(d) if isinstance(o, torch.Tensor) else o for o in out)
        return out
    return wrap

if dev.type == "xpu":
    for name in ("solve", "inv", "svd", "eigh", "lstsq", "pinv", "det", "qr", "cholesky"):
        setattr(torch.linalg, name, _cpu_linalg(getattr(torch.linalg, name)))
    torch.inverse = _cpu_linalg(torch.inverse)
print(f"장치: {dev}", torch.xpu.get_device_name(0) if dev.type == "xpu" else "")

# 1. 프레임 고르게 뽑기 (길이와 상관없이 N장)
dur = float(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=nw=1:nk=1", args.video]).decode().strip())
C = args.frames * args.pick   # 후보 장수
fps = C / dur
t = time.time()
subprocess.run(["ffmpeg", "-v", "error", "-i", args.video, "-vf",
                f"fps={fps},scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'",
                "-q:v", "2", "-frames:v", str(C),
                os.path.join(FR, "c_%03d.jpg")], check=True)
cand = sorted(glob.glob(os.path.join(FR, "c_*.jpg")))
if args.pick > 1:
    # 후보를 pick 장씩 묶어 그중 가장 선명한(라플라시안 분산이 큰) 한 장만 남긴다.
    # 구간마다 한 장씩이라 촬영 경로를 고르게 덮으면서 흔들린 장을 피한다.
    import cv2
    sharp = [cv2.Laplacian(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var() for p in cand]
    chosen = []
    for i in range(0, len(cand), args.pick):
        grp = list(range(i, min(i + args.pick, len(cand))))
        chosen.append(max(grp, key=lambda j: sharp[j]))
    s_all = np.array(sharp); s_ch = s_all[chosen]
    print(f"선명도(라플라시안 분산) 후보 중앙값 {np.median(s_all):.0f}, 최저 {s_all.min():.0f} → "
          f"고른 장 중앙값 {np.median(s_ch):.0f}, 최저 {s_ch.min():.0f}")
else:
    chosen = list(range(len(cand)))
for k, j in enumerate(chosen):
    os.replace(cand[j], os.path.join(FR, f"f_{k+1:03d}.jpg"))
for p in glob.glob(os.path.join(FR, "c_*.jpg")):
    os.remove(p)
paths = sorted(glob.glob(os.path.join(FR, "f_*.jpg")))
print(f"영상 {dur:.1f}초 → 후보 {len(cand)}장 → 프레임 {len(paths)}장 ({time.time()-t:.1f}초)")

# 2. 모델 준비와 복원
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

t = time.time()
model = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
print(f"모델 준비 {time.time()-t:.0f}초")

views = load_images(paths)
print("모델 입력 크기", tuple(views[0]["img"].shape[-2:]))
if dev.type == "xpu":
    torch.xpu.reset_peak_memory_stats()
t = time.time()
if dev.type == "cuda":
    with torch.no_grad():
        pred = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                           use_amp=True, amp_dtype="fp16",
                           apply_mask=True, mask_edges=True, apply_confidence_mask=args.conf_pct > 0,
                   confidence_percentile=args.conf_pct or 10)
else:
    # model.infer 를 풀어서: forward 는 GPU(XPU), 후처리는 CPU.
    # Arc 에서는 후처리 단계의 작은 커널들이 드라이버 assert 로 죽는다.
    from mapanything.utils.inference import (validate_input_views_for_inference,
        preprocess_input_views_for_inference, postprocess_model_outputs_for_inference)
    skip = {"instance", "idx", "true_shape", "data_norm_type"}
    vv = validate_input_views_for_inference(views)
    for v in vv:
        for k in list(v):
            if k not in skip and hasattr(v[k], "to"):
                v[k] = v[k].to(dev)
    pv = preprocess_input_views_for_inference(vv)
    model._configure_geometric_input_config(use_calibration=True, use_depth=True, use_pose=True,
                                            use_depth_scale=True, use_pose_scale=True)
    with torch.inference_mode(), torch.autocast(dev.type, enabled=True, dtype=torch.float16):
        raw = model.forward(pv, memory_efficient_inference=True, minibatch_size=1)
    torch.xpu.synchronize()
    print(f"forward(GPU) {time.time()-t:.1f}초", flush=True)
    to_cpu = lambda d: {k: (x.float().cpu() if torch.is_tensor(x) and x.is_floating_point()
                            else x.cpu() if torch.is_tensor(x) else x) for k, x in d.items()}
    raw = [to_cpu(r) for r in raw]
    pv = [to_cpu(v) for v in pv]
    with torch.inference_mode():
        pred = postprocess_model_outputs_for_inference(raw_outputs=raw, input_views=pv,
                   apply_mask=True, mask_edges=True, apply_confidence_mask=args.conf_pct > 0,
                   confidence_percentile=args.conf_pct or 10)
t_rec = time.time() - t
peak = torch.xpu.max_memory_allocated() / 1e9 if dev.type == "xpu" else float("nan")
print(f"복원 {t_rec:.1f}초, 최대 GPU 메모리 {peak:.2f}GB")

# 3. 점 모으기, 튄 점 버리기, 카메라 퍼짐 비
pts, cols, cams = [], [], []
for v in pred:
    m = v["mask"][0, ..., 0].cpu().numpy().astype(bool)
    pts.append(v["pts3d"][0].float().cpu().numpy()[m])
    cols.append((v["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[m])
    cams.append(v["camera_poses"][0].float().cpu().numpy())
intr = np.stack([v["intrinsics"][0].float().cpu().numpy() for v in pred])   # 모델 입력 해상도 기준
model_wh = tuple(int(x) for x in pred[0]["img_no_norm"].shape[2:0:-1])      # (W, H)
xyz = np.concatenate(pts); rgb = np.concatenate(cols); cams = np.stack(cams)
n_raw = len(xyz)
lo = np.percentile(xyz, 0.5, axis=0); hi = np.percentile(xyz, 99.5, axis=0)
keep = np.all((xyz > lo) & (xyz < hi), axis=1)
xyz, rgb = xyz[keep], rgb[keep]
cpos = cams[:, :3, 3]
spread = float(np.linalg.norm(cpos.max(0) - cpos.min(0)))
diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
ratio = spread / diag
print(f"점 {n_raw:,} → 튄 점 제외 {len(xyz):,}")
print(f"카메라 퍼짐 {spread:.3f} / 장면 대각선 {diag:.3f} = 퍼짐 비 {ratio:.4f}",
      "(걸으며 찍은 것으로 보임)" if ratio > 0.1 else "(카메라가 거의 안 움직임 — 다시 찍기 검토)")
np.savez_compressed(os.path.join(OUT, "points.npz"), xyz=xyz, rgb=rgb, cams=cams,
                    intrinsics=intr, model_wh=np.array(model_wh),
                    frames=np.array([os.path.basename(p) for p in paths]))

# 4. 스플랫 PLY (INRIA, 17 floats = 68B/알갱이)
if args.n_keep and args.n_keep < len(xyz):
    idx = np.linspace(0, len(xyz) - 1, args.n_keep).astype(np.int64)
    xyz, rgb = xyz[idx], rgb[idx]
SH_C0 = 0.28209479177387814
n = len(xyz)
extent = float((xyz.max(0) - xyz.min(0)).max())
r = args.K * extent / np.sqrt(n)
p = xyz.astype(np.float32).copy(); p[:, 1] *= -1; p[:, 2] *= -1
o = np.zeros((n, 17), np.float32)
o[:, 0:3] = p
o[:, 6:9] = (rgb.astype(np.float32) / 255 - 0.5) / SH_C0
o[:, 9] = np.log(0.9 / 0.1)
o[:, 10:13] = np.log(r)
o[:, 13] = 1.0
props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
hdr = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
       + "".join(f"property float {q}\n" for q in props) + "end_header\n").encode()
t = time.time()
ply = os.path.join(OUT, "scene.ply")
with open(ply, "wb") as f:
    f.write(hdr); f.write(o.tobytes())
size = os.path.getsize(ply)
print(f"scene.ply 알갱이 {n:,}개, K {args.K}, 반지름 {r:.5f}, 장면 긴 변 {extent:.3f}")
print(f"파일 {size:,}B ({size/1e6:.1f}MB), 알갱이당 {(size-len(hdr))/n:.0f}B, 저장 {time.time()-t:.1f}초")

json.dump(dict(device=str(dev), frames=len(paths), video_sec=dur, recon_sec=t_rec,
               peak_gb=peak, points_raw=n_raw, splats=n, spread_ratio=ratio,
               K=args.K, radius=r, extent=extent, ply_bytes=size),
          open(os.path.join(OUT, "stats.json"), "w"), indent=2, ensure_ascii=False)
