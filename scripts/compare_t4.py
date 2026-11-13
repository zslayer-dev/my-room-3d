"""A770 복원과 코랩 T4 복원 비교.

사용: python scripts/compare_t4.py <A770 stats.json> <A770 points.npz> <T4 stats.json> <T4 points.npz>
점구름 거리는 '한쪽 표본 20만 개 → 다른 쪽 전체' 최근접 거리로 잰다.
(양쪽 다 표본만 쓰면 표본 간격 자체가 거리로 잡힌다 — 같은 A770 끼리도 5.5mm 가 나옴)
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree

sa, pa, st, pt = sys.argv[1:5]
a_stats, t_stats = json.load(open(sa)), json.load(open(st))
a, t = np.load(pa), np.load(pt)

print("| 항목 | A770 | T4 |\n|---|---|---|")
print(f"| 복원 시간(초) | {a_stats['recon_sec']:.1f} | {t_stats['recon_sec']:.1f} (첫 실행 {t_stats['recon_warm_sec']:.1f}) |")
print(f"| 최대 GPU 메모리(GB) | {a_stats['peak_gb']:.2f} | {t_stats['peak_gb']:.2f} |")
print(f"| 점 (원래) | {a_stats['points_raw']:,} | {t_stats['points_raw']:,} |")
print(f"| 점 (튄 점 제외) | {a_stats['splats']:,} | {t_stats['splats']:,} |")
print(f"| 퍼짐 비 | {a_stats['spread_ratio']:.4f} | {t_stats['spread_ratio']:.4f} |")

diag = float(np.linalg.norm(a["xyz"].max(0) - a["xyz"].min(0)))
d = np.linalg.norm(a["cams"][:, :3, 3] - t["cams"][:, :3, 3], axis=1)
print(f"\n카메라 위치 차이: 평균 {d.mean()*1000:.2f}mm, 최대 {d.max()*1000:.2f}mm (장면 대각선 {diag:.2f}m)")

rng = np.random.default_rng(1)
q = a["xyz"][rng.choice(len(a["xyz"]), 200_000, replace=False)]
d = cKDTree(t["xyz"]).query(q)[0]
print(f"점구름 거리 A770 표본 → T4 전체: 중앙값 {np.median(d)*1000:.2f}mm, 95% {np.percentile(d,95)*1000:.2f}mm")
s2 = a["xyz"][rng.choice(len(a["xyz"]), 200_000, replace=False)]
d = cKDTree(s2).query(q)[0]
print(f"(참고) 표본끼리 기준값 A770 ↔ A770: 중앙값 {np.median(d)*1000:.2f}mm")
