"""按组内渲染图的定义算停留段和停留密度热力图（只读轨迹，不读切片）。

定义来源：/data/nas_tmp/ldk_tmp/Pathtrace/track_analysis 的 run_summary.json 和 heatmap.jpg 图上说明。
  - 停留段：连续帧 x,y,w,h,z 完全相同；帧间隔上限 1000 毫秒；至少 1.0 秒。
  - 热力图：每段停留时长摊到整个视野面积，低倍大视野单位面积贡献小；模糊只用于显示，这里不做。
  - 超出切片的部分直接裁掉，不重新分配到组织上。

我们自己的选择（渲染图没写，待确认）：
  - 时长 = 该段最后一帧 t - 第一帧 t（all_dwells.csv 的 start_t_ms、end_t_ms 与此一致）；
  - 轨迹没有 z 字段时用 mag 代替；
  - 用 16 像素格子，沿用 viewing_heatmap.py；与医生框的比较也沿用它，二值化阈值 0.5。

用法：
  python dwell_density.py --check <某个轨迹 .jsonl>     # 打印停留段，用于和 all_dwells.csv 对照
  python dwell_density.py --organ breast --out <输出.csv>   # 批量
"""
import argparse
import csv
import glob
import json
import statistics

import numpy as np

from viewing_heatmap import annotation_boxes, box_mask, compare_with_boxes, load_trajectory

TRAJ_DIR = "/data/nas_tmp/ldk_tmp/viewer_and_report/PathTrace_viewer/trajectory"
REPORT = ["iou", "precision", "recall", "f1", "cc", "nss", "heat_share_in_boxes", "box_area_share"]


def frame_key(fr):
    return fr["x"], fr["y"], fr["w"], fr["h"], fr.get("z", fr.get("mag"))


def stable_dwells(frames, min_dwell=1.0, max_gap_ms=1000.0):
    """切出连续不动的段，返回达到最短时长的停留段。"""
    runs, run = [], []
    for fr in frames:
        if run and (frame_key(fr) != frame_key(run[-1]) or not 0 <= fr["t"] - run[-1]["t"] <= max_gap_ms):
            runs.append(run)
            run = []
        run.append(fr)
    if run:
        runs.append(run)
    dwells = []
    for r in runs:
        duration = (r[-1]["t"] - r[0]["t"]) / 1000
        if duration >= min_dwell:
            x, y, w, h, z = frame_key(r[0])
            dwells.append({"start_ms": r[0]["t"], "end_ms": r[-1]["t"], "duration": duration,
                           "frames": len(r), "x": x, "y": y, "w": w, "h": h, "z": z})
    return dwells


def density_map(dwells, width, height, cell):
    """时长 / 视野面积，累加到视野覆盖且在切片内的格子，再归一化到 0–1。"""
    grid = np.zeros((int(np.ceil(height / cell)), int(np.ceil(width / cell))))
    for d in dwells:
        if d["w"] <= 0 or d["h"] <= 0:
            continue
        x0, y0 = max(d["x"], 0), max(d["y"], 0)
        x1, y1 = min(d["x"] + d["w"], width), min(d["y"] + d["h"], height)
        if x1 <= x0 or y1 <= y0:
            continue
        grid[int(y0 // cell):int(np.ceil(y1 / cell)), int(x0 // cell):int(np.ceil(x1 / cell))] += \
            d["duration"] / (d["w"] * d["h"])
    return grid / grid.max() if grid.max() > 0 else grid


def analyse(path, cell=16):
    meta, frames, annotations = load_trajectory(path)
    width, height = meta["image_width"], meta["image_height"]
    dwells = stable_dwells(frames)
    heat = density_map(dwells, width, height, cell)
    boxes = annotation_boxes(annotations)
    wsi = (meta.get("medical_context") or {}).get("wsi") or {}
    return {
        "wsi_name": wsi.get("wsi_name") or meta.get("wsi_name"),
        "frames": len(frames), "dwells": len(dwells),
        "dwell_total_s": round(sum(d["duration"] for d in dwells), 2),
        "n_boxes": len(boxes),
        **compare_with_boxes(heat, box_mask(boxes, heat.shape, cell)),
    }


def organ_of(path):
    with open(path) as f:
        meta = json.loads(f.readline())
    return ((meta.get("medical_context") or {}).get("wsi") or {}).get("organ_code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--organ", default="breast")
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.check:
        meta, frames, _ = load_trajectory(args.check)
        print("首帧字段", sorted(frames[0]) if frames else None)
        dwells = stable_dwells(frames)
        print(f"frames {len(frames)}  dwells {len(dwells)}  total {sum(d['duration'] for d in dwells):.1f}s")
        for d in sorted(dwells, key=lambda d: -d["duration"]):
            print(d)
        return

    paths = [p for p in sorted(glob.glob(f"{args.traj_dir}/*.jsonl")) if organ_of(p) == args.organ]
    rows = [{**analyse(p), "traj": p} for p in paths]
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if "iou" in r]
    print(f"部位 {args.organ}: 轨迹 {len(rows)}，能比较的 {len(ok)}")
    print("每张停留段数 中位数", statistics.median(r["dwells"] for r in rows))
    print("每张停留总秒数 中位数", statistics.median(r["dwell_total_s"] for r in rows))
    for k in REPORT:
        v = [r[k] for r in ok]
        print(f"  {k:<20} 中位数 {statistics.median(v):.3f}   均值 {statistics.mean(v):.3f}")
    print("结果表：", args.out)


if __name__ == "__main__":
    main()
