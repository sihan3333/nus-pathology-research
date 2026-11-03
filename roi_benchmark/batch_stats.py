"""对某个部位的全部医生阅片轨迹批量统计：停留时间有多少落在医生画的框里（只读数据，不读切片）。

每张切片算：
  active_minutes       去掉长空档后的有效看片时长；
  max_mag              最大放大倍数；
  n_boxes / box_types  医生标注的个数和几何类型（只有 rect 会参与统计）；
  box_area_share       框占整张切片面积的比例；
  dwell_share_in_boxes 停留时间落在框里的比例；
  enrichment           上面两者之比，>1 说明医生在框里停留得比平均多。

用法：
  python batch_stats.py --organ breast --out /data/nas_tmp/zsh/brca_benchmark/stats_breast.csv
"""
import argparse
import collections
import csv
import glob
import json
import statistics

import numpy as np

from dwell_heatmap import annotation_boxes, dwell_map, frame_durations, load_trajectory

TRAJ_DIR = "/data/nas_tmp/ldk_tmp/viewer_and_report/PathTrace_viewer/trajectory"


def organ_of(path):
    with open(path) as f:
        meta = json.loads(f.readline())
    return ((meta.get("medical_context") or {}).get("wsi") or {}).get("organ_code")


def stats_for(path, cell, cap, idle):
    meta, frames, annotations = load_trajectory(path)
    wsi = (meta.get("medical_context") or {}).get("wsi") or {}
    width, height = meta["image_width"], meta["image_height"]
    boxes = annotation_boxes(annotations)
    dur = frame_durations(frames, cap, idle) if frames else np.zeros(0)
    grid = dwell_map(frames, dur, width, height, cell)
    total = grid.sum()
    covered = np.zeros_like(grid, dtype=bool)
    for x0, y0, x1, y1, _ in boxes:
        covered[int(y0 // cell):int(np.ceil(y1 / cell)), int(x0 // cell):int(np.ceil(x1 / cell))] = True
    share = float(grid[covered].sum() / total) if total else None
    area = float(covered.mean())
    return {
        "wsi_name": wsi.get("wsi_name"),
        "user_id": meta.get("user_id"),
        "frames": len(frames),
        "active_minutes": round(float(dur.sum()) / 60, 2),
        "max_mag": round(max((fr.get("mag", 0) for fr in frames), default=0), 1),
        "n_boxes": len(annotations),
        "box_types": "|".join(sorted({a.get("geometry_type", "?") for a in annotations})),
        "box_area_share": round(area, 4),
        "dwell_share_in_boxes": None if share is None else round(share, 3),
        "enrichment": None if share is None or area == 0 else round(share / area, 1),
        "traj": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="breast")
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--idle", type=float, default=60.0)
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(f"{args.traj_dir}/*.jsonl")) if organ_of(p) == args.organ]
    rows = []
    for i, p in enumerate(paths, 1):
        rows.append(stats_for(p, args.cell, args.cap, args.idle))
        if i % 25 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["enrichment"] is not None]
    enr = [r["enrichment"] for r in ok]
    print(f"\n部位 {args.organ}: 轨迹 {len(rows)}，能算比例的 {len(ok)}（没有视野帧或没有矩形框的除外）")
    print("标注几何类型:", dict(collections.Counter(r["box_types"] for r in rows)))
    print("框占面积比例 中位数 %.3f；停留落在框内的比例 中位数 %.3f" % (
        statistics.median(r["box_area_share"] for r in ok), statistics.median(r["dwell_share_in_boxes"] for r in ok)))
    print("富集倍数 中位数 %.1f；>1 的 %d 张，>5 的 %d 张，<1 的 %d 张" % (
        statistics.median(enr), sum(e > 1 for e in enr), sum(e > 5 for e in enr), sum(e < 1 for e in enr)))
    print("结果表：", args.out)


if __name__ == "__main__":
    main()
