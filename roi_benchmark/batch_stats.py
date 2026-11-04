"""对某个部位的全部医生阅片轨迹批量统计（只读数据，不读切片）。

方法见 viewing_heatmap.py（Ghezloo et al., J Imaging Inform Med 2024 及其公开代码）：
每张切片提取看片重要区域 → 时长热力图 → 与医生画的框比较（IoU/precision/recall/F1、CC、NSS、热力落在框内比例）。

用法：
  python batch_stats.py --organ breast --out /data/nas_tmp/zsh/brca_benchmark/stats_breast_v1.csv
"""
import argparse
import csv
import glob
import json
import statistics

from viewing_heatmap import analyse

TRAJ_DIR = "/data/nas_tmp/ldk_tmp/viewer_and_report/PathTrace_viewer/trajectory"
REPORT = ["iou", "precision", "recall", "f1", "cc", "nss", "heat_share_in_boxes", "box_area_share"]


def organ_of(path):
    with open(path) as f:
        meta = json.loads(f.readline())
    return ((meta.get("medical_context") or {}).get("wsi") or {}).get("organ_code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="breast")
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--all-viewports", action="store_true")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(f"{args.traj_dir}/*.jsonl")) if organ_of(p) == args.organ]
    rows = []
    for i, p in enumerate(paths, 1):
        summary = analyse(p, args.cell, args.all_viewports)[0]
        rows.append({**summary, "traj": p})
        if i % 25 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if "iou" in r]
    print(f"\n部位 {args.organ}: 轨迹 {len(rows)}，能比较的 {len(ok)}（没有重要视野或没有框的除外）")
    print("每张的重要视野数 中位数 %d；其中放大峰/注视/慢速平移 中位数 %d / %d / %d" % tuple(
        statistics.median(r[k] for r in rows) for k in ("viewing_rois", "zoom_peaks", "fixations", "slow_pannings")))
    for k in REPORT:
        v = [r[k] for r in ok]
        print(f"  {k:<20} 中位数 {statistics.median(v):.3f}   均值 {statistics.mean(v):.3f}")
    print("结果表：", args.out)


if __name__ == "__main__":
    main()
