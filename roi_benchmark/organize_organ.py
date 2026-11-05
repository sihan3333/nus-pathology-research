"""把某个部位的轨迹和报告复制到自己的目录，并生成一张总表。

原始目录只读：这里只读取和复制，不改动、不删除任何原文件。切片原图不复制，总表里记录共享盘路径。
已存在且内容相同的文件跳过；已存在但内容不同时报错停止，不覆盖。

用法：
  python organize_organ.py --organ breast --out /data/nas_tmp/zsh/brca_benchmark
"""
import argparse
import csv
import filecmp
import glob
import json
import os
import re
import shutil

SRC = "/data/nas_tmp/ldk_tmp/viewer_and_report"
OWN_ROOT = "/data/nas_tmp/zsh/"


def read_trajectory(path):
    meta, n_frames, annotations = None, 0, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "meta":
                meta = r
            elif r.get("type") == "frame":
                n_frames += 1
            elif r.get("type") == "finally_result_snapshot":
                annotations = (r.get("data") or {}).get("annotations", [])
    return meta, n_frames, annotations


def safe_copy(src, dst):
    if os.path.exists(dst):
        if not filecmp.cmp(src, dst, shallow=False):
            raise RuntimeError(f"目标已存在且内容不同，不覆盖：{dst}")
        return
    shutil.copy2(src, dst)
    if not filecmp.cmp(src, dst, shallow=False):
        raise RuntimeError(f"复制后内容不一致：{dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="breast")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if not out.startswith(OWN_ROOT):
        raise SystemExit(f"输出目录必须在 {OWN_ROOT} 下")

    traj_out, report_out = os.path.join(out, "trajectory"), os.path.join(out, "report")
    os.makedirs(traj_out, exist_ok=True)
    os.makedirs(report_out, exist_ok=True)

    with open(f"{SRC}/trajectory_wsi_mapping.csv", encoding="utf-8-sig") as f:
        wsi_paths = {r["trajectory_name"]: r["wsi_path"] for r in csv.DictReader(f)}
    report_dir = f"{SRC}/PathTrace_viewer/report"
    reports = sorted(os.listdir(report_dir))

    rows = []
    for path in sorted(glob.glob(f"{SRC}/PathTrace_viewer/trajectory/*.jsonl")):
        with open(path) as f:
            first = json.loads(f.readline())
        if ((first.get("medical_context") or {}).get("wsi") or {}).get("organ_code") != args.organ:
            continue
        meta, n_frames, annotations = read_trajectory(path)
        name = os.path.basename(path)
        safe_copy(path, os.path.join(traj_out, name))

        wsi_name = meta.get("wsi_name") or ""
        match = re.match(r"BL-\d+-\d+", wsi_name)
        code = match.group(0) if match else ""
        hits = [r for r in reports if code and r.startswith(code)]
        for r in hits:
            safe_copy(os.path.join(report_dir, r), os.path.join(report_out, r))

        wsi_path = wsi_paths.get(name, "")
        tcga = ((meta.get("medical_context") or {}).get("wsi") or {}).get("wsi_name") or os.path.basename(wsi_path)
        rows.append({
            "tcga_file": tcga, "pathology_code": code, "trajectory": name,
            "report_files": ";".join(hits), "n_reports": len(hits),
            "wsi_path": wsi_path, "wsi_exists": bool(wsi_path) and os.path.exists(wsi_path),
            "n_frames": n_frames, "n_boxes": len(annotations), "user_id": meta.get("user_id"),
            "image_width": meta.get("image_width"), "image_height": meta.get("image_height"), "mpp": meta.get("mpp"),
        })

    manifest = os.path.join(out, f"manifest_{args.organ}.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("轨迹", len(rows), "目录内文件", len(os.listdir(traj_out)))
    print("报告：有 1 份", sum(r["n_reports"] == 1 for r in rows), "多份", sum(r["n_reports"] > 1 for r in rows),
          "没有", sum(r["n_reports"] == 0 for r in rows), "目录内文件", len(os.listdir(report_out)))
    print("切片路径：有效", sum(r["wsi_exists"] for r in rows), "无效", sum(not r["wsi_exists"] for r in rows))
    print("无视野帧", [r["tcga_file"] for r in rows if r["n_frames"] == 0])
    print("示例", rows[0])
    print("报告文件名示例", reports[:3])
    print("总表", manifest)


if __name__ == "__main__":
    main()
