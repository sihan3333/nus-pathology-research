"""从医生阅片轨迹提取“看片重要区域”（viewing ROIs）并生成热力图，按 Ghezloo et al. 2024 的做法。

文献：Ghezloo et al., "Robust ROI Detection in Whole Slide Images Guided by Pathologists' Viewing Patterns",
J Imaging Inform Med 2024（DOI 10.1007/s10278-024-01202-x），以及作者公开代码
github.com/fGhezloo/ROI-Localization-melanoma（importData.py、soft_ROIs.py）。

每一步的出处：
  1. 合并帧为视野记录   —— 我们自己的适配：本阅片器约每 0.1 秒记一帧（画面不动也记），
                           Ghezloo 的日志是一次操作一条；所以把位置和倍率都不变的连续帧合并成一条记录，
                           持续时间 = 到下一条记录开始的时间。
  2. 去掉超出切片范围的视野、持续 > 60 秒的记录   —— importData.clean_tracking_data
  3. 放大峰：倍率比前后两条都高                      —— soft_ROIs.extract_zoompeaks
  4. 注视：持续 > 2 秒且倍率 > 5                     —— soft_ROIs.extract_fixations
  5. 慢速平移：倍率 > 5 且不变、相邻视野中心位移 < 100 × 倍率，
     连续一段累计持续 > 1 秒                         —— soft_ROIs.extract_slowpanning
     （位移的计算函数 helper.py 不在公开仓库里，这里用视野中心的欧氏距离，0 级像素——我们自己的选择）
  6. 热力图：把视野范围内的格子加上该记录的持续时间，再归一化到 0–1
                                                      —— soft_ROIs.generate_heatmaps（格子 = 16 像素）
     论文写的是只累加上面 3–5 提取的重要视野（默认）；作者代码实际调用时累加的是清洗后的全部视野
     （--all-viewports 复现这种做法）。
  7. 与医生框比较：热力图以 0.5 二值化后算 IoU / precision / recall / F1（论文 Quantitative Assessment）；
     另算 CC、NSS（Chakraborty et al., MICCAI 2024 引 Bylinskii et al., TPAMI 2018）
     以及热力值落在框内的比例（类比 Nan et al., Nat Commun 2025 的“注视点落在病灶内比例”）。

用法：
  python viewing_heatmap.py --traj <某个 .jsonl> --slide-dir <TCGA-BRCA/raw> --out <输出目录>
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def load_trajectory(path):
    meta, frames, annotations = None, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "meta":
                meta = r
            elif r.get("type") == "frame":
                frames.append(r)
            elif r.get("type") == "finally_result_snapshot":
                annotations = (r.get("data") or {}).get("annotations", [])
    return meta, frames, annotations


def frames_to_entries(frames):
    """步骤 1：位置和倍率不变的连续帧合并成一条视野记录。"""
    entries = []
    for fr in frames:
        key = (fr["x"], fr["y"], fr["w"], fr["h"], fr.get("mag", 0))
        if entries and entries[-1]["key"] == key:
            continue
        entries.append({"key": key, "t": fr["t"] / 1000, "x": fr["x"], "y": fr["y"],
                        "w": fr["w"], "h": fr["h"], "zoom": fr.get("mag", 0)})
    last_t = frames[-1]["t"] / 1000 if frames else 0
    for a, b in zip(entries, entries[1:] + [None]):
        a["duration"] = (b["t"] if b else last_t) - a["t"]
    return entries


def clean_entries(entries, width, height, max_duration=60):
    """步骤 2：去掉超出切片范围的视野和持续过长的记录，并算相邻视野中心位移。"""
    kept = [e for e in entries
            if e["x"] > 0 and e["y"] > 0 and e["x"] + e["w"] <= width and e["y"] + e["h"] <= height
            and e["duration"] <= max_duration]
    for a, b in zip([None] + kept, kept):
        b["displacement"] = 0.0 if a is None else float(np.hypot(
            (b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2), (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)))
    return kept


def zoom_peaks(entries):
    return [b for a, b, c in zip(entries, entries[1:], entries[2:]) if b["zoom"] > a["zoom"] and c["zoom"] < b["zoom"]]


def fixations(entries, min_duration=2, min_zoom=5):
    return [e for e in entries[1:] if e["duration"] > min_duration and e["zoom"] > min_zoom]


def slow_pannings(entries, min_zoom=5, pan=100, min_duration=1):
    picked, run, run_dur = [], [], 0.0
    old_zoom = entries[0]["zoom"] if entries else 0
    for i in range(1, len(entries)):
        e = entries[i]
        if e["zoom"] <= min_zoom:
            continue
        if e["zoom"] == old_zoom:
            if e["displacement"] < pan * e["zoom"]:
                run_dur += e["duration"]
                run = run or [i - 1]
                run.append(i)
                continue
            if run_dur > min_duration:
                picked.extend(run)
        else:
            old_zoom = e["zoom"]
            if run_dur > min_duration:
                picked.extend(run)
        run, run_dur = [], 0.0
    return [entries[i] for i in sorted(set(picked))]


def viewing_rois(entries):
    """步骤 3–5 的并集（同一条记录只算一次）。"""
    seen, out = set(), []
    for e in zoom_peaks(entries) + fixations(entries) + slow_pannings(entries):
        if id(e) not in seen:
            seen.add(id(e))
            out.append(e)
    return out


def heatmap(entries, width, height, cell):
    """步骤 6：按持续时间累加到视野覆盖的格子，再归一化到 0–1。"""
    grid = np.zeros((int(np.ceil(height / cell)), int(np.ceil(width / cell))))
    for e in entries:
        x0, y0 = int(e["x"] // cell), int(e["y"] // cell)
        grid[y0:y0 + int(e["h"] // cell), x0:x0 + int(e["w"] // cell)] += e["duration"]
    return grid / grid.max() if grid.max() > 0 else grid


def box_mask(boxes, shape, cell):
    mask = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1, _ in boxes:
        mask[int(y0 // cell):int(np.ceil(y1 / cell)), int(x0 // cell):int(np.ceil(x1 / cell))] = True
    return mask


def annotation_boxes(annotations):
    boxes = []
    for a in annotations:
        pts = (a.get("geometry_data") or {}).get("coordinates") or []
        if a.get("geometry_type") == "rect" and len(pts) == 2:
            (xa, ya), (xb, yb) = pts
            boxes.append((min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), a.get("notes") or ""))
    return boxes


def compare_with_boxes(heat, mask, threshold=0.5):
    """步骤 7：热力图 vs 医生框。"""
    if heat.max() <= 0 or not mask.any():
        return {}
    pred = heat >= threshold
    tp, fp, fn = (pred & mask).sum(), (pred & ~mask).sum(), (~pred & mask).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    z = (heat - heat.mean()) / heat.std() if heat.std() > 0 else heat * 0
    return {
        "iou": round(float(tp / (tp + fp + fn)), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 3),
        "cc": round(float(np.corrcoef(heat.ravel(), mask.ravel().astype(float))[0, 1]), 3),
        "nss": round(float(z[mask].mean()), 3),
        "heat_share_in_boxes": round(float(heat[mask].sum() / heat.sum()), 3),
        "box_area_share": round(float(mask.mean()), 4),
    }


def analyse(path, cell=16, all_viewports=False):
    meta, frames, annotations = load_trajectory(path)
    width, height = meta["image_width"], meta["image_height"]
    entries = clean_entries(frames_to_entries(frames), width, height)
    rois = entries if all_viewports else viewing_rois(entries)
    heat = heatmap(rois, width, height, cell)
    boxes = annotation_boxes(annotations)
    wsi = (meta.get("medical_context") or {}).get("wsi") or {}
    summary = {
        "wsi_name": wsi.get("wsi_name") or meta.get("wsi_name"),
        "user_id": meta.get("user_id"),
        "frames": len(frames), "entries": len(entries), "viewing_rois": len(rois),
        "zoom_peaks": len(zoom_peaks(entries)), "fixations": len(fixations(entries)),
        "slow_pannings": len(slow_pannings(entries)),
        "minutes": round(sum(e["duration"] for e in entries) / 60, 2),
        "max_zoom": round(max((e["zoom"] for e in entries), default=0), 1),
        "n_boxes": len(boxes),
        **compare_with_boxes(heat, box_mask(boxes, heat.shape, cell)),
    }
    return summary, heat, boxes, (width, height)


def find_slide(slide_dir, name):
    hits = glob.glob(os.path.join(slide_dir, "**", name), recursive=True) if slide_dir and name else []
    return hits[0] if hits else None


def thumbnail(slide_path, width, height, size):
    scale = size / max(width, height)
    tw, th = max(1, round(width * scale)), max(1, round(height * scale))
    if slide_path:
        try:
            try:
                import openslide
                img = openslide.OpenSlide(slide_path).get_thumbnail((tw, th))
            except ImportError:
                import tiffslide
                img = tiffslide.TiffSlide(slide_path).get_thumbnail((tw, th))
            return img.convert("RGB").resize((tw, th)), scale
        except Exception as e:  # noqa: BLE001
            print("读取切片失败，改用灰底：", e)
    return Image.new("RGB", (tw, th), (220, 220, 220)), scale


def overlay(heat, boxes, img, scale):
    v = np.clip(heat, 0, 1)
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (255 * (1 - v)).astype(np.uint8)
    rgba[..., 3] = (200 * np.sqrt(v)).astype(np.uint8)
    layer = Image.fromarray(rgba, "RGBA").resize(img.size, Image.NEAREST)
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (x0, y0, x1, y1, _) in enumerate(boxes, 1):
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=(0, 90, 255), width=3)
        draw.text((x0 * scale + 4, y0 * scale + 2), str(i), fill=(0, 90, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--slide-dir", default="")
    ap.add_argument("--out", default="vis")
    ap.add_argument("--cell", type=int, default=16, help="格子边长（0 级像素），作者代码为 16")
    ap.add_argument("--all-viewports", action="store_true", help="累加全部清洗后视野（作者代码的做法）")
    ap.add_argument("--thumb", type=int, default=1600)
    args = ap.parse_args()

    summary, heat, boxes, (width, height) = analyse(args.traj, args.cell, args.all_viewports)
    img, scale = thumbnail(find_slide(args.slide_dir, summary["wsi_name"]), width, height, args.thumb)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, summary["wsi_name"].split(".")[0] + ("_allvp" if args.all_viewports else ""))
    overlay(heat, boxes, img, scale).save(stem + "_viewing.png")
    with open(stem + "_viewing.json", "w") as f:
        json.dump({**summary, "boxes": [{"box": b[:4], "notes": b[4]} for b in boxes]}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))
    print("图片：", stem + "_viewing.png")


if __name__ == "__main__":
    main()
