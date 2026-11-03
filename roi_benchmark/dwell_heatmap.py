"""从一条医生阅片轨迹算出“停留热力图”，和医生画的框一起叠加到切片缩略图上（只读数据）。

轨迹格式（PathTrace_viewer/trajectory/*.jsonl）：
  第 1 行 meta：wsi 尺寸、mpp、medical_context.wsi.wsi_name（TCGA 原始文件名）等；
  中间 frame 行：t（毫秒）、x/y/w/h（当前屏幕视野在切片 0 级像素坐标里的矩形）、mag（倍率）；
  最后一行 finally_result_snapshot：医生画的标注框和文字描述。

停留时间的算法：
  阅片器大约每 0.1 秒记一帧，画面不动也会记录，所以每一帧的停留时间 = 到下一帧的间隔。
  间隔超过 --idle 秒（默认 60）视为阅片器开着没人看，不计；其余间隔最多按 --cap 秒（默认 1）计。
  每一帧把它的停留时间平均摊到视野覆盖的格子上（时间 / 视野面积），
  所以放大看小区域时单位面积得到的时间更多，缩小看全貌时被摊薄。

用法：
  python dwell_heatmap.py --traj <某个 .jsonl> --slide-dir <TCGA-BRCA/raw> --out <输出目录>
输出：<切片名>_overlay.png（缩略图 + 热力图 + 医生的框）和 <切片名>_summary.json。
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def load_trajectory(path):
    meta, frames, annotations = None, [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "meta":
                meta = r
            elif r.get("type") == "frame":
                frames.append(r)
            elif r.get("type") == "finally_result_snapshot":
                annotations = (r.get("data") or {}).get("annotations", [])
    return meta, frames, annotations


def frame_durations(frames, cap, idle):
    """每一帧的停留秒数：到下一帧的间隔，超过 idle 记 0，其余最多记 cap。最后一帧记 0。"""
    t = np.array([fr["t"] for fr in frames], dtype=float) / 1000
    gap = np.diff(t, append=t[-1] if len(t) else 0)
    return np.where(gap > idle, 0.0, np.minimum(gap, cap))


def dwell_map(frames, durations, width, height, cell):
    gw, gh = int(np.ceil(width / cell)), int(np.ceil(height / cell))
    grid = np.zeros((gh, gw))
    for fr, d in zip(frames, durations):
        if d <= 0:
            continue
        x0, y0 = max(fr["x"], 0), max(fr["y"], 0)
        x1, y1 = min(fr["x"] + fr["w"], width), min(fr["y"] + fr["h"], height)
        if x1 <= x0 or y1 <= y0:
            continue
        c0, r0 = int(x0 // cell), int(y0 // cell)
        c1, r1 = int(np.ceil(x1 / cell)), int(np.ceil(y1 / cell))
        grid[r0:r1, c0:c1] += d / ((r1 - r0) * (c1 - c0))
    return grid


def annotation_boxes(annotations):
    boxes = []
    for a in annotations:
        g = a.get("geometry_data") or {}
        pts = g.get("coordinates") or []
        if a.get("geometry_type") == "rect" and len(pts) == 2:
            (xa, ya), (xb, yb) = pts
            boxes.append((min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), a.get("notes") or ""))
    return boxes


def find_slide(slide_dir, name):
    hits = glob.glob(os.path.join(slide_dir, "**", name), recursive=True) if slide_dir else []
    return hits[0] if hits else None


def thumbnail(slide_path, width, height, size):
    """返回缩略图；读不了切片时用灰底代替，只看热力图和框的相对位置。"""
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


def heat_colors(v):
    """0–1 映射成 透明→黄→红 的 RGBA。"""
    v = np.clip(v, 0, 1)
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (255 * (1 - v)).astype(np.uint8)
    rgba[..., 3] = (200 * np.sqrt(v)).astype(np.uint8)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--slide-dir", default="")
    ap.add_argument("--out", default="vis")
    ap.add_argument("--cell", type=int, default=256, help="格子边长（0 级像素）")
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--idle", type=float, default=60.0)
    ap.add_argument("--thumb", type=int, default=1600, help="缩略图长边像素")
    args = ap.parse_args()

    meta, frames, annotations = load_trajectory(args.traj)
    wsi = (meta.get("medical_context") or {}).get("wsi") or {}
    name = wsi.get("wsi_name") or meta.get("wsi_name")
    width, height = meta["image_width"], meta["image_height"]
    boxes = annotation_boxes(annotations)

    dur = frame_durations(frames, args.cap, args.idle) if frames else np.zeros(0)
    grid = dwell_map(frames, dur, width, height, args.cell)
    covered = np.zeros_like(grid, dtype=bool)  # 框重叠时不重复计算
    for x0, y0, x1, y1, _ in boxes:
        covered[int(y0 // args.cell):int(np.ceil(y1 / args.cell)), int(x0 // args.cell):int(np.ceil(x1 / args.cell))] = True
    in_box = grid[covered].sum()
    total = grid.sum()

    slide_path = find_slide(args.slide_dir, name)
    img, scale = thumbnail(slide_path, width, height, args.thumb)
    if total > 0:
        norm = grid / np.percentile(grid[grid > 0], 99)
        heat = Image.fromarray(heat_colors(norm), "RGBA").resize(img.size, Image.NEAREST)
        img = Image.alpha_composite(img.convert("RGBA"), heat).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (x0, y0, x1, y1, _) in enumerate(boxes, 1):
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=(0, 90, 255), width=3)
        draw.text((x0 * scale + 4, y0 * scale + 2), str(i), fill=(0, 90, 255))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, name.split(".")[0])
    img.save(stem + "_overlay.png")
    summary = {
        "wsi_name": name, "slide_found": slide_path, "image_size": [width, height],
        "frames": len(frames), "active_minutes": round(float(dur.sum()) / 60, 2),
        "max_mag": max((fr.get("mag", 0) for fr in frames), default=0),
        "boxes": [{"box": b[:4], "notes": b[4]} for b in boxes],
        "dwell_share_in_boxes": round(float(in_box / total), 3) if total else None,
        "box_area_share": round(sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / (width * height), 4),
    }
    with open(stem + "_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "boxes"}, ensure_ascii=False))
    print("图片：", stem + "_overlay.png")


if __name__ == "__main__":
    main()
