"""在本地把停留密度热力图和医生框叠到切片缩略图上，用于目视检查。

热力图算法见 dwell_density.py。按“热力落在框内比例”的 5%、25%、50%、75%、95% 分位各取 1 张，
避免只看到结果好的切片；格子大小和模糊只用于显示，不改变统计结果。

用法：
  python local_overlay.py --data <本地数据_乳腺> --out <输出目录>
"""
import argparse
import csv
import os

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from dwell_density import density_map, stable_dwells
from viewing_heatmap import annotation_boxes, load_trajectory

PERCENTILES = [5, 25, 50, 75, 95]


def render(traj_path, thumb_path):
    meta, frames, annotations = load_trajectory(traj_path)
    width, height = meta["image_width"], meta["image_height"]
    dwells = stable_dwells(frames)
    thumb = Image.open(thumb_path).convert("RGB")
    scale = thumb.size[0] / width
    cell = max(1, round(1 / scale))
    heat = density_map(dwells, width, height, cell)
    layer = Image.fromarray((heat * 255).astype(np.uint8)).resize(thumb.size, Image.BILINEAR)
    layer = layer.filter(ImageFilter.GaussianBlur(3))
    v = np.asarray(layer, dtype=float) / 255
    rgba = (matplotlib.colormaps["inferno"](v) * 255).astype(np.uint8)
    rgba[..., 3] = (np.sqrt(v) * 200).astype(np.uint8)
    out = Image.alpha_composite(thumb.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1, _ in annotation_boxes(annotations):
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=(0, 120, 255), width=4)
    return out, len(dwells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.data, "manifest_breast.csv"), encoding="utf-8") as f:
        manifest = {r["trajectory"]: r for r in csv.DictReader(f)}
    with open(os.path.join(args.data, "stats_breast_dwell.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("heat_share_in_boxes")]
    rows.sort(key=lambda r: float(r["heat_share_in_boxes"]))

    for pct in PERCENTILES:
        r = rows[round((len(rows) - 1) * pct / 100)]
        traj = os.path.basename(r["traj"].replace("\\", "/"))
        tcga = manifest[traj]["tcga_file"].rsplit(".", 1)[0]
        image, n = render(os.path.join(args.data, "trajectory", traj), os.path.join(args.data, "thumbnails", tcga + ".jpg"))
        name = f"p{pct:02d}_{tcga[:12]}_share{float(r['heat_share_in_boxes']):.3f}.jpg"
        image.save(os.path.join(args.out, name), quality=88)
        print(name, "停留段", n, "IoU", r["iou"], "框面积", r["box_area_share"], image.size)


if __name__ == "__main__":
    main()
