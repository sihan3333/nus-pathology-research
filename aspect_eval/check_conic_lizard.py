"""检查 CoNIC 小块能否代替原版 Lizard，给 PathoVernier 重建图片。

思路：CoNIC 把每张 Lizard 原图按 256x256 切块（数量 = 行数 x 列数）。
如果 PathoVernier 的裁剪坐标 (x0, y0) 正好落在 CoNIC 的网格上，
那么第 k 块就是要找的图。用 CoNIC 标注里的细胞核数量和 patch_index.csv
最后一列对比，看哪种网格排法对得上。

用法：python check_conic_lizard.py <patch_index.csv> <CoNIC_2022 目录>
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

index_csv, conic_dir = sys.argv[1], Path(sys.argv[2])

with open(index_csv) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [r for r in reader if r[1] == "lizard"]
print("patch_index 表头:", header)
print("Lizard 题目图片数:", len(rows))

with open(conic_dir / "patch_info.csv") as f:
    names = [line.strip() for line in f][1:]
by_src = defaultdict(list)
for i, name in enumerate(names):
    src, k = name.rsplit("-", 1)
    by_src[src].append((int(k), i))
labels = np.load(conic_dir / "labels.npy", mmap_mode="r")


def n_nuclei(i):
    inst = np.asarray(labels[i, :, :, 0])
    return len(np.unique(inst)) - 1


missing = sorted({r[2] for r in rows} - by_src.keys())
print("CoNIC 里找不到的原图:", missing if missing else "无")

# 两种网格排法：按行（先横着排）/ 按列（先竖着排）
hits = Counter()
shown = 0
for r in rows:
    src, x0, y0 = r[2], int(r[3]), int(r[4])
    if src not in by_src:
        continue
    a, b = int(r[6]), int(r[7])
    n = len(by_src[src])
    k_to_i = dict(by_src[src])
    for hname, (H, W) in {"H=a,W=b": (a, b), "H=b,W=a": (b, a)}.items():
        R, C = math.ceil(H / 256), math.ceil(W / 256)
        if R * C != n:
            continue
        for order, k in {"按行": (y0 // 256) * C + x0 // 256,
                         "按列": (x0 // 256) * R + y0 // 256}.items():
            if k in k_to_i:
                got = n_nuclei(k_to_i[k])
                if got == int(r[-1]):
                    hits[(hname, order)] += 1
                if shown < 6:
                    print(f"  {r[0]}: 索引最后一列={r[-1]}，{hname} {order} 第{k}块细胞核数={got}")
    shown += 1

print("\n各种排法中，细胞核数完全对上的题目数（共", len(rows), "）：")
for key, v in hits.most_common():
    print(f"  {key[0]} {key[1]}: {v}")
if not hits:
    print("  一个都没对上：最后一列可能不是细胞核总数，需要换别的方法核对")
