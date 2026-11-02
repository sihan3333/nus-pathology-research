"""检查 CoNSeP 数据集是否完整（非官方来源下载后用）。

用法：python check_consep.py <解压后的 CoNSeP 目录>
官方版本：Train 27 张、Test 14 张，1000x1000 的 PNG；
每张图在 Labels 下有同名 .mat，含 inst_map / type_map / inst_type / inst_centroid。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

EXPECTED = {"Train": 27, "Test": 14}
MAT_KEYS = {"inst_map", "type_map", "inst_type", "inst_centroid"}


def check_split(root, split, n_expected):
    ok = True
    img_dir, lab_dir = root / split / "Images", root / split / "Labels"
    imgs = sorted(img_dir.glob("*.png"))
    print(f"[{split}] 图片 {len(imgs)} 张（应为 {n_expected}）")
    ok &= len(imgs) == n_expected
    for img_path in imgs:
        mat_path = lab_dir / (img_path.stem + ".mat")
        try:
            img = Image.open(img_path)
            img.load()
        except Exception as e:
            print(f"  ✗ {img_path.name} 打不开：{e}")
            ok = False
            continue
        if img.size != (1000, 1000):
            print(f"  ✗ {img_path.name} 尺寸是 {img.size}，应为 (1000, 1000)")
            ok = False
        if not mat_path.exists():
            print(f"  ✗ 缺少标注 {mat_path.name}")
            ok = False
            continue
        try:
            mat = loadmat(mat_path)
        except Exception as e:
            print(f"  ✗ {mat_path.name} 读不了：{e}")
            ok = False
            continue
        missing = MAT_KEYS - mat.keys()
        if missing:
            print(f"  ✗ {mat_path.name} 缺少字段 {sorted(missing)}")
            ok = False
            continue
        n_inst = len(np.unique(mat["inst_map"])) - 1
        if n_inst != len(mat["inst_type"]):
            print(f"  ✗ {mat_path.name} 实例数 {n_inst} 和 inst_type 长度 {len(mat['inst_type'])} 对不上")
            ok = False
    return ok


def main():
    root = Path(sys.argv[1])
    ok = all([check_split(root, s, n) for s, n in EXPECTED.items()])
    print("\n结论：", "✅ 完整" if ok else "❌ 有问题，见上面的 ✗")


if __name__ == "__main__":
    main()
