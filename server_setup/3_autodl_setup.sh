#!/usr/bin/env bash
# ASPECT 评测复现：AutoDL 环境准备 + 数据下载
# 用法（在 AutoDL 终端里，建议先用无卡模式开机）：
#   bash 3_autodl_setup.sh
# 运行前需要你自己完成：
#   1. Hugging Face 上 Mikezcy/ASPECT-8B 和 Mikezcy/PathoVernier 的访问申请已通过
#   2. 已执行 hf auth login（token 自己粘贴，不要写进脚本）
#   3. Lizard / CoNSeP 需要 Warwick 账号登录才能下载，自己下好后上传到 $RAW/lizard 和 $RAW/consep
set -euo pipefail

ROOT=/root/autodl-tmp/aspect
RAW=$ROOT/raw
mkdir -p "$RAW"/{puma,pannuke,nucls,lizard,consep}
# HF 缓存默认在系统盘，放到数据盘避免写满
export HF_HOME=/root/autodl-tmp/hf_cache

# AutoDL 学术加速（访问 GitHub / Hugging Face / Zenodo 用）
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi

# ---------- 1. 代码与依赖 ----------
cd "$ROOT"
[ -d ASPECT ] || git clone https://github.com/ChyaZhang/ASPECT.git
pip install -U pip
pip install -r ASPECT/requirements.txt
pip install -U "huggingface_hub[cli]" gdown

# ---------- 2. 权重与基准（gated，需要已登录） ----------
hf auth whoami || { echo "请先执行 hf auth login"; exit 1; }
hf download Mikezcy/ASPECT-8B --local-dir "$ROOT/ASPECT-8B"
hf download Mikezcy/PathoVernier --repo-type dataset --local-dir "$ROOT/ASPECT/PathoVernier"
echo ">>> 先看一下 reconstruct_images.py 期望的目录结构："
python "$ROOT/ASPECT/PathoVernier/reconstruct_images.py" --help || true

# ---------- 3. 公开可直接下载的数据集 ----------
# PUMA（Zenodo，CC0）。只下 1024x1024 ROI 和标注，14 GB 的 context ROI 一般用不到
cd "$RAW/puma"
for f in 01_training_dataset_tif_ROIs.zip 01_training_dataset_geojson_nuclei.zip 01_training_dataset_geojson_tissue.zip; do
  [ -f "$f" ] || wget -c "https://zenodo.org/records/14869398/files/$f?download=1" -O "$f"
done
echo "6311f46efc043caf78b926f2942c4b1c  01_training_dataset_tif_ROIs.zip
7f47ccc71d94061b4dae058b5dc8f628  01_training_dataset_geojson_nuclei.zip
bc8bc686c77935380c53aa9b009742be  01_training_dataset_geojson_tissue.zip" | md5sum -c
for f in *.zip; do unzip -qn "$f"; done

# PanNuke（Warwick，CC BY-NC-SA 4.0，不需要登录）
cd "$RAW/pannuke"
for i in 1 2 3; do
  [ -f fold_$i.zip ] || wget -c "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_$i.zip"
  unzip -qn fold_$i.zip
done

# NuCLS corrected single-rater（Google Drive，CC0）。gdown 失败的话就手动下载再上传
cd "$RAW/nucls"
gdown --folder "https://drive.google.com/drive/folders/1eGlF9Dgu3WMEik4fqj0wJ13LKVufsfZ0" || \
  echo "!!! NuCLS 自动下载失败，请手动下载后放到 $RAW/nucls"

# ---------- 4. 需要手动准备的数据集 ----------
for d in lizard consep; do
  if [ -z "$(ls -A "$RAW/$d")" ]; then
    echo "!!! $RAW/$d 还是空的：请登录 Warwick 账号下载后上传到这里"
  fi
done

echo
echo "全部数据就绪后，对照 reconstruct_images.py 的 --help 填路径，例如："
echo "python $ROOT/ASPECT/PathoVernier/reconstruct_images.py --lizard $RAW/lizard --pannuke $RAW/pannuke \\"
echo "    --consep $RAW/consep --puma $RAW/puma --nucls $RAW/nucls --out $ROOT/ASPECT/PathoVernier/images"
echo
echo "然后换成 GPU 模式开机，先冒烟测试，再全量评测（MODEL 指向本地权重，避免重复下载）："
echo "cd $ROOT/ASPECT && python -m aspect.inference --model $ROOT/ASPECT-8B --image images/example_puma.png --question \"...\""
echo "cd $ROOT/ASPECT && MODEL=$ROOT/ASPECT-8B BENCH=PathoVernier/pathovernier.jsonl IMAGES=PathoVernier/images NUM_SHARDS=1 bash scripts/eval_pathovernier.sh"
