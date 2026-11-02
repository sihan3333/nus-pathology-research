#!/usr/bin/env bash
# ASPECT 评测复现：组里服务器版
# 约定（师姐要求）：代码在 /home 下，数据和模型放在 /data/nas_tmp 下自己的目录
# 用法：
#   bash 4_server_setup.sh          # 准备环境 + 下载
#   bash 4_server_setup.sh eval     # 数据就绪后，在 GPU 2 上跑评测
# 运行前需要你自己完成：
#   1. Hugging Face 上 Mikezcy/ASPECT-8B 和 Mikezcy/PathoVernier 的访问申请已通过
#   2. 已执行 hf auth login（token 自己粘贴，不要写进脚本）
#   3. Lizard / CoNSeP 需要 Warwick 账号登录下载，自己下好后放到 $RAW/lizard 和 $RAW/consep
#   4. 按集群说明配好外网代理：把 /data/nas_share/puzzlelogic/readonly/system_tools/install_clash.zip
#      复制到 /home 下解压，按里面的 readme 安装并打开代理（GitHub / Hugging Face / Zenodo 都要用）
#   5. 在「Cluster Usage Plan」表里登记要用的服务器和 GPU，用完再更新
# 集群规定：不要装 conda（用 venv 或 miniforge）；不要用 scp 往服务器传超过 10GB 的数据
set -euo pipefail

# ---------- 路径与 GPU（按需修改） ----------
WORK=/home/zsh/aspect                 # 代码、虚拟环境、评测输出
STORE=/data/nas_tmp/zsh/aspect        # 权重、数据集、HF 缓存（/data/nas_tmp/zsh 是自己的目录）
GPU=2                                 # 0、1 有别人的 vLLM 服务，不要用

RAW=$STORE/raw
MODEL_DIR=$STORE/ASPECT-8B
BENCH_DIR=$STORE/PathoVernier
export HF_HOME=$STORE/hf_cache
# 官方地址连不上时，可以试试镜像（gated 仓库需要已登录的 token）：
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$WORK" "$RAW"/{puma,pannuke,nucls,lizard,consep}

# ---------- 评测（bash 4_server_setup.sh eval） ----------
if [ "${1:-}" = "eval" ]; then
  cd "$WORK/ASPECT"
  source "$WORK/venv/bin/activate"
  nvidia-smi --query-gpu=index,memory.used --format=csv
  # 官方 eval_pathovernier.sh 会把分片 i 绑到 GPU i（NUM_SHARDS=1 时就是 GPU 0），所以这里直接调用各步骤
  CUDA_VISIBLE_DEVICES=$GPU python -m aspect.eval_pathovernier --model "$MODEL_DIR" \
      --bench "$BENCH_DIR/pathovernier.jsonl" --images "$BENCH_DIR/images" --out outputs
  python -m aspect.merge_shards --bench "$BENCH_DIR/pathovernier.jsonl" --out outputs
  python -m aspect.score_pathovernier outputs/aspect_raw.jsonl --bench "$BENCH_DIR/pathovernier.jsonl"
  exit 0
fi

# ---------- 1. 代码与依赖 ----------
# 外网连通性检查（不通就先去开 clash 代理）
curl -sI -m 10 https://huggingface.co >/dev/null || { echo "!!! 连不上 huggingface.co，请先按集群说明打开 clash 代理"; exit 1; }
cd "$WORK"
[ -d ASPECT ] || git clone https://github.com/ChyaZhang/ASPECT.git
[ -d venv ] || python3 -m venv venv     # 集群不让装 conda；需要 Python 3.11 的话改用 miniforge
source venv/bin/activate
pip install -U pip
pip install -r ASPECT/requirements.txt
pip install -U "huggingface_hub[cli]" gdown

# ---------- 2. 权重与基准（gated，需要已登录） ----------
hf auth whoami || { echo "请先执行 hf auth login"; exit 1; }
hf download Mikezcy/ASPECT-8B --local-dir "$MODEL_DIR"
hf download Mikezcy/PathoVernier --repo-type dataset --local-dir "$BENCH_DIR"
echo ">>> reconstruct_images.py 期望的参数："
python "$BENCH_DIR/reconstruct_images.py" --help || true

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
echo "全部数据就绪后，对照上面 reconstruct_images.py 的 --help 填路径，例如："
echo "python $BENCH_DIR/reconstruct_images.py --lizard $RAW/lizard --pannuke $RAW/pannuke \\"
echo "    --consep $RAW/consep --puma $RAW/puma --nucls $RAW/nucls --out $BENCH_DIR/images"
echo
echo "冒烟测试（GPU $GPU）："
echo "cd $WORK/ASPECT && CUDA_VISIBLE_DEVICES=$GPU python -m aspect.inference --model $MODEL_DIR --image images/example_puma.png --question \"...\""
echo "全量评测：bash $0 eval"
