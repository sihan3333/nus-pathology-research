"""按论文附录 D.4 的基线提示词，在 PathoVernier 上评测原版 Qwen3-VL-8B。

和官方 aspect/eval_pathovernier.py 的区别：
  1. 加上论文给基线用的 system prompt（要求先在 <observe> 里报计数，最后写 FINAL: 选项）；
  2. user prompt 按论文模板；题目文字和官方脚本给 ASPECT 的完全相同（论文 C.1：所有模型收到相同的题目和选项）；
  3. 支持断点续跑：已经做过的题会跳过，容器重启后重新运行同一条命令即可。
图像处理（缩放到 512x512）、贪心解码、答案解析都直接复用官方代码；输出上限 512 token（论文表 C.9）。
输出格式与官方一致，可直接用 aspect.score_pathovernier 打分。

用法（PYTHONPATH 指向官方代码目录，才能导入 aspect 模块）：
  PYTHONPATH=/home/zsh/aspect/ASPECT CUDA_VISIBLE_DEVICES=2 python /home/zsh/aspect/qwen_baseline_eval.py \
      --model <Qwen3-VL-8B-Instruct 目录> --bench <pathovernier.jsonl> --images <images 目录> \
      --out /home/zsh/aspect/outputs --tag qwen3vl
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from aspect.inference import load, prepare_image
from aspect.scoring.answer_parser import parse_pred

# 论文附录 D.4「PathoVernier: Baseline evaluation」原文
SYSTEM_PROMPT = (
    "You are an expert pathologist analyzing an H&E-stained histology image. You MUST base your answer "
    "strictly on what is visible in THIS image: actually look at and count the nuclei you can see. "
    "Never answer from priors, definitions, or the wording of the question alone — if you did not look "
    "at the image, you cannot answer.\n"
    "Think step by step.\n"
    "Then output ONE JSON object wrapped in <observe>...</observe>: FIRST the whole-image count of each "
    "main nucleus type you can see — epithelial-like, stromal-like, lymphocyte, inflammatory, and tumor "
    "separately when identifiable (do NOT report a type that is not present in this patch) — THEN the "
    "specific quantities the question asks you to report. Finally, output a line ‘FINAL: X’ where X "
    "is EXACTLY one of the allowed option tokens, copied verbatim."
)
USER_TEMPLATE = (
    "{question}, Choose exactly one of these option tokens: {options}. Respond with your reasoning, "
    "then <observe>{{...}}</observe>, then a final line ‘FINAL: <token>’."
)


@torch.inference_mode()
def generate(model, processor, image, question, options, max_new_tokens):
    user = USER_TEMPLATE.format(question=question.rstrip(), options="{" + ", ".join(options) + "}")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[prepare_image(image)], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, help="pathovernier.jsonl")
    ap.add_argument("--images", required=True, help="reconstruct_images.py 生成的目录")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="qwen3vl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（试跑用），0 表示全部")
    ap.add_argument("--full-question", action="store_true",
                    help="保留题目末尾 'In <observe>...' 的计数键名说明（不加时和官方给 ASPECT 的题目一样）")
    args = ap.parse_args()
    torch.manual_seed(42)

    rows = [json.loads(l) for l in open(args.bench)]
    rows.sort(key=lambda x: (x["skill"], x["item_id"]))
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}_raw.jsonl"
    done = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done[r["item_id"]] = r
    todo = [g for g in rows if g["item_id"] not in done]
    print(f"共 {len(rows)} 题，已完成 {len(rows) - len(todo)}，本次要跑 {len(todo)}", flush=True)
    n_correct = sum(done[g["item_id"]]["correct"] for g in rows if g["item_id"] in done)

    if todo:
        model, processor = load(args.model)
    n_truncated = 0
    with open(path, "a") as f:
        for i, g in enumerate(todo, len(rows) - len(todo) + 1):
            image = Image.open(Path(args.images) / g["dataset"] / f"{g['patch_id']}.png")
            question = g["question"] if args.full_question else re.split(r"\n?In <observe>", g["question"])[0].rstrip()
            raw = generate(model, processor, image, question, g["options"], args.max_new_tokens)
            n_truncated += "FINAL" not in raw.upper()
            pred = parse_pred(raw, g["options"])
            n_correct += pred == g["answer"]
            f.write(json.dumps(dict(item_id=g["item_id"], skill=g["skill"], dataset=g["dataset"],
                                    gold=g["answer"], pred=pred, correct=pred == g["answer"],
                                    raw=raw), ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(rows)}  acc={n_correct / i:.3f}  本次没写 FINAL 的回复={n_truncated}", flush=True)
    print(f"n={len(rows)}  acc={n_correct / max(len(rows), 1):.4f}  -> {path}")


if __name__ == "__main__":
    main()
