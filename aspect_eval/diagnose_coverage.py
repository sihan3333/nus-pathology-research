"""排查 Qwen 基线计数覆盖率偏低的原因（只读已有结果，不跑模型）。

把每条回复归到一类：
  no_observe   回复里没有 <observe>
  truncated    有 <observe> 但没有 </observe>，也没写 FINAL（写满 512 token 被截断）
  no_close_tag 有 <observe> 也写了 FINAL，只是漏了 </observe>
  bad_json     <observe> 里的内容不是合法 JSON
  missing_keys JSON 合法，但少了题目 "In <observe>..." 里要求的键
  complete     要求的键都在
注意：官方打分程序的解析可能比 json.loads 宽松，所以 complete 的数量
应接近但不一定等于打分结果里的 complete-count coverage（141）。

用法：
  python diagnose_coverage.py --raw outputs/qwen3vl_full_raw.jsonl --bench <pathovernier.jsonl>
"""
import argparse
import json
import re
from collections import Counter, defaultdict

CATS = ["complete", "missing_keys", "bad_json", "no_close_tag", "truncated", "no_observe"]


def required_keys(question):
    parts = re.split(r"In <observe>", question, maxsplit=1)
    if len(parts) < 2:
        return []
    # 键名形如 tumor_c1、stromal_like_num：小写字母加下划线
    return list(dict.fromkeys(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", parts[1])))


def classify(raw, keys):
    start = raw.rfind("<observe>")
    if start < 0:
        return "no_observe", None
    end = raw.find("</observe>", start)
    if end < 0:
        return ("truncated" if "FINAL" not in raw.upper() else "no_close_tag"), None
    body = raw[start + len("<observe>"):end].strip()
    body = re.sub(r"^```(?:json)?|```$", "", body).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return "bad_json", body
    if not isinstance(obj, dict):
        return "bad_json", body
    missing = [k for k in keys if k not in obj]
    return ("missing_keys" if missing else "complete"), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--examples", type=int, default=2, help="每类打印几个例子")
    args = ap.parse_args()

    bench = {g["item_id"]: g for g in map(json.loads, open(args.bench))}
    rows = [json.loads(l) for l in open(args.raw)]

    first = next(iter(bench.values()))["question"]
    print("题目里的计数说明示例：", first[first.find("In <observe>"):][:400], "\n")

    by_skill = defaultdict(Counter)
    examples = defaultdict(list)
    no_final = 0
    for r in rows:
        keys = required_keys(bench[r["item_id"]]["question"])
        cat, detail = classify(r["raw"], keys)
        by_skill[r["skill"]][cat] += 1
        no_final += "FINAL" not in r["raw"].upper()
        if len(examples[cat]) < args.examples:
            examples[cat].append((r["item_id"], keys, detail, r["raw"][-300:]))

    total = Counter()
    for c in by_skill.values():
        total.update(c)
    print(f"共 {len(rows)} 条；没写 FINAL 的 {no_final} 条\n")
    print(f"{'题型':<16}" + "".join(f"{c:>13}" for c in CATS))
    for skill in sorted(by_skill):
        print(f"{skill:<16}" + "".join(f"{by_skill[skill][c]:>13}" for c in CATS))
    print(f"{'合计':<16}" + "".join(f"{total[c]:>13}" for c in CATS))

    for cat in CATS[1:]:
        for item_id, keys, detail, tail in examples[cat]:
            print(f"\n=== {cat} | {item_id} | 要求的键 {keys}")
            if cat == "missing_keys":
                print(f"缺少：{detail}")
            print(f"回复末尾：...{tail}")


if __name__ == "__main__":
    main()
