"""检查交付文件的结构、对应关系、分组、敏感标识和长度报告。"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


units = load("knowledge_units.jsonl")
manifest = load("manifest.jsonl")
sources = json.loads((DATA / "sources.json").read_text())
registry = json.loads((DATA / "dataset_info.json").read_text())
by_unit = {u["unit_id"]: u for u in units}
by_source = {s["source_id"]: s for s in sources}
assert len(units) == len(by_unit) == len(manifest)
assert len(sources) == 13
assert sum(s["origin_candidate"] for s in sources) == 11
groups, texts, hashes = {}, {}, {}
for split in ("train", "valid", "auxiliary"):
    rows = load(f"risk_cpt_{split}.jsonl")
    assert registry[f"risk_cpt_{split}"]["columns"] == {"prompt": "text"}
    entries = [m for m in manifest if m["split"] == split]
    assert len(rows) == len(entries)
    groups[split] = {e["group_id"] for e in entries}
    texts[split] = [r["text"] for r in rows]
    hashes[split] = set()
    for line, (row, entry) in enumerate(zip(rows, entries), 1):
        assert set(row) == {"text"} and isinstance(row["text"], str)
        assert entry["line"] == line
        unit = by_unit[entry["unit_id"]]
        assert unit["split"] == split
        assert unit["source_ids"] == entry["source_ids"]
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        assert digest == entry["text_sha256"]
        assert digest not in hashes[split]
        hashes[split].add(digest)
        for field in ("definition", "mechanism", "evidence", "scope", "provenance"):
            assert unit[field]
        if unit["counterexamples"] is None:
            assert unit["counterexample_status"] == "not_provided"
            assert "所读来源未提供正常反例" in row["text"]
        else:
            assert len(unit["counterexamples"]) == len(unit["counterexample_item_types"])
        assert all(by_source[sid]["group_id"] == entry["group_id"] for sid in unit["source_ids"])
        assert all(unit["unit_id"] in by_source[sid]["unit_ids"] for sid in unit["source_ids"])
        for pattern in (r"https?://", r"\bou_[a-zA-Z0-9]+", r"\b(?:uid|did)\s*[:=]\s*\d+", r"\d{15,}", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", r"(?i)(?:access_token|app_secret|bearer)\s*[:=]"):
            assert not re.search(pattern, row["text"]), (unit["unit_id"], pattern)
assert not (groups["train"] & groups["valid"])
assert not (hashes["train"] & hashes["valid"])
assert not (groups["auxiliary"] & groups["valid"])
assert sum(len(v) for v in hashes.values()) == len(set.union(*hashes.values()))


def shingles(text):
    text = re.sub(r"\s+", "", text)
    return {text[i:i + 5] for i in range(max(0, len(text) - 4))}


best = (0.0, None, None)
for i, a in enumerate(texts["train"]):
    for j, b in enumerate(texts["valid"]):
        sa, sb = shingles(a), shingles(b)
        score = len(sa & sb) / len(sa | sb)
        if score > best[0]:
            best = (score, i + 1, j + 1)
assert best[0] < 0.6
token_check = json.loads((ROOT / "tokenizer_validation.json").read_text())
assert token_check["all_samples_fit"]
for split, counts in token_check["counts"].items():
    assert len(counts["lengths"]) == len(texts[split])
    assert counts["text_sha256"] == [hashlib.sha256(text.encode()).hexdigest() for text in texts[split]]
    assert counts["max_tokens_including_one_eos"] < 2048
import yaml
config = yaml.safe_load((ROOT / "configs" / "cpt_lora_candidate.yaml").read_text())
assert config["stage"] == "pt" and config["packing"] is False
assert config["dataset"] == "risk_cpt_train" and config["eval_dataset"] == "risk_cpt_valid"
assert config["freeze_vision_tower"] and config["freeze_multi_modal_projector"]
assert not config["freeze_language_model"]
assert "max_steps" not in config
report = {
    "status": "PASS",
    "checked": ["JSONL结构", "五字段与缺失标记", "来源对应与最小摘录", "11候选加2引用来源", "manifest行号和SHA256", "来源组隔离", "精确重复", "跨集合5字符片段近似重复", "训练正文敏感标识模式", "Tokenizer长度报告", "YAML及数据登记"],
    "counts": dict(Counter(u["split"] for u in units)),
    "max_cross_split_shingle_jaccard": round(best[0], 4),
    "groups": {k: sorted(v) for k, v in groups.items()},
    "training_executed": False,
    "limits": ["敏感标识检查包含人工选材和模式扫描，不保证识别所有敏感语义", "来源核对不是业务签核或平台生效核验", "近似重复检测不能证明不存在语义泄漏", "未运行训练框架参数解析、模型加载或GPU训练"],
}
(ROOT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
stats = json.loads((ROOT / "statistics.json").read_text())
stats["tokenizer_verified"] = True
stats["tokenizer_revision"] = token_check["tokenizer_revision"]
stats["token_count"] = {s: c["tokens_including_one_eos"] for s, c in token_check["counts"].items()}
(ROOT / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
