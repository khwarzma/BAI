#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Set

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
CHECKPOINT_ROOT = ROOT / "data" / "checkpoint"

VALID_CATEGORIES = {"INBOX_PINNED", "INBOX", "BAIT", "BAIS", "BAIADS"}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def sha256_hex(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def ensure_dirs() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def validate_and_autofix_record(record: Dict[str, Any], seen_hashes: Set[str]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(record, dict):
        return False, "record is not an object", {}

    # 1. توحيد واستخراج النص
    text = str(record.get("text", "")).strip()
    if not text or len(text) < 10:
        return False, "text empty or too short (< 10 chars)", {}

    # 2. توحيد التصنيف
    category = str(record.get("category_label") or record.get("category") or "").strip()
    if category not in VALID_CATEGORIES:
        return False, f"invalid category: {category}", {}

    # 3. استنتاج وتثبيت اللهجة واللغة
    dialect = str(record.get("dialect", "")).strip()
    if not dialect:
        dialect = "global"

    language = str(record.get("language", "")).lower().strip()
    if not language:
        if dialect.startswith("ar"):
            language = "ar"
        elif dialect.startswith("en"):
            language = "en"
        else:
            language = "global"

    # 4. المعالجة التلقائية لعينات OTP والـ Confidence
    otp_label = float(record.get("otp_label", 1.0 if "otp" in dialect else 0.0))
    confidence_multiplier = float(record.get("confidence_multiplier", 1.5 if otp_label == 1.0 else 1.0))

    # 5. تنظيف وتطبيق المعايير
    clean_text = normalize_text(text)
    record_hash = sha256_hex(clean_text)
    if record_hash in seen_hashes:
        return False, "duplicate text content", {}
    
    seen_hashes.add(record_hash)

    cleaned_record = {
        "text": clean_text,
        "category_label": category,
        "otp_label": otp_label,
        "confidence_multiplier": confidence_multiplier,
        "language": language,
        "dialect": dialect
    }

    return True, "ok", cleaned_record


def iter_raw_records(raw_root: Path) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    if not raw_root.exists():
        return collected
    for jsonl_path in sorted(raw_root.rglob("*.jsonl")):
        for entry in read_jsonl(jsonl_path):
            collected.append(entry)
    return collected


def build_stats(valid_records: Sequence[Dict[str, Any]], invalid_count: int, duplicate_count: int) -> Dict[str, Any]:
    category_counter = Counter(record.get("category_label", "UNKNOWN") for record in valid_records)
    dialect_counter = Counter(record.get("dialect", "UNKNOWN") for record in valid_records)
    language_counter = Counter(record.get("language", "UNKNOWN") for record in valid_records)

    return {
        "total_valid_records": len(valid_records),
        "invalid_records": invalid_count,
        "duplicate_records": duplicate_count,
        "category_distribution": dict(sorted(category_counter.items())),
        "dialect_distribution": dict(sorted(dialect_counter.items())),
        "language_distribution": dict(sorted(language_counter.items())),
        "dataset_balance": {
            "category_coverage": sorted(category_counter.keys()),
            "dialect_coverage": sorted(dialect_counter.keys()),
            "dialects_count": len(dialect_counter),
        },
    }


def stratified_split(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record.get("category_label", "UNKNOWN"))].append(record)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    rng = random.Random(42)

    for category_records in by_category.values():
        shuffled = category_records[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = max(1, int(total * 0.8))
        val_end = train_end + max(1, int(total * 0.1))
        train.extend(shuffled[:train_end])
        val.extend(shuffled[train_end:val_end])
        test.extend(shuffled[val_end:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw BAI email samples and split into train/val/test datasets.")
    parser.add_argument("--raw-root", type=str, default=str(RAW_ROOT), help="Root directory for raw JSONL files.")
    parser.add_argument("--processed-root", type=str, default=str(PROCESSED_ROOT), help="Directory for validated and split outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)

    seen_hashes: Set[str] = set()
    valid_records: List[Dict[str, Any]] = []
    invalid_count = 0
    duplicate_count = 0

    print("🔍 Validating raw files with Auto-Fix and Normalization Enabled...")
    for record in iter_raw_records(raw_root):
        ok, reason, cleaned = validate_and_autofix_record(record, seen_hashes)
        if not ok:
            if reason == "duplicate text content":
                duplicate_count += 1
            invalid_count += 1
            continue
        valid_records.append(cleaned)

    stats = build_stats(valid_records, invalid_count, duplicate_count)
    (processed_root / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("✂️ Stratified Splitting into Train (80%) / Val (10%) / Test (10%)...")
    train, val, test = stratified_split(valid_records)
    write_jsonl(processed_root / "train.jsonl", train)
    write_jsonl(processed_root / "val.jsonl", val)
    write_jsonl(processed_root / "test.jsonl", test)

    print("=" * 60)
    print(f"✅ Valid Records Prepared : {len(valid_records):,}")
    print(f"⚠️ Invalid Skipped       : {invalid_count:,}")
    print(f"♻️ Duplicates Removed    : {duplicate_count:,}")
    print(f"📁 Saved Splits          : Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())