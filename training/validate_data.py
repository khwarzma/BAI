#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
CHECKPOINT_ROOT = ROOT / "data" / "checkpoint"

REQUIRED_FIELDS = {"text", "category_label", "otp_label", "language", "dialect"}
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


def validate_record(record: Dict[str, Any], hash_registry: sqlite3.Connection) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(record, dict):
        return False, "record is not an object", {}

    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        return False, f"missing required keys: {sorted(missing)}", {}

    text = str(record.get("text", ""))
    category = str(record.get("category_label", ""))
    if category not in VALID_CATEGORIES:
        return False, f"invalid category_label: {category}", {}

    language = str(record.get("language", "")).lower()
    dialect = str(record.get("dialect", ""))
    if language not in {"ar", "en"}:
        return False, "language must be 'ar' or 'en'", {}
    if not dialect:
        return False, "dialect must be supplied", {}

    record["text"] = normalize_text(text)
    if category == "BAIT":
        otp_value = float(record.get("otp_label", 0.0))
        confidence = float(record.get("confidence_multiplier", 0.0))
        if otp_value != 1.0:
            return False, "BAIT records must have otp_label = 1.0", {}
        if confidence != 1.5:
            return False, "BAIT records must have confidence_multiplier = 1.5", {}

    record_hash = sha256_hex(json.dumps(record, sort_keys=True, ensure_ascii=False))
    row = hash_registry.execute("SELECT 1 FROM seen_hashes WHERE hash = ?", (record_hash,)).fetchone()
    if row is not None:
        return False, "duplicate hash", {}
    hash_registry.execute("INSERT INTO seen_hashes (hash, created_at) VALUES (?, datetime('now'))", (record_hash,))
    hash_registry.commit()

    record["language"] = language
    record["dialect"] = dialect
    return True, "ok", record


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
    bait_otp_valid = sum(
        1
        for record in valid_records
        if str(record.get("category_label", "")) == "BAIT" and float(record.get("otp_label", 0.0)) == 1.0
    )

    return {
        "total_valid_records": len(valid_records),
        "invalid_records": invalid_count,
        "duplicate_records": duplicate_count,
        "category_distribution": dict(sorted(category_counter.items())),
        "dialect_distribution": dict(sorted(dialect_counter.items())),
        "language_distribution": dict(sorted(language_counter.items())),
        "bait_otp_valid_records": bait_otp_valid,
        "bait_records": category_counter.get("BAIT", 0),
        "dataset_balance": {
            "category_coverage": sorted(category_counter.keys()),
            "dialect_coverage": sorted(dialect_counter.keys()),
            "all_29_dialects_present": len(dialect_counter) >= 29,
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

    hash_db_path = CHECKPOINT_ROOT / "hash_registry_validation.db"
    if hash_db_path.exists():
        hash_db_path.unlink()

    hash_registry = sqlite3.connect(str(hash_db_path))
    hash_registry.execute(
        "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
    )
    hash_registry.commit()

    valid_records: List[Dict[str, Any]] = []
    invalid_count = 0
    duplicate_count = 0

    for record in iter_raw_records(raw_root):
        ok, reason, cleaned = validate_record(record, hash_registry)
        if not ok:
            if reason == "duplicate hash":
                duplicate_count += 1
            invalid_count += 1
            continue
        valid_records.append(cleaned)

    stats = build_stats(valid_records, invalid_count, duplicate_count)
    (processed_root / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train, val, test = stratified_split(valid_records)
    write_jsonl(processed_root / "train.jsonl", train)
    write_jsonl(processed_root / "val.jsonl", val)
    write_jsonl(processed_root / "test.jsonl", test)

    print(f"Valid records: {len(valid_records)}")
    print(f"Invalid records: {invalid_count}")
    print(f"Duplicate records: {duplicate_count}")
    print(f"Saved splits: train={len(train)}, val={len(val)}, test={len(test)}")
    hash_registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
