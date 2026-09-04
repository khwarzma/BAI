"""Prepare fixed-length v1.1 JSONL data from the protected archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer


SEQUENCE_LENGTH = 256
SPLITS = ("train", "val", "test")
TEXT_KEYS = ("text", "raw_text", "body", "email", "content", "combined_text")


def extract_text(record: dict, location: str) -> str | None:
    """Return raw text when present, or None for pre-tokenized records."""
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    if "input_ids" in record and "attention_mask" in record:
        return None
    keys = ", ".join(sorted(record))
    raise ValueError(
        f"{location} has no supported raw text or pre-tokenized fields. "
        f"Available keys: {keys}"
    )


def iter_records(archive: tarfile.TarFile, member_name: str) -> Iterator[dict]:
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Missing {member_name} in archive")
    for line_number, raw_line in enumerate(member, start=1):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON at {member_name}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Record at {member_name}:{line_number} is not an object")
        extract_text(record, f"{member_name}:{line_number}")
        yield record


def prepare(archive_path: Path, tokenizer_path: Path, output_dir: Path) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer does not define [PAD]")
    if tokenizer.get_vocab_size(with_added_tokens=True) != 50_257:
        raise RuntimeError("Tokenizer must contain exactly 50,257 entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for split in SPLITS:
            source_name = f"data/processed/{split}.jsonl"
            target_path = output_dir / f"{split}.jsonl"
            count = 0
            with target_path.open("w", encoding="utf-8") as target:
                for record in iter_records(archive, source_name):
                    text = extract_text(record, f"{source_name}:{count + 1}")
                    if text is None:
                        raw_input_ids = record["input_ids"]
                        raw_attention_mask = record["attention_mask"]
                        if (not isinstance(raw_input_ids, list)
                                or not isinstance(raw_attention_mask, list)):
                            raise ValueError(f"Invalid pre-tokenized fields at {source_name}:{count + 1}")
                        if (len(raw_input_ids) > SEQUENCE_LENGTH
                                or len(raw_attention_mask) > SEQUENCE_LENGTH):
                            raise ValueError(f"Pre-tokenized record exceeds {SEQUENCE_LENGTH} positions")
                        input_ids = list(raw_input_ids)
                        attention_mask = list(raw_attention_mask)
                        padding = SEQUENCE_LENGTH - len(input_ids)
                        input_ids += [pad_id] * padding
                        attention_mask += [0] * padding
                    else:
                        encoding = tokenizer.encode(text)
                        input_ids = encoding.ids[:SEQUENCE_LENGTH]
                        attention_mask = [1] * len(input_ids)
                        padding = SEQUENCE_LENGTH - len(input_ids)
                        input_ids += [pad_id] * padding
                        attention_mask += [0] * padding
                    prepared = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "category_label": record["category_label"],
                        "otp_label": record["otp_label"],
                        "dialect": record.get("dialect"),
                    }
                    target.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                    count += 1
            print(f"Wrote {count} records to {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("zg_files/bai_dataset_backup.tar.gz"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v1.1"))
    args = parser.parse_args()
    prepare(args.archive, args.tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
