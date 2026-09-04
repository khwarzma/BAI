"""Train the canonical BAI v1.1 byte-level BPE tokenizer.

This script is intentionally manual-only. It reads train.jsonl from the
protected backup archive and writes models/tokenizer.json and models/vocab.json.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


ARCHIVE_MEMBER = "data/processed/train.jsonl"
VOCAB_SIZE = 50_257
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def iter_texts(archive_path: Path) -> Iterator[str]:
    """Yield source text without extracting or modifying the archive."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.extractfile(ARCHIVE_MEMBER)
        if member is None:
            raise FileNotFoundError(f"Missing {ARCHIVE_MEMBER} in {archive_path}")
        for line_number, raw_line in enumerate(member, start=1):
            try:
                record = json.loads(raw_line)
                text = record["text"]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid record at line {line_number}") from exc
            if not isinstance(text, str) or not text:
                raise ValueError(f"Missing non-empty text at line {line_number}")
            yield text


def train(archive_path: Path, output_dir: Path) -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_texts(archive_path), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer size is {tokenizer.get_vocab_size(with_added_tokens=True)}, "
            f"expected {VOCAB_SIZE}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as vocab_file:
        json.dump(tokenizer.get_vocab(), vocab_file, ensure_ascii=False, indent=2, sort_keys=True)
        vocab_file.write("\n")

    print(f"Saved {output_dir / 'tokenizer.json'}")
    print(f"Saved {output_dir / 'vocab.json'}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size(with_added_tokens=True)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("zg_files/bai_dataset_backup.tar.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
