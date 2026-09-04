"""Resource-safe BAI v1.1 CPU training entry point.

Run manually after training the tokenizer and preparing processed_v1.1 data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import BaiMicroEncoder


SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
NUM_WORKERS = 0
CHECKPOINT_DIR = Path("training/checkpoints")


class BaiDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets = []
        with open(path, "rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                self.offsets.append(offset)
        self._source = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._source is None:
            self._source = open(self.path, encoding="utf-8")
        self._source.seek(self.offsets[index])
        item = json.loads(self._source.readline())
        if len(item["input_ids"]) != SEQUENCE_LENGTH or len(item["attention_mask"]) != SEQUENCE_LENGTH:
            raise ValueError("Prepared records must contain exactly 256 positions")
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "category_labels": torch.tensor(item["category_label"], dtype=torch.long),
            "otp_labels": torch.tensor(item["otp_label"], dtype=torch.float32),
        }


class BaiTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int, learning_rate: float, checkpoint_interval: int):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.total_steps = max(1, epochs * len(train_loader))
        self.warmup_steps = max(1, self.total_steps // 10)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_scale)
        self.checkpoint_interval = checkpoint_interval
        self.global_step = 0
        self.category_loss = nn.CrossEntropyLoss()
        self.otp_loss = nn.BCEWithLogitsLoss()
        self.confidence_loss = nn.MSELoss()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def _lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _loss(self, outputs, batch):
        logits_category, logits_otp, confidence = outputs
        category = self.category_loss(logits_category, batch["category_labels"])
        otp = self.otp_loss(logits_otp.squeeze(-1), batch["otp_labels"])
        with torch.no_grad():
            target_confidence = (logits_category.argmax(-1) == batch["category_labels"]).float().unsqueeze(-1)
        confidence_loss = self.confidence_loss(confidence, target_confidence)
        return 0.7 * category + 0.3 * otp + 0.2 * confidence_loss

    def save_checkpoint(self, name: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, CHECKPOINT_DIR / f"{name}.pt")

    def evaluate(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                total += self._loss(self.model(batch["input_ids"], batch["attention_mask"]), batch).item()
        return total / max(1, len(self.val_loader))

    def run(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in progress:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model(batch["input_ids"], batch["attention_mask"]), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}", step=self.global_step)
                if self.global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            self.save_checkpoint(f"epoch_{epoch + 1}")
            print(f"validation_loss={self.evaluate():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_v1.1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    args = parser.parse_args()

    torch.set_num_threads(2)
    device = torch.device("cpu")
    train_loader = DataLoader(
        BaiDataset(str(args.data_dir / "train.jsonl")), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        BaiDataset(str(args.data_dir / "val.jsonl")), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=False,
    )
    model = BaiMicroEncoder(
        vocab_size=50_257, d_model=256, num_heads=8, num_layers=6,
        d_ff=1024, max_seq_length=SEQUENCE_LENGTH, num_categories=5,
    ).to(device)
    BaiTrainer(model, train_loader, val_loader, args.epochs, args.learning_rate,
               args.checkpoint_interval).run()


if __name__ == "__main__":
    main()
