import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List
import math

from model import BaiMicroEncoder


class BaiDataset(Dataset):
    """Dataset loader for BAI training data."""
    
    def __init__(self, jsonl_path: str, max_seq_length: int = 512):
        self.data = []
        self.max_seq_length = max_seq_length
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'][:self.max_seq_length], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'][:self.max_seq_length], dtype=torch.long),
            'category_label': torch.tensor(item['category_label'], dtype=torch.long),
            'otp_label': torch.tensor(item['otp_label'], dtype=torch.float),
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Collate batch with padding to max length within batch."""
    max_len = max(x['input_ids'].shape[0] for x in batch)
    
    input_ids_padded = []
    attention_mask_padded = []
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad input_ids with pad_token_id
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids_padded.append(padded_input_ids)
        
        # Pad attention_mask with 0 (padding indicator)
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask_padded.append(padded_attention_mask)
    
    return {
        'input_ids': torch.stack(input_ids_padded),
        'attention_mask': torch.stack(attention_mask_padded),
        'category_labels': torch.stack([x['category_label'] for x in batch]),
        'otp_labels': torch.stack([x['otp_label'] for x in batch]),
    }


class BaiTrainer:
    """Training orchestrator with auto-checkpointing and FP16 mixed precision."""
    
    def __init__(
        self,
        model: BaiMicroEncoder,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 10000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 500,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        category_weight: float = 0.7,
        otp_weight: float = 0.3,
        conf_weight: float = 0.2,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler()
        
        self.category_weight = category_weight
        self.otp_weight = otp_weight
        self.conf_weight = conf_weight
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.otp_loss_fn = nn.BCEWithLogitsLoss()
        self.conf_loss_fn = nn.MSELoss()
        
        self.global_step = 0
    
    def _create_scheduler(self):
        """Linear warmup with cosine decay scheduler."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps)))))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _compute_loss(self, logits_category, logits_otp, confidence, category_labels, otp_labels):
        """Compute weighted multi-task loss with calibrated confidence loss."""
        category_loss = self.category_loss_fn(logits_category, category_labels)
        otp_loss = self.otp_loss_fn(logits_otp.squeeze(-1), otp_labels)

        with torch.no_grad():
            preds = torch.argmax(logits_category, dim=-1)
            correct_mask = (preds == category_labels).float().unsqueeze(-1)

        conf_loss = self.conf_loss_fn(confidence, correct_mask)

        total_loss = (
            self.category_weight * category_loss
            + self.otp_weight * otp_loss
            + self.conf_weight * conf_loss
        )
        return total_loss, category_loss, otp_loss, conf_loss
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            category_labels = batch['category_labels'].to(self.device)
            otp_labels = batch['otp_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Determine device type for autocast
            device_type = 'cuda' if self.device == 'cuda' else 'cpu'
            
            with autocast(device_type=device_type):
                logits_category, logits_otp, confidence = self.model(
                    input_ids, attention_mask
                )
                loss, cat_loss, otp_loss, conf_loss = self._compute_loss(
                    logits_category, logits_otp, confidence, category_labels, otp_labels
                )
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'cat_loss': cat_loss.item(),
                'otp_loss': otp_loss.item(),
                'conf_loss': conf_loss.item(),
                'step': self.global_step,
            })
            
            # Auto-checkpoint
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
                val_loss = self.evaluate()
                print(f"Validation Loss at Step {self.global_step}: {val_loss:.4f}")
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                category_labels = batch['category_labels'].to(self.device)
                otp_labels = batch['otp_labels'].to(self.device)
                
                logits_category, logits_otp, confidence = self.model(
                    input_ids, attention_mask
                )
                loss, _, _, _ = self._compute_loss(
                    logits_category, logits_otp, confidence, category_labels, otp_labels
                )
                total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded: {checkpoint_path}")


def train(
    train_data_path: str = "data/processed/train.jsonl",
    val_data_path: str = "data/processed/val.jsonl",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    pad_token_id: int = 0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 500,
):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = BaiDataset(train_data_path)
    val_dataset = BaiDataset(val_data_path)
    
    # Create collate function with pad_token_id
    def collate_with_pad(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_with_pad,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_pad,
    )
    
    # Initialize model
    model = BaiMicroEncoder(
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_length=512,
        num_categories=5,
    )
    
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)
    
    # Initialize trainer
    trainer = BaiTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        device=str(device),
        conf_weight=0.2,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_loss = trainer.train_epoch()
        val_loss = trainer.evaluate()
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        trainer.save_checkpoint(f"epoch_{epoch + 1}")
    
    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    train()
