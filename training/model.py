import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class BaiMicroEncoder(nn.Module):
    """
    Lightweight Micro-Transformer Encoder with multi-task classification heads.
    Language-agnostic architecture for classification tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        num_categories: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Multi-task classification heads
        self.category_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_categories),
        )
        
        self.otp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=math.sqrt(2 / self.d_model))
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Micro-Transformer encoder.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
                           1 for tokens to attend to, 0 for padding.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - logits_category: Classification logits (batch_size, num_categories)
            - logits_otp: Binary OTP logits (batch_size, 1)
            - confidence: Confidence scores (batch_size, 1)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.positional_embedding(positions)
        
        # Prepare attention mask for transformer
        # Convert (batch_size, seq_len) where 1=real token, 0=pad
        # to src_key_padding_mask (batch_size, seq_len) where True=pad (ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        
        # Encode
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply layer norm for safe CLS token extraction
        if self.encoder.norm is not None:
            encoded = self.encoder.norm(encoded)
        
        # Replace CLS pooling with length-aware mean pooling weighted by attention mask.
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size()).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = encoded.mean(dim=1)

        # Multi-task heads
        logits_category = self.category_head(pooled_output)
        logits_otp = self.otp_head(pooled_output)
        confidence = self.confidence_head(pooled_output)
        
        return logits_category, logits_otp, confidence
