# TASK INSTRUCTION: Critical Refactoring for BAI Model Architecture & Training Pipeline

## Context & Objective
Following the audit report from Khwarzma Audit, we need to apply 3 critical technical fixes to our PyTorch model definition and training loop (`training/model.py`, `training/train.py`, and data generation) before launching the training on Google Colab. 

These updates ensure model confidence calibration, prevent legal execution risks, and replace standard [CLS] pooling with length-aware Mean Pooling for long-form email inputs.

---

## Required Action Items:

### 1. Refactor `training/model.py` (Mean Pooling Integration)
In the `forward` pass of `BaiMicroEncoder`, replace the single CLS token representation (`encoded[:, 0, :]`) with Mean Pooling weighted by the `attention_mask`.

**Implementation Target:**
```python
# Inside forward() method of BaiMicroEncoder:
if attention_mask is not None:
    mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size()).float()
    sum_embeddings = torch.sum(encoded * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    pooled_output = sum_embeddings / sum_mask
else:
    pooled_output = encoded.mean(dim=1)

logits_category = self.category_head(pooled_output)
logits_otp = self.otp_head(pooled_output)
confidence = self.confidence_head(pooled_output)

return logits_category, logits_otp, confidence
2. Update training/train.py (Confidence Head Loss Function)
In corporate production, the confidence output must be explicitly trained to reflect true classification accuracy.

Implementation Target:

Add conf_weight: float = 0.2 to BaiTrainer.__init__.

Initialize self.conf_loss_fn = nn.MSELoss().

Update _compute_loss method to compute target correctness dynamically:

Python
def _compute_loss(self, logits_category, logits_otp, confidence, category_labels, otp_labels):
    category_loss = self.category_loss_fn(logits_category, category_labels)
    otp_loss = self.otp_loss_fn(logits_otp.squeeze(-1), otp_labels)
    
    # Calculate empirical confidence target (1.0 if correct prediction, 0.0 otherwise)
    with torch.no_grad():
        preds = torch.argmax(logits_category, dim=-1)
        correct_mask = (preds == category_labels).float().unsqueeze(-1)
        
    conf_loss = self.conf_loss_fn(confidence, correct_mask)
    
    total_loss = (self.category_weight * category_loss) + \
                 (self.otp_weight * otp_loss) + \
                 (self.conf_weight * conf_loss)
                 
    return total_loss, category_loss, otp_loss, conf_loss
3. Ensure Dataset Balance & Validation (generate_data.py / validate_data.py)
Confirm dataset generation yields exactly 10,000 samples per dialect (19 Arabic + 10 English = 290,000 total across 5 categories: INBOX_PINNED, INBOX, BAIT, BAIS, BAIADS).

Include standalone datasets for general OTP (10,000 samples) and general ADS (10,000 samples).

Execute validate_data.py to ensure zero broken UTF-8 symbols and 100% strict JSON schema compliance before launching Colab training.

Execution Instructions:
Apply the C++ / Python modifications to training/model.py and training/train.py.

Verify syntax and unit test compatibility locally.

Save changes and confirm readiness for Colab execution.