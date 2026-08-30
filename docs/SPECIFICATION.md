# BAI (Bareeed Artificial Intelligence) — Technical Specification

> **Specification Standard:** AM Standard (SAM Category)
>
>
> **Document Identifier:** AM-SPEC-BAI-1.0.0
> **Initiator:** Khwarzma
> **Target Application:** Bareeed Email Infrastructure
> **Status:** Active Technical Specification

---

## 1. System Constraints & Boundary Conditions

BAI v1 is classified as a Small Arabic Model (SAM) specialized for single-task transactional email processing. All operational implementations must adhere strictly to the numerical thresholds defined below:

| Metric / Parameter | Boundary Limit | Enforcement Mechanism |
| --- | --- | --- |
| **Model Weight Count** | $\le 25 \times 10^6$ parameters | Encoder layer parameter capping

 |
| **Inference Latency** | $\le 15 \text{ ms}$ per email payload | Native C++23 execution engine

 |
| **Active VRAM / RAM Footprint** | $\le 150 \text{ MB}$ total memory | Static memory allocation & Quantization

 |
| **CPU Execution Limits** | 2 Cores max, thread-bound | Cgroup thread limiting & CPU affinity

 |
| **Disk I/O Footprint** | $0 \text{ bytes}$ written during inference | Pure in-memory payload processing

 |
| **Quantization Format** | GGUF / ONNX (INT8 / INT4) | Symmetric uniform quantization

 |

---

## 2. Input Payload & Normalization Pipeline

The inference core accepts incoming email payloads containing raw headers, subject line, and plaintext content body. Pre-processing is executed via the `languages.json` configuration without mutating core C++ structures.

### 2.1 Text Normalization Rules

Before tokenization, text stream buffers undergo zero-allocation byte-level normalization:

* **Arabic Normalization:** Strip decorative Tatweel (`U+0640`), unify Alef variants (`أ`, `إ`, `آ` $\rightarrow$ `ا`), unify Yaa/Alif Maqsoora (`ى` $\rightarrow$ `ي`), and remove non-essential diacritics (Tashkeel).


* **English Normalization:** Case folding to lowercase, ASCII control character stripping, and whitespace collapse.
* **OTP Number Preservation:** Numeric sequences within token windows surrounding keyword boundaries (`code`, `OTP`, `رمز`, `تأكيد`) are preserved without token decomposition.



---

## 3. Classification Engine & JSON Schema Output

The inference engine evaluates the normalized sequence and emits a deterministic, strict JSON string payload to the Django host process.

### 3.1 Category Taxonomy Definitions

1. **`INBOX_PINNED`:** Critical transactional messages, direct account alerts, or explicit priority items requiring immediate user visibility.
2. **`INBOX`:** Standard user-to-user correspondence and routine communications requiring no automated intervention.
3. **`BAIT` (Bareeed Artificial Intelligence Trash):** Expired OTP notices, transient system logs, and obsolete transactional receipts.
4. **`BAIS` (Bareeed Artificial Intelligence Spam):** Unsolicited commercial bulk email, phishing attempts, and unauthorized distribution list mailings.
5. **`BAIADS` (Bareeed Artificial Intelligence Ads):** Recognized commercial promotions, newsletters, and marketing campaigns from verified senders.

### 3.2 Canonical JSON Output Schema

```json
{
  "category": "BAIT",
  "confidence": 0.9842,
  "is_otp": true,
  "otp_metadata": {
    "detected": true,
    "validity_window_minutes": 5,
    "expiration_action": "PURGE_IMMEDIATE"
  },
  "lifecycle_policy": {
    "auto_purge": true,
    "purge_horizon_days": 0,
    "prompt_user_confirmation": false
  },
  "telemetry": {
    "inference_time_ms": 8.42,
    "language_detected": "ar",
    "dialect_code": "ar-EG"
  }
}

```

---

## 4. Lifecycle Management & Lifecycle Rules

BAI v1 specifies exact lifecycle horizons for automated deletion and user prompting to ensure optimal inbox hygiene without intrusive notification behavior.

### 4.1 OTP Expiration & Purge Matrix

* **Immediate Purge (`BAIT`):** Upon detecting an OTP pattern, the validity window (e.g., 5 or 10 minutes) is extracted from the body context. When the validity time expires, the message is flagged for auto-purge from `BAIT` without triggering user notifications.
* **Promotional Auto-Purge (`BAIADS`):** High-confidence commercial advertisements are assigned a default retention horizon (e.g., 30 days) before background deletion.
* **Spam Isolation (`BAIS`):** Messages routed to `BAIS` are isolated immediately and scheduled for automated batch purging after a 7-day safety buffer.

### 4.2 Non-Intrusive Prompting Logic

* **High Confidence ($\ge 0.85$):** The classification action is executed automatically. `prompt_user_confirmation` is set to `false`.
* **Low Confidence ($< 0.85$):** The message is routed to its most probable folder, but `prompt_user_confirmation` is set to `true`, instructing the UI to display a subtle, non-blocking categorization query.

---

## 5. Security, Ethics, and Compliance (AM Standard)

BAI v1 operates in full alignment with the ethical and privacy imperatives of the **AM Standard**:

1. **Zero User Data Training:** Model training datasets are generated exclusively through synthetic pipelines and open evaluation sources. Under no circumstances are live user emails or private logs used for model training or refinement.


2. **Jailbreak and Injection Resistance:** The classification engine ignores prompt injection patterns embedded within email bodies designed to manipulate category assignments or trigger system commands.


3. **Islamic & Moral Standards:** Built-in classifier boundaries automatically isolate and flag malicious, pornographic, scam, or harmful email content into `BAIS` to protect the end-user experience.
