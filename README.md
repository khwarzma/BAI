# BAI (Bareeed Artificial Intelligence) — Core Engine v1.0

> **Specification Standard:** AM Standard (SAM Category)
> **Initiator & Owner:** Khwarzma
> **Target Application:** Bareeed Email Infrastructure
> **Repository Type:** Open-Source Evaluation & Source-Available System
> **Licensing:** Dual-Licensed under BSL 1.1 (Business Source License) & Khwarzma Source-Available Terms

---

## 📌 Executive Overview & Philosophy

Modern email delivery architectures face an unsustainable burden: handling high volumes of transactional, promotional, and automated messages using resource-heavy large language models (LLMs). This operational strategy results in server CPU saturation, excessive memory consumption, high latency timeouts, and environmental waste.

**BAI (Bareeed Artificial Intelligence) v1** is a micro-task triage and classification engine built specifically for the **Bareeed** email platform by **Khwarzma**. Engineered strictly under the **AM Standard (SAM - Small Arabic Model category)**, BAI shifts the architectural focus from raw weight scaling to high-efficiency, task-bound intelligence.

BAI v1 processes incoming mail (بريد وارد) in real-time, instantly routing messages into distinct categories (Pinned Inbox, BAIT, BAIS, BAIADS, or standard Inbox) while identifying OTP parameters and executing automated lifecycle policies—all within strict execution constraints on standard two-core CPU server environments.

---

## 🛠 Core Functional Objectives (v1 Scope)

BAI v1 operates exclusively as an in-memory, zero-disk-footprint triage engine designed to execute the following single micro-task responsibilities:

1. **Transactional OTP Identification:** Detection of multi-lingual OTP patterns, validation window extraction, and scheduling immediate post-expiration auto-purge routines.
2. **BAIT Routing (Bareeed Artificial Intelligence Trash):** Dynamic isolation of obsolete transactional data, expired notifications, and candidate trash items.
3. **BAIS Routing (Bareeed Artificial Intelligence Spam):** Instant isolation of low-value, malicious, or unsolicited email patterns.
4. **BAIADS Routing (Bareeed Artificial Intelligence Ads):** Identification of commercial campaigns, promotional offers, and newsletters with non-intrusive lifecycle management.
5. **Inbox Prioritization (PIN & Keep):** Contextual recognition of high-priority emails requiring top-of-inbox pinning vs. standard passive inbox placement.
6. **Non-Intrusive Prompting:** Automated resolution for high-confidence predictions, deferring to subtle user confirmation only during boundary confidence thresholds.

---

## 🏗 Repository & Deployment Architecture

This project is architected into two completely decoupled layers to balance community transparency with lean, secure production deployment.

### 1. Source & Open-Source Repository (Public / Dev)

The public codebase maintained for source evaluation, security audits, benchmarking, and community review.

```text
BAI/
├── README.md                     <-- Core Orientation Guide (You are here)
├── LICENSE                       <-- BSL 1.1 & Khwarzma Proprietary Terms
├── docs/                         <-- Engineering Specifications & Manuals
│   ├── SPECIFICATION.md          <-- Mathematical & Architectural Constraints
│   ├── ARCHITECTURE.md           <-- Transformer Design & C++23 Runtime Specification
│   ├── DATASETS_AND_DIALECTS.md   <-- Multilingual & 29 Dialects Data Pipeline
│   └── DEPLOYMENT.md             <-- Production Isolated Runtime Setup Guide
├── config/                       <-- Dynamic Language & Engine Rules
│   ├── languages.json            <-- Dialect Maps & Pre-processing Normalization Rules
│   └── rules.json                <-- System Rules for Purge Horizons & OTP Expirations
├── data/                         <-- Synthetic Generation & Verification Datasets
│   ├── raw/                      <-- Raw Generation Traces (ar/, en/)
│   ├── processed/                <-- Validated JSONL Datasets (train/val)
│   └── dialects_mapping/         <-- 19 Arabic & 10 English Linguistic Mappings
├── training/                     <-- PyTorch Model Construction & Export Pipeline
│   ├── generate_data.py          <-- Synthetic Pipeline via External Providers
│   ├── validate_data.py          <-- Dataset Integrity & Quality Verification
│   ├── model.py                  <-- PyTorch Micro-Transformer Encoder Class
│   ├── train.py                  <-- Execution Script for Fine-Tuning Environments
│   └── export.py                 <-- Quantization & ONNX/GGUF (.bai) Serializer
├── core/                         <-- C++23 High-Performance Inference Source
│   ├── CMakeLists.txt            <-- Cross-Platform Build Definition
│   ├── include/                  <-- High-Speed Tokenizer & ONNX Engine Headers
│   ├── src/                      <-- Zero-Allocation Core Implementation
│   └── bindings/                 <-- Pybind11 Python Native Wrappers
└── tests/                        <-- Dialect Integrity & Red-Teaming Tests
    ├── test_dialects.py          <-- Evaluation Across All 29 Supported Dialects
    └── test_performance.py       <-- Thread Safety & Latency Benchmark Tests

```

### 2. Server Production Footprint (Isolated Runtime)

The lightweight production deployment footprint operating strictly in our servers.

```text
/our/server/root
├── config/
│   ├── languages.json            <-- Runtime Language Configuration
│   └── rules.json                <-- Runtime Expiration Policies
├── core/
│   └── bai_engine.so             <-- Compiled C++23 Shared Library Object
├── bridge/
│   └── bai_pybind.so             <-- Native Compiled Python Extension Module
└── models/
    └── v1.bai                    <-- Quantized Micro-Weights Package

```

---

## 🌐 Multilingual & Dialect Scope

BAI v1 incorporates a language-agnostic tokenizer and semantic encoder capable of processing code-switching, script mixing (Arabic/Latin), and regional dialect variations without modifications to the native C++ core:

* **Arabic Dialect Matrix (19 Variants):** Comprehensive coverage including Egyptian, Saudi, Levantine, Gulf, Maghrebi, Sudanese, and regional sub-variants.
* **English Dialect Matrix (10 Variants):** Full recognition across American, British, Canadian, Australian, International, and regional syntax styles.
* **Normalization Engine:** Integrated pre-processing pipeline standardizing letterforms, removing decorative tatweel, and isolating numeric sequences required for OTP parsing.

---

## 🔒 Security, Isolation & Performance Boundaries

To adhere to the **AM Standard** and ensure safe operation alongside Django and co-located web applications:

* **Hardware Limits:** Operational target capped at $\le 150 \text{ MB}$ RAM footprint and $0\%$ dynamic disk writes during inference.
* **Execution Latency:** Total end-to-end classification sub-15 milliseconds via native C++ shared library execution.
* **User Isolation:** Execution restricted to system user `ai` and group `ais` with zero elevate privileges or raw database access.
* **Hot-Swapping:** Model weight updates (`v1.bai` to `v1.1.bai`) execute atomically via memory-mapped pointer swapping (`mmap`) without reloading Gunicorn or restarting application processes.

---

## 📜 Intellectual Property & Licensing

Copyright (c) 2026 **Khwarzma**. All rights reserved.

This repository is published for **public inspection, security auditing, and architectural evaluation**. The source code is dual-licensed under:

1. **Business Source License 1.1 (BSL 1.1):** Grants public access for code review, testing, and educational analysis. Commercial use, public hosting, or deployment within competing email systems is strictly prohibited.
2. **Khwarzma Source-Available Commercial License:** Grants explicit operational rights exclusively to Khwarzma and authorized Bareeed production instances.

For licensing inquiries or enterprise permissions, contact the system architects at **Khwarzma**.
