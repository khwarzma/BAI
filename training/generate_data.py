#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest" / "generation_manifest.json"
COMPANY_PATH = ROOT / "data" / "manifest" / "company_patterns.json"
AR_DIALECTS_PATH = ROOT / "data" / "dialects_mapping" / "ar_dialects.json"
EN_DIALECTS_PATH = ROOT / "data" / "dialects_mapping" / "en_dialects.json"
CHECKPOINT_DIR = ROOT / "data" / "checkpoint"
RAW_DIR = ROOT / "data" / "raw"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def sha256_hex(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in (CHECKPOINT_DIR, RAW_DIR / "ar", RAW_DIR / "en", RAW_DIR / "global"):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class HashRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
        )
        self._conn.commit()
        self._lock = threading.RLock()

    def exists(self, value: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM seen_hashes WHERE hash = ?",
                (value,),
            ).fetchone()
            return row is not None

    def add(self, value: str) -> bool:
        with self._lock:
            if self.exists(value):
                return False
            self._conn.execute(
                "INSERT INTO seen_hashes (hash, created_at) VALUES (?, datetime('now'))",
                (value,),
            )
            self._conn.commit()
            return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def load_progress_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": "1.0.0", "dialect_counts": {}, "global_counts": {}, "updated_at": None}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"version": "1.0.0", "dialect_counts": {}, "global_counts": {}, "updated_at": None}
    data.setdefault("dialect_counts", {})
    data.setdefault("global_counts", {})
    data.setdefault("updated_at", None)
    return data


def save_progress_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def get_all_dialects(ar_cfg: Dict[str, Any], en_cfg: Dict[str, Any]) -> List[str]:
    dialects = list(ar_cfg.get("dialects", {}).keys()) + list(en_cfg.get("dialects", {}).keys())
    return sorted(set(dialects))


def get_category_labels(manifest: Dict[str, Any]) -> List[str]:
    return list(manifest.get("categories", {}).keys())


def pick_company(company_patterns: Dict[str, Any]) -> Dict[str, Any]:
    companies = company_patterns.get("companies", [])
    if not companies:
        return {
            "company_name": "ExampleCorp",
            "sector": "General Business",
            "dialects_supported": "all",
            "otp_templates": [
                "Your verification code is [CODE]. Valid for [TIME].",
                "Use code [CODE] to confirm your login.",
            ],
            "header_indicators": ["Verification required"],
            "confidence_multiplier": 1.5,
        }
    return random.choice(companies)


def choose_otp_code(category: str) -> Tuple[str, str, str]:
    if category != "BAIT":
        return "", "", ""
    patterns = [
        ("digit4", f"{random.randint(0, 9999):04d}", "4"),
        ("digit6", f"{random.randint(0, 999999):06d}", "6"),
        ("digit8", f"{random.randint(0, 99999999):08d}", "8"),
        ("alpha", f"G-{random.randint(1000, 9999)}", "G-1234"),
        ("formatted", f"{random.randint(100, 999)}-{random.randint(100, 999)}", "123-456"),
    ]
    _, code, _ = random.choice(patterns)
    expiry_minutes = str(random.choice([3, 5, 8, 10, 15]))
    return code, expiry_minutes, "otp"


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.8, "top_p": 0.9},
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/generate"
        request_obj = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        last_error: Optional[Exception] = None
        for attempt in range(1, 6):
            try:
                with request.urlopen(request_obj, timeout=120) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return str(parsed.get("response", ""))
            except (error.URLError, error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama request failed after retries: {last_error}")


def fallback_email_text(
    category: str,
    company: Dict[str, Any],
    scenario: Dict[str, Any],
    language: str,
    otp_code: Optional[str],
    expiry_minutes: Optional[str],
) -> str:
    greeting = "السلام عليكم" if language == "ar" else "Hello"
    closing = "مع الاحترام" if language == "ar" else "Best regards"
    company_name = company.get("company_name", "ExampleCorp")
    subject = scenario.get("subject_template", "Notification")
    body = scenario.get("body_template", "Please review this message.")

    if category == "BAIT":
        if language == "ar":
            text = (
                f"{greeting},\n\n"
                f"{company_name} يتطلب تأكيد هوية الحساب الخاص بك.\n\n"
                f"رمز التحقق الخاص بك هو {otp_code}.\n"
                f"هذا الرمز صالح لمدة {expiry_minutes} دقيقة فقط.\n\n"
                f"يرجى إدخاله في الموقع الآمن قبل انتهاء الوقت.\n"
                f"إذا لم تكن أنت من قام بهذا الطلب، يمكنك تحديث الحساب عبر الرابط الآمن.\n\n"
                f"{closing},\n"
                f"فريق {company_name}"
            )
        else:
            text = (
                f"{greeting},\n\n"
                f"{company_name} requires identity verification for your account.\n\n"
                f"Your verification code is {otp_code}.\n"
                f"This code is valid for {expiry_minutes} minutes only.\n\n"
                f"Please enter it on the secure page before the time expires.\n"
                f"If you did not initiate this request, update your account immediately.\n\n"
                f"{closing},\n"
                f"{company_name} Security Team"
            )
        return f"Subject: {subject}\n\n{text}"

    if category == "BAIS":
        if language == "ar":
            text = (
                f"{greeting},\n\n"
                f"نود تنبيهك إلى وجود طلب عاجل يتطلب مراجعة فورية.\n"
                f"يرجى التحقق من تفاصيل المبلغ أو التحديث المالي قبل إغلاق المهمة.\n\n"
                f"{body}\n\n"
                f"{closing},\n"
                f"إدارة {company_name}"
            )
        else:
            text = (
                f"{greeting},\n\n"
                f"We need your immediate attention regarding an urgent account action.\n"
                f"Please review the financial or security details before the deadline expires.\n\n"
                f"{body}\n\n"
                f"{closing},\n"
                f"{company_name} Operations"
            )
        return f"Subject: {subject}\n\n{text}"

    if language == "ar":
        text = (
            f"{greeting},\n\n"
            f"{body}\n\n"
            f"نود متابعة الطلب معك في أقرب وقت ممكن.\n"
            f"{closing},\n"
            f"{company_name}"
        )
    else:
        text = (
            f"{greeting},\n\n"
            f"{body}\n\n"
            f"We appreciate your prompt attention and look forward to the next update.\n"
            f"{closing},\n"
            f"{company_name}"
        )
    return f"Subject: {subject}\n\n{text}"


def render_prompt(
    category: str,
    dialect: str,
    language: str,
    company: Dict[str, Any],
    scenario: Dict[str, Any],
    system_prompt: str,
) -> str:
    company_name = company.get("company_name", "ExampleCorp")
    template = (
        random.choice(company.get("otp_templates", ["Use [CODE] to verify your account."]))
        if category == "BAIT"
        else random.choice(company.get("header_indicators", ["Action required"]))
    )
    return (
        f"Generate an email in {language.upper()} for dialect {dialect}.\n"
        f"Category: {category}\n"
        f"Company: {company_name}\n"
        f"Scenario: {scenario.get('subdomain', 'general')}\n"
        f"Subject template: {scenario.get('subject_template', 'Notification')}\n"
        f"Body emphasis: {scenario.get('body_template', 'Please review this message.')}\n"
        f"Use a realistic corporate email structure and natural language.\n"
        f"Output strictly as JSON with fields: text, category_label, otp_label, language, dialect, confidence_multiplier, company_name.\n"
        f"If category is BAIT, include a valid OTP pattern and set otp_label to 1.0 and confidence_multiplier to 1.5.\n"
        f"Example template: {template}\n\n"
        f"System instructions: {system_prompt}"
    )


def generate_record(
    category: str,
    dialect: str,
    language: str,
    scenario: Dict[str, Any],
    company: Dict[str, Any],
    ollama_client: Optional[OllamaClient],
    system_prompt: str,
) -> Dict[str, Any]:
    otp_code = None
    expiry_minutes = None
    if category == "BAIT":
        otp_code, expiry_minutes, _ = choose_otp_code(category)

    if ollama_client is not None:
        prompt = render_prompt(category, dialect, language, company, scenario, system_prompt)
        try:
            raw_response = ollama_client.generate(prompt, system_prompt)
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                text = normalize_text(str(parsed.get("text", "")))
                if text:
                    record = {
                        "text": text,
                        "category_label": str(parsed.get("category_label", category)),
                        "otp_label": float(parsed.get("otp_label", 1.0 if category == "BAIT" else 0.0)),
                        "language": str(parsed.get("language", language)),
                        "dialect": str(parsed.get("dialect", dialect)),
                        "confidence_multiplier": float(parsed.get("confidence_multiplier", 1.5 if category == "BAIT" else 1.0)),
                        "company_name": str(parsed.get("company_name", company.get("company_name", "ExampleCorp"))),
                        "subject": parsed.get("subject", scenario.get("subject_template", "Notification")),
                    }
                    return record
        except Exception:
            pass

    text = fallback_email_text(
        category=category,
        company=company,
        scenario=scenario,
        language=language,
        otp_code=otp_code,
        expiry_minutes=expiry_minutes,
    )
    return {
        "text": normalize_text(text),
        "category_label": category,
        "otp_label": 1.0 if category == "BAIT" else 0.0,
        "language": language,
        "dialect": dialect,
        "confidence_multiplier": 1.5 if category == "BAIT" else 1.0,
        "company_name": company.get("company_name", "ExampleCorp"),
        "subject": scenario.get("subject_template", "Notification"),
    }


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def task_for_dialect(
    category: str,
    dialect: str,
    count: int,
    manifest: Dict[str, Any],
    company_patterns: Dict[str, Any],
    llm_client: Optional[OllamaClient],
    state: Dict[str, Any],
    hash_registry: HashRegistry,
) -> List[Dict[str, Any]]:
    # 1. التحديد الديناميكي للغة والمسار
    if dialect.startswith("ar"):
        language = "ar"
        output_path = RAW_DIR / "ar" / f"{dialect}.jsonl"
    elif dialect.startswith("en"):
        language = "en"
        output_path = RAW_DIR / "en" / f"{dialect}.jsonl"
    else:
        # للحالات الخاصة والملفات النمطية العالمية (Global)
        language = "global"
        output_path = RAW_DIR / "global" / f"{dialect}.jsonl"

    scenarios = [item for item in manifest.get("scenarios_matrix", []) if item.get("category") == category]
    if not scenarios:
        return []

    created_records: List[Dict[str, Any]] = []
    used = 0
    while used < count:
        scenario = random.choice(scenarios)
        company = pick_company(company_patterns)
        record = generate_record(category, dialect, language, scenario, company, llm_client, manifest.get("system_prompt_template", ""))
        record_hash = sha256_hex(json.dumps(record, sort_keys=True, ensure_ascii=False))
        if not hash_registry.add(record_hash):
            continue
        created_records.append(record)
        used += 1

    key = f"{dialect}:{category}"
    state.setdefault("dialect_counts", {})
    state["dialect_counts"][key] = state["dialect_counts"].get(key, 0) + len(created_records)

    # 2. الكتابة في المسار المحدد فقط (منع التكرار المزدوج)
    append_jsonl(output_path, created_records)
    
    return created_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic email samples for BAI training.")
    parser.add_argument("--samples-per-category", type=int, default=25, help="Number of samples to generate per category per dialect.")
    parser.add_argument("--max-workers", type=int, default=4, help="Thread count for parallel generation.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Local Ollama API base URL.")
    parser.add_argument("--ollama-model", type=str, default="llama3.1", help="Ollama model name for LLM generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    manifest = load_json(MANIFEST_PATH)
    company_patterns = load_json(COMPANY_PATH)
    ar_cfg = load_json(AR_DIALECTS_PATH)
    en_cfg = load_json(EN_DIALECTS_PATH)
    dialects = get_all_dialects(ar_cfg, en_cfg)
    categories = get_category_labels(manifest)

    progress_path = CHECKPOINT_DIR / "progress_state.json"
    state = load_progress_state(progress_path)
    hash_registry = HashRegistry(CHECKPOINT_DIR / "hashes_registry.db")

    ollama_client: Optional[OllamaClient] = None
    try:
        ollama_client = OllamaClient(args.ollama_url, args.ollama_model)
        ollama_client.generate("ping", "ping")
    except Exception:
        ollama_client = None

    tasks: List[Tuple[str, str]] = []
    for dialect in dialects:
        for category in categories:
            key = f"{dialect}:{category}"
            current_total = state.get("dialect_counts", {}).get(key, 0)
            if current_total >= args.samples_per_category:
                continue
            remaining = max(0, args.samples_per_category - current_total)
            if remaining > 0:
                tasks.append((dialect, category))

    if not tasks:
        save_progress_state(progress_path, state)
        hash_registry.close()
        print("No work required. Progress state already satisfies the requested totals.")
        return 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = []
        for dialect, category in tasks:
            futures.append(
                pool.submit(
                    task_for_dialect,
                    category,
                    dialect,
                    args.samples_per_category,
                    manifest,
                    company_patterns,
                    ollama_client,
                    state,
                    hash_registry,
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - surfaced to console for operator visibility
                print(f"Generation task failed: {exc}", flush=True)

    save_progress_state(progress_path, state)
    hash_registry.close()
    print(f"Completed generation for {len(tasks)} dialect/category tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
