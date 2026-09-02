import json
import re
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 620000

print("🔬 Starting Deep Structural & Pattern Audit...")

# Pool breakdown: (dialect, category) -> list of valid records
pool = defaultdict(list)
seen_hashes = set()

stats = {
    "total_read": 0,
    "corrupted_json": 0,
    "missing_fields": 0,
    "too_short": 0,
    "noise_or_repetitive": 0,
    "duplicates": 0,
    "valid_clean": 0
}

# Regex pattern for minimum quality text (must contain valid words, not just spaces/symbols)
VALID_TEXT_PATTERN = re.compile(r'[\u0600-\u06FFa-zA-Z]{3,}')

for jsonl_file in RAW_DIR.rglob('*.jsonl'):
    filename = jsonl_file.stem  # e.g., ar-EG or INBOX
    
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in f:
            stats["total_read"] += 1
            line = line.strip()
            if not line:
                continue
            
            # 1. JSON Structural Parse
            try:
                data = json.loads(line)
            except Exception:
                stats["corrupted_json"] += 1
                continue
            
            text = str(data.get('text', '')).strip()
            category = data.get('category') or data.get('label') or (filename if filename in ['INBOX', 'INBOX_PINNED', 'BAIS', 'BAIT', 'BAIADS'] else None)
            dialect = data.get('dialect') or (filename if filename not in ['INBOX', 'INBOX_PINNED', 'BAIS', 'BAIT', 'BAIADS'] else 'global')
            
            # 2. Strict Field Validation
            if not text or not category or not dialect:
                stats["missing_fields"] += 1
                continue
            
            # 3. Quality & Pattern Inspection
            if len(text) < 15:
                stats["too_short"] += 1
                continue
                
            if not VALID_TEXT_PATTERN.search(text):
                stats["noise_or_repetitive"] += 1
                continue
            
            # 4. Strict Exact & Normalized Deduplication
            norm_text = re.sub(r'\s+', ' ', text.lower().strip())
            if norm_text in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(norm_text)
            
            stats["valid_clean"] += 1
            pool[(dialect, category)].append({
                "text": text,
                "category": category,
                "dialect": dialect
            })

print("\n=================== 📊 DEEP AUDIT REPORT ===================")
print(f" Total Raw Lines Read       : {stats['total_read']:,}")
print(f" ❌ Corrupted JSON Format    : {stats['corrupted_json']:,}")
print(f" ❌ Missing Key Fields       : {stats['missing_fields']:,}")
print(f" ❌ Too Short (<15 chars)    : {stats['too_short']:,}")
print(f" ❌ Low Quality / Noise      : {stats['noise_or_repetitive']:,}")
print(f" ❌ Duplicate Records        : {stats['duplicates']:,}")
print(f" ✅ Clean & Verified Pool    : {stats['valid_clean']:,}")
print(f" 📌 Total Active Combinations: {len(pool)} (Expected max 145)")
print("============================================================\n")

# Detailed dialect breakdown report
print("--- Sample Distribution Breakdown (Top 15 Categories) ---")
sorted_combos = sorted(pool.items(), key=lambda x: len(x[1]), reverse=True)
for (d, c), recs in sorted_combos[:15]:
    print(f"   [{d} | {c}]: {len(recs):,} valid samples")

# Final Export Process
if stats["valid_clean"] > 0:
    import random
    random.seed(42)
    
    all_clean_records = []
    for recs in pool.values():
        all_clean_records.extend(recs)
    
    random.shuffle(all_clean_records)
    
    total = len(all_clean_records)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_data = all_clean_records[:train_end]
    val_data = all_clean_records[train_end:val_end]
    test_data = all_clean_records[val_end:]
    
    with (PROCESSED_DIR / 'train.jsonl').open('w', encoding='utf-8') as f:
        for r in train_data: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (PROCESSED_DIR / 'val.jsonl').open('w', encoding='utf-8') as f:
        for r in val_data: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (PROCESSED_DIR / 'test.jsonl').open('w', encoding='utf-8') as f:
        for r in test_data: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
    print(f"\n🚀 Splits generated:")
    print(f"   - Train (80%): {len(train_data):,}")
    print(f"   - Val   (10%): {len(val_data):,}")
    print(f"   - Test  (10%): {len(test_data):,}")

