### Steele Runbook

This is the one-stop guide to run Steele on a small sample and on the full dataset, with and without AI, plus test commands.

### Quick start
- No AI (recommended):
```bash
python /Users/gordonlewis/ABAP/Steele/enhanced_pattern_processor.py data/samples/steele_test_1000.csv --mode vectorized
python /Users/gordonlewis/ABAP/Steele/enhanced_pattern_processor.py data/processed/steele_processed_complete.csv --mode vectorized
```
- Outputs: `Steele/data/results/enhanced_pattern_*.csv`

- Ultra-optimized AI batch:
```bash
export OPENAI_API_KEY="..."
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --submit-only
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --status <batch_id>
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --retrieve <batch_id>
```

- Classic batch:
```bash
python /Users/gordonlewis/ABAP/Steele/main.py --submit-only
python /Users/gordonlewis/ABAP/Steele/main.py --status <batch_id>
python /Users/gordonlewis/ABAP/Steele/main.py --retrieve <batch_id>
```

### 0) Prerequisites
- **Python**: 3.10+ recommended
- **Install deps** (from repo root):

```bash
cd /Users/gordonlewis/ABAP
python3 -m venv .venv && source .venv/bin/activate
pip install -r project/requirements.txt
# Optional: if missing, install extras used here
pip install pandas pytest openai python-dotenv pydantic pytest-cov
```

- **Golden dataset required**: `shared/data/master_ultimate_golden.csv`
  - Already in the repo. If missing, copy it to the path above.

### 1) Data locations you’ll use
- Small sample: `Steele/data/samples/steele_test_1000.csv` (or `steele_sample.csv`)
- Full dataset: `Steele/data/processed/steele_processed_complete.csv`
- Pattern map (optional, speeds up pattern-only): `Steele/data/pattern_car_id_mapping.json`

### 2) Fastest recommended flow (no AI): Enhanced Pattern Processor
Two-stage matching: exact matches from the golden dataset, then pattern mapping fallback. Vectorized and very fast.

- Run on a small sample (quick check):
```bash
python /Users/gordonlewis/ABAP/Steele/enhanced_pattern_processor.py \
  data/samples/steele_test_1000.csv --mode vectorized
```

- Run on the full dataset:
```bash
python /Users/gordonlewis/ABAP/Steele/enhanced_pattern_processor.py \
  data/processed/steele_processed_complete.csv --mode vectorized
```

Outputs go to: `Steele/data/results/enhanced_pattern_<input>_<timestamp>.csv`

Tip: You can also launch via the wrapper with banners and tips:
```bash
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --enhanced-pattern data/samples/steele_test_1000.csv
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --enhanced-pattern data/processed/steele_processed_complete.csv
```

### 3) Pattern-only tagging (no AI, uses prebuilt mapping)
Uses `Steele/data/pattern_car_id_mapping.json` only. Fast, but won’t add exact golden-only matches.

```bash
python /Users/gordonlewis/ABAP/Steele/pattern_processor.py
# Or specify input file via the optimized wrapper
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --pattern data/processed/steele_processed_complete.csv
```

Validate the latest pattern output at a glance:
```bash
python /Users/gordonlewis/ABAP/Steele/validate_pattern_results.py
```

### 4) Ultra-optimized + AI (Batch API, deduped by pattern) — optional
If you want AI disambiguation for patterns without exact matches. Requires `OPENAI_API_KEY`.

- Env setup:
```bash
export OPENAI_API_KEY="your-api-key"
```

- Submit optimized batch only (return a batch ID to use later):
```bash
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --submit-only
```

- Check status / retrieve when complete:
```bash
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --status <batch_id>
python /Users/gordonlewis/ABAP/Steele/main_optimized.py --retrieve <batch_id>
```

- One-shot complete run (blocks until results and finishes output):
```bash
python /Users/gordonlewis/ABAP/Steele/main_optimized.py
```

Notes
- Batch files and results are managed in `Steele/data/batch/`.
- Final outputs saved under `Steele/data/transformed/` or `Steele/data/results/` depending on the entry point.

### 5) Classic Batch API flow — optional
Alternative entry points that use the batch queue without pattern deduplication.

- Blocking full run (submits and waits):
```bash
python /Users/gordonlewis/ABAP/Steele/main.py
```

- Submit-only / later retrieve:
```bash
python /Users/gordonlewis/ABAP/Steele/main.py --submit-only
python /Users/gordonlewis/ABAP/Steele/main.py --status <batch_id>
python /Users/gordonlewis/ABAP/Steele/main.py --retrieve <batch_id>
```

- Async CLI (separate commands):
```bash
python /Users/gordonlewis/ABAP/Steele/main_async.py submit
python /Users/gordonlewis/ABAP/Steele/main_async.py status <batch_id>
python /Users/gordonlewis/ABAP/Steele/main_async.py retrieve <batch_id>
```

### 6) Run tests (TDD workflow)
From repo root or `Steele/`:

```bash
cd /Users/gordonlewis/ABAP/Steele
python run_tests.py
```

Or individually with pytest:
```bash
pytest tests/test_steele_data_loader.py -v
pytest tests/test_steele_golden_integration.py -v
pytest tests/test_steele_transformer.py -v
pytest tests/test_product_import_validation.py -v
# AI tests (only if OPENAI_API_KEY is set)
pytest tests/test_steele_ai_integration.py -v -m ai
```

### 7) What to expect (outputs)
- Enhanced pattern processor: `Steele/data/results/enhanced_pattern_*.csv`
- Pattern-only: `Steele/data/results/pattern_tagged_shopify_*.csv`
- Ultra-optimized AI: `Steele/data/transformed/steele_ultra_optimized*.csv`
- Classic batch AI: `Steele/data/transformed/steele_batch_*.csv`

### 9) Troubleshooting
- If you see a warning about Shopify columns, the code will fall back to a minimal set and still produce output.
- Ensure golden file exists at `shared/data/master_ultimate_golden.csv`.
- For AI flows, ensure `OPENAI_API_KEY` is exported and your network allows API access.


