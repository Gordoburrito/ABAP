### Steele pipeline: dictionary/pattern-based matching

**What it is**: Steele uses a pattern dictionary keyed by `Year_Make_Model` to assign vehicle compatibility (car_ids/tags) at scale. It first tries exact matches against the golden dataset, then falls back to the dictionary. For large runs that need AI, it deduplicates work to one AI task per unique pattern and applies results to all matching rows.

### Inputs
- **Steele processed CSV**: Rows with `Year`, `Make`, `Model`, `StockCode`, etc.
- **Golden dataset**: `shared/data/master_ultimate_golden.csv` with `year`, `make`, `model`, `car_id`.
- **Pattern mapping JSON**: `Steele/data/pattern_car_id_mapping.json` mapping `Year_Make_Model` → `[car_id, ...]` or tags.

### Core flows
- **Two-stage matching (no AI)**
  1) Exact join to golden on `(Year, Make, Model)` → sets `car_ids`, `match_type = exact`.
  2) Fallback to pattern dictionary: build `pattern_key = Year_Make_Model`, lookup in JSON → sets `car_ids`, `match_type = pattern`.
  - Vectorized implementation for speed.

- **Pattern-tagging flow (tags)**
  - Build `pattern_key` per row → lookup `tags` list from the JSON → join to string → consolidate by `StockCode`.

- **SKU consolidation and Shopify output**
  - Group rows by `StockCode` and union tags/car_ids → one product per SKU.
  - Generate all 65 Shopify columns using `product_import-column-requirements.py`.

- **Optional AI (per-pattern, not per-row)**
  - Group by unique `(Year, Make, Model)` patterns → validate against golden.
  - Only patterns that need disambiguation queue a single AI task each.
  - Apply AI result back to all rows that share that pattern.

### Key files
- `Steele/enhanced_pattern_processor.py`
  - Stage 1: exact golden matching
  - Stage 2: dictionary fallback (`pattern_car_id_mapping.json`)
  - Consolidation → Shopify format (65 columns)

- `Steele/pattern_processor.py`
  - Pure dictionary-based tag assignment using `pattern_key` lookups
  - Consolidation by `StockCode`, Shopify-ready output

- `Steele/utils/optimized_batch_steele_transformer.py`
  - Builds per-pattern mapping and queues at most one AI task per unique pattern
  - Applies AI/golden results to all rows sharing that pattern

### Minimal code landmarks
- Create and map pattern keys
  - `df['pattern_key'] = df.apply(create_year_make_model_key, axis=1)`
  - `df['car_ids'] = df['pattern_key'].map(pattern_mapping)` (fallback)

- Exact matching (vectorized)
  - Build `lookup_key` on both input and golden, then `map` to retrieve `car_id` lists.

### Typical commands
- Dictionary + golden two-stage flow (recommended):
  - `python Steele/enhanced_pattern_processor.py data/processed/steele_processed_complete.csv --mode vectorized`

- Simple dictionary tagging (tags only):
  - `python Steele/pattern_processor.py`

- Ultra-optimized pattern + batch AI flow:
  - Use `OptimizedBatchSteeleTransformer.process_ultra_optimized_pipeline()` inside `Steele/utils/optimized_batch_steele_transformer.py`.

### Outputs
- Consolidated, Shopify-ready CSV with all 65 columns, where `Tags`/`car_ids` reflect exact golden matches or dictionary pattern matches.


