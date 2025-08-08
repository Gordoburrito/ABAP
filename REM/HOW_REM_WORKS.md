### REM pipeline: per-row lookups with golden dataset + AI refinement

**What it is**: REM processes each vendor row individually. For every row, it calls an AI-assisted extractor constrained by the golden dataset to produce a standardized `ProductData`. A second pass refines models to car_ids per row using the golden dataset (and optional AI), then outputs formatted data for downstream steps (e.g., ABAP formatter).

### Inputs
- **REM CSV**: Vendor file with SKU, description, and pricing columns (renamed/cleaned on load).
- **Golden dataset**: `shared/data/master_ultimate_golden.csv` with `year`, `make`, `model`, `car_id`.

### Core flow
1) Load vendor CSV and normalize columns.
2) Load golden dataset for valid options (`year/make/model`) and `car_id`s.
3) Per-row AI extraction (`extract_product_data_with_ai`):
   - Prompt includes valid makes/models/years and product types from golden.
   - Produces `ProductData` fields: title, year_min/max, make, model(s), mpn, price, collection, product_type, SEO.
4) Second pass per-row model expansion (`refine_models_with_ai`):
   - If model is `ALL` (or body-style variants), expand across the golden dataset for the row's year range and make(s).
   - Otherwise, use AI to refine models against allowed golden models, then map to `car_id`s.
5) Output a DataFrame with standardized columns; downstream formatter converts to ABAP when needed.

### Key files
- `project/src/transformers/transform_REM.py`
  - Reads REM CSV, cleans/renames columns, loads golden, calls common transformer.

- `project/src/transform.py`
  - `extract_product_data_with_ai(row, golden_df, client)` → per-row AI extraction constrained by golden.
  - `refine_models_with_ai(df, golden_df, client)` → per-row model expansion to `car_id`s.
  - `transform_data_with_ai(golden_df, vendor_df)` → orchestrates both passes across rows.

- `project/src/formatters/format_REM_to_ABAP.py`
  - Loads transformed REM data and converts to ABAP format for import.

### Minimal code landmarks
- Per-row extraction
  - Iterate rows → `extract_product_data_with_ai(vendor_row, golden_df, client)`

- Valid option constraints from golden
  - `valid_makes = sorted(golden_df['make'].dropna().unique())`
  - `valid_models = sorted(golden_df['model'].dropna().unique())`

- Per-row model refinement
  - Expand `ALL`/body-style to model lists using golden dataset for the row’s year range and make(s)
  - Else, AI refines comma-separated models → map back to golden `car_id`s

### Typical commands
- Transform REM CSV to standardized intermediate:
  - `python project/src/transformers/transform_REM.py` (updates `data/transformed/REM_transformed.csv`)

- Convert to ABAP format:
  - `python project/src/formatters/format_REM_to_ABAP.py`

### Outputs
- Standardized DataFrame with per-row fields and `models_expanded` mapped to `car_id`s, ready for formatting to ABAP import.


