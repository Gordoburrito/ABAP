# Steele Data Processing - Usage Guide

This guide explains how to use the `run_everything.py` master script to process Steele data with different options.

## Prerequisites

1. **Set OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Ensure data files exist:**
   - `data/processed/steele_processed_complete.csv` (main processed data)
   - `data/results/first_100_products_*.csv` (existing results, optional)

## Usage Options

### 1. Quick Test (Recommended First Step)
Test with 3-5 products to verify everything is working:
```bash
python run_everything.py test --size 3
```

### 2. Process Unknown SKUs
Process all products with `0_Unknown_UNKNOWN` tags:
```bash
python run_everything.py unknown
```

Limit to first 50 unknown products:
```bash
python run_everything.py unknown --limit 50
```

### 3. Process Specific Batch Size
Process exactly 100 products:
```bash
python run_everything.py batch --size 100
```

Process exactly 500 products:
```bash
python run_everything.py batch --size 500
```

### 4. Run Full Production Pipeline
Process the entire dataset (WARNING: This will take hours and cost significant API credits):
```bash
python run_everything.py full
```

## Examples

### Quick Test Run
```bash
# Test with 3 products to verify system is working
python run_everything.py test --size 3
```

### Process First 100 Unknown Products
```bash
# Process first 100 products with unknown tags
python run_everything.py batch --size 100
```

### Process All Unknown SKUs (Limited)
```bash
# Process all unknown SKUs but limit to first 200
python run_everything.py unknown --limit 200
```

## Output Locations

All results are saved to `data/results/` with timestamps:

- **Test results:** `test_sample_N_products_YYYYMMDD_HHMMSS.csv`
- **Batch results:** `unknown_skus_processed_YYYYMMDD_HHMMSS.csv`
- **Full pipeline:** `steele_shopify_complete.csv`

## What Each Mode Does

### Test Mode
- Uses predefined test products with known issues
- Processes 3-5 products quickly
- Shows detailed results for verification
- Perfect for testing after changes

### Unknown Mode
- Finds products with `0_Unknown_UNKNOWN` tags
- Uses two-pass AI extraction to generate proper tags
- Can process all unknown products or limit to specific number

### Batch Mode
- Processes exactly the specified number of products
- Takes from unknown SKUs or processed data
- Good for controlled testing of larger batches

### Full Mode
- Runs complete production pipeline
- Processes entire dataset
- Includes consolidation and Shopify formatting
- **WARNING:** Expensive and time-consuming

## Expected Results

### Before Processing
```
Tags: 0_Unknown_UNKNOWN
```

### After Processing
```
Tags: 1920_Ford_Model T, 1921_Ford_Model T, 1922_Chevrolet_490, ...
```

## Troubleshooting

### API Key Issues
```
❌ Error: OPENAI_API_KEY environment variable not set
```
**Solution:** Set your OpenAI API key as shown in prerequisites.

### No Data Files
```
❌ Error: Main data file not found
```
**Solution:** Ensure `data/processed/steele_processed_complete.csv` exists.

### AI Extraction Errors
- Check your API key has sufficient credits
- Try with smaller batch sizes first
- Review the error messages in the output

## Cost Estimation

- **Test mode (3-5 products):** ~$0.01-0.02
- **Batch 100 products:** ~$0.20-0.50
- **Batch 500 products:** ~$1.00-2.50
- **Full pipeline:** ~$50-200+ (depending on dataset size)

## Monitoring Progress

The script provides real-time progress updates:
```
🔄 Processing 100 products...
⏱️  Processing completed in 0:02:45
📈 PROCESSING SUMMARY:
   Total products processed: 100
   Successful extractions: 98
   Success rate: 98.0%
   Products with improved tags: 85
   Improvement rate: 85.0%
```

## Next Steps After Processing

1. **Review Results:** Check the output CSV file
2. **Validate Tags:** Spot-check a few products manually
3. **Scale Up:** If results look good, process larger batches
4. **Production:** Use full pipeline for complete dataset 