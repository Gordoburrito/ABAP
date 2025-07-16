# Steele Two-Pass AI Debugging Framework

## Overview
This debugging framework allows you to test each stage of the two-pass AI approach independently. Each script focuses on a specific part of the pipeline and provides detailed debugging output.

## Quick Start

### Run Complete Pipeline
```bash
python debug_pipeline.py
```

### Run Individual Stages
```bash
python debug_pass1.py           # Stage 1: Initial AI extraction
python debug_pass2.py           # Stage 2: Golden master refinement  
python debug_golden_master.py   # Stage 3: Golden master lookup
python debug_tags.py            # Stage 4: Tag generation
```

## Prerequisites

1. **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
2. **Test Data**: Ensure `data/results/debug_test_input.csv` exists
3. **Golden Master**: File `../shared/data/master_ultimate_golden.csv` must exist
4. **Utils Modules**: `utils/ai_extraction.py` and `utils/golden_master_tag_generator.py`

## Stage Details

### Stage 1: Pass 1 AI Extraction (`debug_pass1.py`)
- **Purpose**: Test initial AI extraction from product descriptions
- **Input**: `data/results/debug_test_input.csv`
- **Output**: `data/results/debug_pass1_results.csv`
- **What it shows**: 
  - Year ranges extracted by AI
  - Makes and models identified
  - Generated titles and descriptions
  - AI processing errors

### Stage 2: Pass 2 Refinement (`debug_pass2.py`)
- **Purpose**: Test golden master refinement process
- **Input**: Pass 1 results + original product data
- **Output**: `data/results/debug_pass2_results.csv`
- **What it shows**:
  - How Pass 1 data gets refined
  - Which makes are found in golden master
  - Changes from Pass 1 to Pass 2
  - Validation against golden master

### Stage 3: Golden Master Lookup (`debug_golden_master.py`)
- **Purpose**: Test golden master data access and validation
- **Input**: Pass 2 results + golden master data
- **Output**: `data/results/debug_golden_master_results.csv`
- **What it shows**:
  - Golden master data structure
  - Make/model lookups
  - Tag generation success/failure
  - Debugging why tags aren't generated

### Stage 4: Tag Generation (`debug_tags.py`)
- **Purpose**: Test final vehicle-specific tag generation
- **Input**: Golden master results
- **Output**: `data/results/debug_tags_results.csv`
- **What it shows**:
  - Test scenarios for different vehicle types
  - Tag format examples (YEAR_MAKE_MODEL)
  - Success rates by make/model
  - Tag generation statistics

## Output Files

All debug outputs are saved to `data/results/`:

| File | Description |
|------|-------------|
| `debug_pass1_results.csv` | Pass 1 AI extraction results |
| `debug_pass2_results.csv` | Pass 2 refinement results |
| `debug_golden_master_results.csv` | Golden master lookup results |
| `debug_tags_results.csv` | Tag generation test results |
| `debug_pipeline_summary.csv` | Complete pipeline execution summary |

## Understanding the Results

### Successful Tag Generation
Look for products with:
- `Tags_Count > 0` in golden master results
- `Success = True` in tag results  
- Tags in format: `1936_Cord_810, 1937_Cord_810`

### Common Issues

1. **No Tags Generated**
   - Make not found in golden master (AC, etc.)
   - Invalid year ranges
   - Universal products (no specific vehicle)

2. **AI Extraction Errors**
   - Missing OPENAI_API_KEY
   - Rate limits or API issues
   - Malformed product descriptions

3. **Golden Master Issues**
   - File not found
   - Column name mismatches
   - Data format problems

## Debugging Tips

1. **Start with Pass 1**: If Pass 1 fails, later stages won't work
2. **Check Makes**: Common makes like Ford, Chevrolet should generate tags
3. **Verify Golden Master**: Ensure `../shared/data/master_ultimate_golden.csv` exists
4. **API Key**: Set `export OPENAI_API_KEY=your_key_here`

## Expected Results

With valid test data, you should see:
- **Universal products**: No tags (expected)
- **Cord products**: Tags like `1936_Cord_810`
- **Hupmobile products**: Tags like `1940_Hupmobile_Skylark, 1941_Hupmobile_Skylark`
- **Invalid makes (AC)**: No tags (expected)

## Next Steps

After debugging, use the insights to:
1. Fix issues in the main pipeline
2. Improve AI prompts for better extraction
3. Enhance golden master validation
4. Optimize tag generation logic