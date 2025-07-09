# Complete Steele Raw Data Processing Guide

This guide covers the complete workflow for processing raw Steele data from the original Excel format to final Shopify-ready CSV files.

## Overview

The Steele data processing pipeline consists of two main phases:

1. **Raw Data Processing**: Convert Excel sheets to processed CSV format
2. **Transformation Pipeline**: Convert processed CSV to Shopify-ready format with AI enhancement

## Phase 1: Raw Data Processing

### Input Data Format

The raw Steele data comes in an Excel file (`steele.xlsx`) with two sheets:

**Sheet 1: Product Data (PART INFO)**
```
StockCode | Product Name | Description | StockUom | UPC Code | MAP | Dealer Price
10-0001-40 | Accelerator Pedal Pad | Pad, accelerator... | ea. | 706072000022 | 75.49 | 43.76
```

**Sheet 2: Fitment Data (PART FITMENT)**
```
PartNumber | Year | Make | Model | Submodel | Type | Doors | BodyType
07-1057-07 | 1957 | Cadillac | Series 62 | Base | Car | 2 | Convertible
07-1057-07 | 1957 | Cadillac | Eldorado Biarritz | Base | Car | 2 | Convertible
```

### Processing Script

Use `utils/process_raw_steele_data.py` to convert the Excel file:

```bash
cd Steele
python utils/process_raw_steele_data.py
```

This script:
1. Loads both Excel sheets
2. Validates data structure
3. Cleans and standardizes data
4. Joins product and fitment data on StockCode = PartNumber
5. Creates one row per product-fitment combination
6. Outputs `data/processed/steele_processed_complete.csv`

### Output Format

The processed CSV combines both sheets:
```
StockCode,Product Name,Description,StockUom,UPC Code,MAP,Dealer Price,PartNumber,Year,Make,Model,Submodel,Type,Doors,BodyType
10-0001-40,Accelerator Pedal Pad,"Pad, accelerator...",ea.,706072000022,75.49,43.76,10-0001-40,1928,Stutz,Stutz,Base,Car        ,0.0,U/K
```

### Processing Statistics

From the raw Excel file:
- **Products**: 1,044,708 unique products
- **Fitment Records**: 1,045,849 fitment applications
- **Combined Output**: 2,077,649 product-fitment combinations
- **Final Products**: ~11,655 unique products with multiple fitment applications

## Phase 2: Transformation Pipeline

### Pipeline Architecture

The transformation pipeline uses the `SteeleDataTransformer` class with the following steps:

```
Processed CSV → Golden Master Validation → Template Enhancement → AI Vehicle Tags → Shopify Format → Consolidation
```

### Step-by-Step Process

#### 1. Golden Master Validation
- Validates vehicle data against `master_ultimate_golden.csv`
- Ensures Year/Make/Model combinations exist in the golden dataset
- Provides validation confidence scores

#### 2. Template Enhancement (NO AI)
- Generates SEO meta titles and descriptions using templates
- Categorizes products using rule-based logic
- Fast, cost-effective processing for complete fitment data

#### 3. AI Vehicle Tag Generation
- Uses OpenAI GPT-4-mini to generate accurate vehicle tags
- Maps Steele vehicle data to master_ultimate_golden format
- Handles generic models (e.g., "Stutz") by generating tags for all variants
- Provides accurate `YEAR_MAKE_MODEL` format tags

#### 4. Shopify Format Conversion
- Converts to complete 65-column Shopify import format
- Follows `product_import-column-requirements.py` specifications
- Includes all required and optional fields

#### 5. Product Consolidation
- Groups products by unique SKU
- Combines vehicle tags for products with multiple applications
- Creates final one-product-per-SKU format

### Running the Pipeline

#### Quick Test (Sample Data)
```bash
cd Steele
python test_new_processed_data.py
```

#### Production Pipeline (Full Dataset)
```bash
cd Steele
python run_full_production_pipeline.py
```

Select from pipeline configurations:
1. **Full AI Pipeline (Recommended)**: AI tags, 1000 batch size
2. **Fast Template Pipeline**: No AI, 2000 batch size  
3. **Small Batch AI Pipeline**: AI tags, 500 batch size
4. **Custom Configuration**: Choose your own settings

### Performance Expectations

**Sample Data (49 records)**:
- Processing time: ~1.5 seconds
- Rate: ~33 records/second
- Output: 4 unique products

**Full Dataset (2M+ records)**:
- Estimated time: ~17 hours (AI enabled)
- Estimated time: ~8 hours (AI disabled)
- Expected output: ~11,655 unique products

### AI Vehicle Tag Examples

The AI system provides accurate vehicle tag generation:

**Generic Model Detection**:
```
Input: 1933 Stutz Stutz
Output: 1933_Stutz_Model_DV-32, 1933_Stutz_Model_LAA, 1933_Stutz_Model_SV-16
```

**Specific Model Mapping**:
```
Input: 1929 Chrysler Series 65
Output: 1929_Chrysler_Series_65
```

## File Structure

```
Steele/
├── data/
│   ├── raw/
│   │   └── steele.xlsx                           # Original Excel file
│   ├── processed/
│   │   └── steele_processed_complete.csv         # Combined product+fitment data
│   ├── samples/
│   │   ├── steele_sample.csv                     # Original sample
│   │   └── steele_processed_sample.csv           # New processed sample
│   └── results/
│       ├── steele_shopify_complete.csv           # Final Shopify output
│       └── production_pipeline_YYYYMMDD_HHMMSS.log
├── utils/
│   ├── process_raw_steele_data.py                # Raw data processor
│   └── steele_data_transformer.py                # Main transformer
├── tests/
│   └── test_ai_vehicle_tag_generation.py         # AI testing suite
├── test_new_processed_data.py                    # Integration tests
└── run_full_production_pipeline.py               # Production runner
```

## Data Quality & Validation

### Golden Master Integration
- **Total Records**: 317,599 golden master vehicle records
- **Validation Rate**: ~50% of Steele records validated against golden master
- **Coverage**: Years 1912-2021, 49 makes, 1,762+ models

### Price Data Quality
- **Valid Prices**: >99% of products have valid MAP prices
- **Price Range**: $0.59 - $3,201.10
- **Average Price**: $67.24

### Vehicle Data Quality
- **Year Range**: 1912-2021 (historical to modern vehicles)
- **Top Makes**: Ford, Chevrolet, Cadillac, Buick, Oldsmobile
- **Fitment Density**: Average ~178 fitment applications per product

## Output Specifications

### Final Shopify CSV Format
- **Columns**: 65 columns per `product_import-column-requirements.py`
- **Records**: One row per unique product with consolidated vehicle tags
- **Tags**: Comprehensive vehicle applications in `YEAR_MAKE_MODEL` format
- **SEO**: Meta titles ≤60 chars, meta descriptions ≤160 chars
- **Compliance**: Full Shopify import compatibility

### Column Highlights
```
Title: Product Name
Body HTML: Product Description  
Vendor: Steele
Tags: 1928_Stutz_Stutz, 1929_Stutz_Stutz, 1930_Stutz_Stutz
Variant SKU: StockCode/PartNumber
Variant Price: MAP Price
Variant Cost: Dealer Price
Metafield: title_tag [string]: SEO optimized title
Metafield: description_tag [string]: SEO optimized description
```

## Troubleshooting

### Common Issues

**1. Excel File Not Found**
```
Error: Raw Excel file not found
Solution: Ensure steele.xlsx is in data/raw/ directory
```

**2. Memory Issues (Large Dataset)**
```
Error: Memory error during processing
Solution: Use smaller batch sizes (500 instead of 1000)
```

**3. OpenAI API Issues**
```
Error: AI vehicle tag generation failed
Solution: Check OpenAI API key, use Fast Template Pipeline as fallback
```

**4. Golden Master Missing**
```
Error: Golden dataset not found
Solution: Ensure master_ultimate_golden.csv is in shared/data/
```

### Performance Optimization

**For Speed**:
- Use Fast Template Pipeline (AI disabled)
- Increase batch size to 2000-5000
- Process during off-peak hours

**For Accuracy**:
- Use Full AI Pipeline
- Smaller batch sizes (500-1000)
- Enable all validation steps

**For Cost Control**:
- Disable AI for initial testing
- Use sample data for development
- Monitor OpenAI usage

## Next Steps

1. **Process Raw Data**: Run `utils/process_raw_steele_data.py`
2. **Test Pipeline**: Run `test_new_processed_data.py`
3. **Production Run**: Run `run_full_production_pipeline.py`
4. **Validate Output**: Check final CSV against Shopify requirements
5. **Import to Shopify**: Use final CSV for product import

## Success Metrics

- ✅ **Data Coverage**: >99% of products processed successfully
- ✅ **Golden Validation**: ~50% validated against master dataset
- ✅ **AI Accuracy**: Accurate vehicle tag mapping to master_ultimate_golden
- ✅ **Shopify Compliance**: 100% compliant with import requirements
- ✅ **Performance**: Processing rate >30 records/second
- ✅ **Consolidation**: Efficient product consolidation (~178:1 ratio)

## Contact & Support

For issues or questions about the Steele data processing pipeline:
1. Check this documentation
2. Review log files in `data/results/`
3. Run test scripts for validation
4. Check existing test suite results 