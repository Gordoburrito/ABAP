# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a multi-data source processing framework for transforming automotive parts data from various vendors (Steele, REM, Ford, ABAP) into standardized Shopify-compatible formats. The framework follows DRY principles, uses template-based development, and implements different processing strategies based on data completeness.

## Architecture

The codebase is organized into several key components:

### Data Source Structure
Each data source (Steele/, REM/, Ford/, etc.) follows a consistent structure:
- `data/` - Raw, processed, and output data files
- `tests/` - Comprehensive test suites following TDD methodology
- `utils/` - Data source-specific utilities and transformers
- `main.py` - Entry point for processing
- `batch_processor.py` - Batch processing capabilities
- `single_processor.py` - Single item processing

### Shared Components
- `shared/` - Common utilities, base classes, and shared data
- `template/` - Template structure for creating new data sources
- `project/` - Legacy project structure (for reference)

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r project/requirements.txt

# Additional dependencies for specific data sources
pip install pandas pytest openai python-dotenv pydantic pytest-cov openpyxl
```

### Testing Commands

#### Steele Data Source (Complete Fitment Data)
```bash
cd Steele/

# Run complete TDD workflow
python run_tests.py

# Run specific test phases
pytest tests/test_steele_data_loader.py -v                    # Data loading tests
pytest tests/test_steele_golden_integration.py -v            # Golden master validation
pytest tests/test_steele_transformer.py -v                   # Transformation tests
pytest tests/test_product_import_validation.py -v            # Final format validation
pytest tests/test_steele_ai_integration.py -v -m ai          # AI integration tests (requires OPENAI_API_KEY)

# Performance tests
pytest tests/ -v -m performance

# Coverage report
pytest tests/ --cov=utils --cov-report=html --cov-report=term-missing
```

#### REM Data Source (Incomplete Fitment Data)
```bash
cd REM/

# Basic processing
python main.py
python batch_processor.py
python single_processor.py
```

#### Ford Data Source
```bash
cd Ford/

# Run tests
pytest tests/test_ford_processors.py -v
```

### Data Processing Commands

#### Create New Data Source
```bash
# Use automated setup script
python setup_new_data_source.py NewVendorName

# This creates complete directory structure with templates
```

#### Process Data
```bash
# Process single items
python single_processor.py --input data/raw/item.json --output data/processed/item.json

# Process batches
python batch_processor.py --input data/raw/batch.json --output data/results/batch.json

# Full pipeline
python main.py pipeline --input-dir data/raw --output-dir data/processed
```

## Key Processing Strategies

### Complete Fitment Data (Steele-style)
For data sources with complete year/make/model information:
- **NO AI required** - uses template-based processing
- Ultra-fast processing (1000+ products/second)
- Golden master validation as primary quality gate
- Template-based SEO and categorization
- Generates all 65 columns required by Shopify import

### Incomplete Fitment Data (REM-style)
For data sources missing fitment information:
- **AI extraction required** - uses OpenAI API to extract vehicle compatibility
- Slower processing (60-80% slower due to AI usage)
- Confidence-based processing with golden master validation
- Comprehensive AI enhancement for categorization and SEO
- Same 65-column output format

## Critical Requirements

### Shopify Column Compliance
All data transformers MUST generate exactly 65 columns in the order specified by:
`shared/data/product_import/product_import-column-requirements.py`

Key columns include:
- Required Always: Title, Body HTML, Vendor, Tags, Variant Price, Variant Cost, etc.
- Required Multi-Variants: Option fields (empty for single variants)
- Required Manual: Command (set to "MERGE"), Custom Collections, Variant SKU, etc.
- No Requirements: Optional fields (can be empty)

### Golden Master Validation
All data sources must validate vehicle compatibility against the golden master dataset:
- Located in `shared/data/master_ultimate_golden.csv`
- Validates year/make/model combinations
- Critical quality gate for all processing

## Data Flow

### Standard Pipeline
1. **Raw Data** → Load and validate structure
2. **Golden Master Validation** → Verify vehicle compatibility
3. **AI-Friendly Format** → Optimize for token efficiency (if using AI)
4. **Enhancement** → Template-based or AI-based processing
5. **Final Format** → Generate all 65 Shopify columns

### Output Formats
- **Processed**: Intermediate validation results
- **Transformed**: AI-friendly format with optimized tokens
- **Final**: Complete 65-column Shopify import format

## Important Rules

### Data Source Classification
- **Complete Fitment**: Use `.cursor/rules/completed-data.mdc` strategy
- **Incomplete Fitment**: Use `.cursor/rules/incomplete-fitment-data.mdc` strategy

### AI Usage Guidelines
- Complete fitment data: NO AI usage (template-based only)
- Incomplete fitment data: AI for extraction and enhancement
- Always validate AI extractions against golden master
- Optimize for token efficiency when using AI

### Testing Requirements
- Follow TDD methodology with comprehensive test coverage
- Include performance benchmarks
- Validate against golden master dataset
- Test complete pipeline integration
- Verify Shopify column compliance

## Common Issues and Solutions

### Performance Optimization
- Use batch processing for large datasets
- Implement token-optimized AI formats
- Cache golden master lookups
- Use pytest markers for test categorization

### Data Quality
- Always validate against golden master
- Handle missing or malformed data gracefully
- Implement confidence scoring for AI extractions
- Provide fallback processing for failed operations

### Cost Management
- Use template-based processing when possible
- Optimize AI prompts for token efficiency
- Implement confidence thresholds for AI processing
- Monitor API usage and costs

## File Structure Reference

```
├── shared/                     # Common utilities and data
│   ├── data/product_import/    # Shopify column requirements
│   ├── base_processor.py       # Base classes
│   └── logging_utils.py        # Logging configuration
├── DataSource/                 # Individual data sources
│   ├── data/                   # Input/output data
│   ├── tests/                  # TDD test suites
│   ├── utils/                  # Source-specific utilities
│   └── main.py                 # Processing entry point
├── template/                   # Template for new sources
└── project/                    # Legacy structure (reference)
```

This framework scales efficiently from 2 to 200+ data sources while maintaining code quality and developer productivity.