# Ford Data Processing

This directory contains data processing infrastructure for Ford data source.

## Files

- `single_processor.py` - Process individual Ford items
- `batch_processor.py` - Process batches of Ford items  
- `main.py` - Main entry point and pipeline orchestrator

## Usage

### Single Item Processing
```bash
cd Ford
python single_processor.py --input data/raw/item.json --output data/processed/item.json
```

### Batch Processing
```bash
cd Ford
python batch_processor.py --input data/raw/batch.json --output data/results/batch_results.json
```

### Full Pipeline
```bash
cd Ford
python main.py pipeline --input-dir data/raw --output-dir data/processed
```

## Customization

1. Edit `single_processor.py` and customize the `_apply_data_source_logic()` method
2. Edit `batch_processor.py` and customize the `_apply_batch_logic()` method  
3. Edit `main.py` and customize the `_is_batch_file()` and `run_full_pipeline()` methods
4. Update the configuration in `config/ford.json`

## Data Structure

```
Ford/
├── batch_ids/              # Batch identifiers
├── data/
│   ├── processed/          # Processed data
│   ├── raw/               # Raw input data
│   ├── results/           # Final results
│   └── samples/           # Sample data
├── tests/                 # Tests
├── utils/                 # Utilities specific to Ford
├── batch_processor.py     # Batch processing
├── single_processor.py    # Single item processing
└── main.py               # Main entry point
```
