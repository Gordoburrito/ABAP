#!/usr/bin/env python3
"""
Setup script to create new data sources from template.
Usage: python setup_new_data_source.py <DataSourceName>
"""

import os
import shutil
import sys
from pathlib import Path


def create_data_source(data_source_name: str):
    """
    Create a new data source from the template.
    
    Args:
        data_source_name: Name of the new data source (e.g., 'Ford', 'Toyota')
    """
    # Validate data source name
    if not data_source_name.isalnum():
        print(f"Error: Data source name '{data_source_name}' must be alphanumeric")
        return False
    
    data_source_lower = data_source_name.lower()
    
    # Check if data source already exists
    if os.path.exists(data_source_name):
        print(f"Error: Data source '{data_source_name}' already exists")
        return False
    
    print(f"Creating new data source: {data_source_name}")
    
    # Create directory structure
    directories = [
        f"{data_source_name}/batch_ids",
        f"{data_source_name}/data/processed", 
        f"{data_source_name}/data/raw",
        f"{data_source_name}/data/results",
        f"{data_source_name}/data/samples",
        f"{data_source_name}/utils",
        f"{data_source_name}/tests",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}")
    
    # Copy and customize template files
    template_files = [
        "single_processor.py",
        "batch_processor.py", 
        "main.py"
    ]
    
    for template_file in template_files:
        template_path = f"template/{template_file}"
        target_path = f"{data_source_name}/{template_file}"
        
        if not os.path.exists(template_path):
            print(f"Warning: Template file {template_path} not found")
            continue
            
        # Read template content
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('{{DATA_SOURCE}}', data_source_name)
        content = content.replace('{{DATA_SOURCE_LOWER}}', data_source_lower)
        
        # Write customized file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created file: {target_path}")
    
    # Create initial configuration file
    config_content = f'''{{
  "data_source": "{data_source_name}",
  "processing": {{
    "batch_size": 100,
    "max_retries": 3,
    "timeout_seconds": 300
  }},
  "logging": {{
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }},
  "paths": {{
    "raw_data": "{data_source_name}/data/raw",
    "processed_data": "{data_source_name}/data/processed", 
    "results": "{data_source_name}/data/results",
    "samples": "{data_source_name}/data/samples"
  }},
  "features": {{
    "enable_preprocessing": true,
    "enable_validation": true,
    "enable_postprocessing": true
  }}
}}'''
    
    config_path = f"config/{data_source_lower}.json"
    os.makedirs("config", exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"  Created config: {config_path}")
    
    # Create README for the new data source
    readme_content = f"""# {data_source_name} Data Processing

This directory contains data processing infrastructure for {data_source_name} data source.

## Files

- `single_processor.py` - Process individual {data_source_name} items
- `batch_processor.py` - Process batches of {data_source_name} items  
- `main.py` - Main entry point and pipeline orchestrator

## Usage

### Single Item Processing
```bash
cd {data_source_name}
python single_processor.py --input data/raw/item.json --output data/processed/item.json
```

### Batch Processing
```bash
cd {data_source_name}
python batch_processor.py --input data/raw/batch.json --output data/results/batch_results.json
```

### Full Pipeline
```bash
cd {data_source_name}
python main.py pipeline --input-dir data/raw --output-dir data/processed
```

## Customization

1. Edit `single_processor.py` and customize the `_apply_data_source_logic()` method
2. Edit `batch_processor.py` and customize the `_apply_batch_logic()` method  
3. Edit `main.py` and customize the `_is_batch_file()` and `run_full_pipeline()` methods
4. Update the configuration in `config/{data_source_lower}.json`

## Data Structure

```
{data_source_name}/
├── batch_ids/              # Batch identifiers
├── data/
│   ├── processed/          # Processed data
│   ├── raw/               # Raw input data
│   ├── results/           # Final results
│   └── samples/           # Sample data
├── tests/                 # Tests
├── utils/                 # Utilities specific to {data_source_name}
├── batch_processor.py     # Batch processing
├── single_processor.py    # Single item processing
└── main.py               # Main entry point
```
"""
    
    readme_path = f"{data_source_name}/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  Created README: {readme_path}")
    
    # Create sample test file
    test_content = f'''"""
Tests for {data_source_name} data processing.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_processor import {data_source_name}SingleProcessor
from batch_processor import {data_source_name}BatchProcessor


class Test{data_source_name}SingleProcessor(unittest.TestCase):
    """Test cases for {data_source_name} single processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = {data_source_name}SingleProcessor()
    
    def test_process_item(self):
        """Test basic item processing."""
        test_item = {{
            "id": "test_001",
            "title": "Test Item",
            "description": "This is a test item"
        }}
        
        result = self.processor.process_item(test_item)
        
        # Check that basic processing occurred
        self.assertIn("data_source", result)
        self.assertEqual(result["data_source"], "{data_source_name}")
        self.assertIn("processing_timestamp", result)
        self.assertIn("processing_completed", result)


class Test{data_source_name}BatchProcessor(unittest.TestCase):
    """Test cases for {data_source_name} batch processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = {data_source_name}BatchProcessor()
    
    def test_process_batch(self):
        """Test basic batch processing."""
        test_batch = [
            {{"id": "test_001", "title": "Test Item 1"}},
            {{"id": "test_002", "title": "Test Item 2"}}
        ]
        
        results = self.processor.process_batch(test_batch)
        
        # Check that all items were processed
        self.assertEqual(len(results), 2)
        
        # Check that each item has required fields
        for result in results:
            self.assertIn("data_source", result)
            self.assertEqual(result["data_source"], "{data_source_name}")


if __name__ == "__main__":
    unittest.main()
'''
    
    test_path = f"{data_source_name}/tests/test_{data_source_lower}_processors.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"  Created test file: {test_path}")
    
    # Create __init__.py files to make directories Python packages
    init_files = [
        f"{data_source_name}/__init__.py",
        f"{data_source_name}/utils/__init__.py", 
        f"{data_source_name}/tests/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(f'"""{data_source_name} package."""\n')
        print(f"  Created package file: {init_file}")
    
    print(f"\n✅ Successfully created {data_source_name} data source!")
    print(f"\nNext steps:")
    print(f"1. Review and customize the processing logic in {data_source_name}/")
    print(f"2. Add your data files to {data_source_name}/data/raw/")
    print(f"3. Run tests: cd {data_source_name} && python -m pytest tests/")
    print(f"4. Start processing: cd {data_source_name} && python main.py --help")
    
    return True


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python setup_new_data_source.py <DataSourceName>")
        print("Example: python setup_new_data_source.py Ford")
        sys.exit(1)
    
    data_source_name = sys.argv[1]
    
    if create_data_source(data_source_name):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 