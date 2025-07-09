# Multi-Data Source Processing Framework

This project provides a scalable, maintainable framework for processing data from multiple sources. It follows software engineering best practices including DRY principles, separation of concerns, and template-based development.

## 🏗️ **Architecture Overview**

### Core Principles
- **DRY (Don't Repeat Yourself)**: Shared utilities prevent code duplication
- **Template Pattern**: Standardized base for new data sources  
- **Separation of Concerns**: Each data source is isolated but shares common infrastructure
- **Scalability**: Easy to add new sources without reinventing the wheel
- **Maintainability**: Fix once in shared utilities, benefits all data sources

### Project Structure

```
├── shared_utils/              # 🔧 Common utilities (DRY principle)
│   ├── __init__.py
│   ├── base_processor.py      # Abstract base classes
│   ├── file_utils.py          # File I/O operations
│   ├── logging_utils.py       # Consistent logging
│   └── config_utils.py        # Configuration management
├── template/                  # 📋 Base template for new data sources
│   ├── single_processor.py    # Template for single item processing
│   ├── batch_processor.py     # Template for batch processing
│   └── main.py               # Template for main entry point
├── config/                    # ⚙️ Shared configuration files
├── docs/                      # 📚 Documentation
├── setup_new_data_source.py   # 🚀 Script to create new data sources
├── REM/                       # 📊 REM Data Source
│   ├── batch_ids/
│   ├── data/{processed,raw,results,samples}/
│   ├── tests/
│   ├── utils/                 # REM-specific utilities
│   ├── batch_processor.py     # REM batch processing
│   ├── single_processor.py    # REM single processing
│   └── main.py               # REM entry point
└── Steele/                    # 📊 Steele Data Source
    ├── batch_ids/
    ├── data/{processed,raw,results,samples}/
    ├── tests/
    ├── utils/                 # Steele-specific utilities
    ├── batch_processor.py     # Steele batch processing
    ├── single_processor.py    # Steele single processing
    └── main.py               # Steele entry point
```

## 🚀 **Quick Start**

### Creating a New Data Source

Use the automated setup script:

```bash
python setup_new_data_source.py Ford
```

This creates a complete data source with:
- ✅ Directory structure
- ✅ Customized processing files  
- ✅ Configuration files
- ✅ Test framework
- ✅ Documentation

### Processing Data

#### REM Data Processing
```bash
cd REM
# Single item
python single_processor.py --input data/raw/item.json --output data/processed/item.json

# Batch processing  
python batch_processor.py --input data/raw/batch.json --output data/results/batch.json

# Full pipeline
python main.py pipeline --input-dir data/raw --output-dir data/processed
```

#### Steele Data Processing
```bash
cd Steele
# Single item
python single_processor.py --input data/raw/item.json --output data/processed/item.json

# Batch processing
python batch_processor.py --input data/raw/batch.json --output data/results/batch.json

# Full pipeline  
python main.py pipeline --input-dir data/raw --output-dir data/processed
```

## 🔧 **Shared Utilities**

### BaseProcessor Classes
All data sources inherit from common base classes:
- `BaseProcessor`: For single item processing
- `BaseBatchProcessor`: For batch processing

### File Management
Consistent file operations across all data sources:
- JSON/CSV reading and writing
- Directory management
- File discovery and filtering

### Logging
Centralized logging with:
- Consistent formatting across all data sources
- Automatic log file organization
- Error tracking with context

### Configuration
Hierarchical configuration system:
1. Data source specific config
2. Shared config directory  
3. Default fallback config

## 🏭 **Development Workflow**

### Adding a New Data Source

1. **Create the data source**:
   ```bash
   python setup_new_data_source.py NewSource
   ```

2. **Customize processing logic**:
   - Edit `NewSource/single_processor.py` → `_apply_data_source_logic()`
   - Edit `NewSource/batch_processor.py` → `_apply_batch_logic()`
   - Edit `NewSource/main.py` → `_is_batch_file()` and `run_full_pipeline()`

3. **Configure**:
   - Update `config/newsource.json` with specific settings

4. **Test**:
   ```bash
   cd NewSource
   python -m pytest tests/
   ```

5. **Deploy**:
   ```bash
   cd NewSource  
   python main.py pipeline --input-dir data/raw --output-dir data/processed
   ```

### Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| **DRY Compliance** | Common functionality in `shared_utils/` prevents duplication |
| **Rapid Development** | New data sources created in minutes, not hours |
| **Consistency** | All data sources follow the same patterns and interfaces |
| **Maintainability** | Bug fixes in shared code benefit all data sources |
| **Testability** | Each data source has its own test suite |
| **Scalability** | Add unlimited data sources without architectural changes |

### Testing

Each data source has isolated tests:
```bash
# Test specific data source
cd REM && python -m pytest tests/

# Test shared utilities  
cd shared_utils && python -m pytest

# Test everything
python -m pytest
```

## 📚 **Documentation**

- **Individual data sources**: See `{DataSource}/README.md`  
- **Shared utilities**: See `shared_utils/` docstrings
- **Templates**: See `template/` for customization examples

## 🔄 **Migration from Legacy**

The old project structure is preserved in `project/` directory for reference during transition.

## 🎯 **Best Practices**

1. **Keep data source logic isolated** - Only put source-specific code in data source directories
2. **Use shared utilities** - Don't duplicate common functionality  
3. **Follow the template** - Use `setup_new_data_source.py` for consistency
4. **Test thoroughly** - Each data source should have comprehensive tests
5. **Document customizations** - Update data source README files with specific logic

This framework scales efficiently from 2 data sources to 200+ while maintaining code quality and developer productivity! 🚀
