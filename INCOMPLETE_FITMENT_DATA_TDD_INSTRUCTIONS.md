# Incomplete Fitment Data Processing - TDD Implementation Guide

## Overview

This guide provides comprehensive Test-Driven Development (TDD) instructions for implementing data processing pipelines for automotive parts data sources with **incomplete fitment information**. This strategy is used when vendor data lacks complete year/make/model information and requires AI extraction to determine vehicle compatibility.

## Architecture Strategy

### Processing Philosophy
- **AI-Required Processing**: Use OpenAI API for extraction and enhancement
- **Golden Master Validation**: All vehicle compatibility validated against master dataset
- **Token Optimization**: Efficient AI prompts to minimize costs
- **Confidence-Based Processing**: Handle AI uncertainty with fallback strategies
- **Final Output**: Generate all 65 columns required for Shopify import

### Data Flow Pipeline
```
Raw Data → AI-Friendly Format → AI Extraction → Golden Master Validation → Model Refinement → Final Format (65 columns)
```

## Prerequisites

### Required Dependencies
```bash
pip install pandas pytest openai python-dotenv pydantic pytest-cov tiktoken
```

### Environment Setup
```bash
# Create .env file with:
OPENAI_API_KEY=your_openai_api_key_here
```

### Required Data Files
- `shared/data/master_ultimate_golden.csv` - Golden master dataset (constant across all data sources)
- `shared/data/product_import/product_import-column-requirements.py` - Shopify column specifications

## TDD Implementation Phases

## Phase 1: Data Loader and AI-Friendly Format Converter

### Objective
Create a standardized intermediate format that optimizes raw data for AI processing while maintaining all essential information.

### Test Cases to Implement

#### Test 1.1: Raw Data Validation
```python
def test_raw_data_loader():
    """Test loading and basic validation of raw vendor data"""
    # Test cases:
    # - Load CSV/Excel files successfully
    # - Handle missing files gracefully
    # - Detect column structure automatically
    # - Validate data types and encoding
    # - Report data quality metrics (completeness, duplicates)
```

#### Test 1.2: AI-Friendly Format Conversion
```python
def test_convert_to_ai_friendly_format():
    """Test conversion of raw data to token-optimized format"""
    # Test cases:
    # - Extract product title/name from various column formats
    # - Combine description fields into coherent product information
    # - Normalize vendor-specific terminology
    # - Remove irrelevant data to reduce token usage
    # - Create standardized product_info string for AI processing
    # - Handle edge cases (empty fields, special characters, encoding issues)
```

#### Test 1.3: Token Estimation
```python
def test_token_cost_estimation():
    """Test token counting and cost estimation functionality"""
    # Test cases:
    # - Count tokens in product information strings
    # - Estimate API costs for batch processing
    # - Provide cost breakdowns (input tokens, output tokens, total cost)
    # - Handle different AI models (gpt-4o, gpt-4.1-mini)
    # - Generate cost reports before processing
```

### Implementation Requirements

#### Data Loader Class
```python
class RawDataLoader:
    def load_data(self, file_path: str) -> pd.DataFrame
    def validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]
```

#### AI-Friendly Converter Class
```python
class AIFriendlyConverter:
    def convert_to_ai_format(self, df: pd.DataFrame) -> pd.DataFrame
    def create_product_info_string(self, row: pd.Series) -> str
    def estimate_tokens(self, text: str) -> int
    def estimate_batch_cost(self, df: pd.DataFrame) -> Dict[str, float]
```

## Phase 2: AI Product Data Extraction

### Objective
Extract structured product information from raw data using OpenAI API with proper validation and error handling.

### Test Cases to Implement

#### Test 2.1: Product Data Extraction
```python
def test_extract_product_data_with_ai():
    """Test AI-powered product data extraction"""
    # Test cases:
    # - Extract year ranges from various formats ("34/64" → "1934-1964")
    # - Extract make/model information from product titles
    # - Handle universal compatibility ("ALL" logic)
    # - Process body type specifications (2-Door, 4-Door, A-Body, etc.)
    # - Generate appropriate product types and collections
    # - Handle edge cases and malformed data
```

#### Test 2.2: Structured Output Validation
```python
def test_product_data_pydantic_validation():
    """Test Pydantic model validation for AI outputs"""
    # Test cases:
    # - Validate all required fields are present
    # - Check data types and constraints
    # - Handle validation errors gracefully
    # - Test with various AI response formats
    # - Ensure consistent output structure
```

#### Test 2.3: AI Error Handling
```python
def test_ai_error_handling():
    """Test handling of AI API errors and failures"""
    # Test cases:
    # - Handle API rate limits
    # - Manage timeouts and connection errors
    # - Process malformed AI responses
    # - Implement retry logic with exponential backoff
    # - Provide meaningful error messages
```

### Implementation Requirements

#### Product Data Model
```python
from pydantic import BaseModel, Field

class ProductData(BaseModel):
    title: str
    year_min: int
    year_max: int
    make: str
    model: str
    mpn: str
    cost: float
    price: float
    body_html: str
    collection: str
    product_type: str
    meta_title: str
    meta_description: str
```

#### AI Extraction Engine
```python
class AIProductExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini")
    def extract_product_data(self, product_info: str, golden_df: pd.DataFrame) -> ProductData
    def create_extraction_prompt(self, product_info: str, valid_options: Dict) -> str
    def handle_ai_errors(self, error: Exception) -> ProductData
```

## Phase 3: Golden Master Validation

### Objective
Validate all extracted vehicle compatibility information against the golden master dataset to ensure accuracy.

### Test Cases to Implement

#### Test 3.1: Golden Master Loading
```python
def test_load_golden_master():
    """Test loading and preprocessing of golden master dataset"""
    # Test cases:
    # - Load master_ultimate_golden.csv successfully
    # - Clean and normalize data (handle NaN values)
    # - Create lookup indices for fast validation
    # - Generate valid options lists (makes, models, years)
    # - Handle data quality issues
```

#### Test 3.2: Vehicle Compatibility Validation
```python
def test_validate_vehicle_compatibility():
    """Test validation of year/make/model combinations"""
    # Test cases:
    # - Validate individual year/make/model combinations
    # - Check year ranges against available data
    # - Handle universal "ALL" specifications
    # - Process body type constraints
    # - Report validation failures with context
```

#### Test 3.3: Invalid Data Handling
```python
def test_handle_invalid_combinations():
    """Test handling of invalid vehicle combinations"""
    # Test cases:
    # - Reject invalid year/make/model combinations
    # - Provide suggested alternatives
    # - Handle edge cases (discontinued models, special editions)
    # - Log validation failures for review
    # - Implement confidence scoring
```

### Implementation Requirements

#### Golden Master Validator
```python
class GoldenMasterValidator:
    def __init__(self, golden_master_path: str)
    def load_golden_master(self) -> pd.DataFrame
    def validate_combination(self, year: int, make: str, model: str) -> bool
    def get_valid_options(self, year_range: tuple = None, make: str = None) -> Dict
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]
```

## Phase 4: Model Refinement with Two-Pass AI Strategy

### Objective
Implement a sophisticated two-pass AI approach that uses golden master data filtering to generate accurate vehicle compatibility tags in YEAR_MAKE_MODEL format.

### Two-Pass AI Strategy

#### Pass 1: Initial Vehicle Information Extraction
**Purpose**: Extract basic vehicle information from raw product descriptions without strict validation.

**Process**:
1. Provide AI with broad golden master context (all valid makes, models, years)
2. Extract initial year_min, year_max, make, model from product description
3. Allow AI to make initial decisions with full context
4. Focus on getting rough vehicle compatibility information

#### Pass 2: Golden Master Filtered Refinement
**Purpose**: Use golden master data to filter and validate AI decisions, generating accurate vehicle-specific tags.

**Process**:
1. **Filter Golden Master**: Use Pass 1 results to filter golden master by year range and make
2. **Extract Valid Options**: Get only valid model combinations for specific year/make pairs
3. **Refined AI Context**: Provide AI with filtered valid options instead of full dataset
4. **Generate car_ids**: Extract pre-formatted YEAR_MAKE_MODEL tags from golden master
5. **Validate Tags**: Ensure all generated tags exist in authoritative golden master dataset

### Test Cases to Implement

#### Test 4.1: Two-Pass AI Integration
```python
def test_two_pass_ai_extraction():
    """Test the complete two-pass AI strategy"""
    # Test cases:
    # - Pass 1: Extract basic vehicle info from product description
    # - Pass 2: Refine using golden master filtered context
    # - Validate that Pass 2 results are more accurate than Pass 1
    # - Test with ambiguous product descriptions
    # - Verify golden master filtering works correctly
```

#### Test 4.2: Golden Master Tag Generation
```python
def test_golden_master_tag_generation():
    """Test generation of YEAR_MAKE_MODEL tags from golden master car_ids"""
    # Test cases:
    # - Extract car_id values from golden master dataset
    # - Generate tags like "1939_Hupmobile_Skylark" from car_id field
    # - Validate all generated tags exist in golden master
    # - Handle multiple vehicle compatibility (multiple car_ids)
    # - Test performance with large datasets
```

#### Test 4.3: Golden Master Filtering
```python
def test_golden_master_filtering():
    """Test filtering of golden master data for AI context"""
    # Test cases:
    # - Filter by year range from Pass 1 results
    # - Filter by make from Pass 1 results
    # - Extract valid models for specific year/make combinations
    # - Test edge cases (invalid combinations, missing data)
    # - Validate filtered results provide better AI context
```

#### Test 4.4: Body Type Processing with Golden Master
```python
def test_body_type_processing():
    """Test processing of body type specifications using golden master"""
    # Test cases:
    # - Handle "ALL (2-Door)" → models available as 2-door variants from golden master
    # - Handle "ALL (4-Door)" → models available as 4-door variants from golden master
    # - Process "ALL (A-Body)" → models with A-Body chassis from golden master
    # - Handle combined specifications "2 & 4-Door Sedan"
    # - Validate against historical model configurations in golden master
```

### Implementation Requirements

#### Two-Pass AI Engine
```python
class TwoPassAIEngine:
    def __init__(self, client: OpenAI, golden_df: pd.DataFrame)
    def extract_initial_vehicle_info(self, product_info: str) -> Dict[str, Any]  # Pass 1
    def refine_with_golden_master(self, initial_data: Dict, product_info: str) -> Dict[str, Any]  # Pass 2
    def filter_golden_master_context(self, year_min: int, year_max: int, make: str) -> pd.DataFrame
    def get_valid_models_for_context(self, filtered_df: pd.DataFrame) -> List[str]
```

#### Golden Master Tag Generator
```python
class GoldenMasterTagGenerator:
    def __init__(self, golden_df: pd.DataFrame)
    def generate_vehicle_tags_from_car_ids(self, year_min: int, year_max: int, make: str, models: List[str]) -> List[str]
    def extract_car_ids_from_golden_master(self, year: int, make: str, models: List[str]) -> List[str]
    def validate_tags_against_golden_master(self, tags: List[str]) -> bool
    def get_car_id_format_tags(self, filtered_df: pd.DataFrame) -> List[str]
```

#### Model Refinement Engine (Updated)
```python
class ModelRefinementEngine:
    def __init__(self, client: OpenAI, golden_df: pd.DataFrame)
    def refine_models_with_two_pass_ai(self, df: pd.DataFrame) -> pd.DataFrame
    def get_refined_models_from_ai(self, title: str, year: int, make: str, models_str: str, valid_models: List[str]) -> str
    def expand_models_to_car_ids(self, row: pd.Series) -> List[str]
    def validate_model_selection(self, models: List[str], year: int, make: str) -> bool
```

### Golden Master Integration Points

#### 1. Car ID Extraction Pattern
```python
# Filter golden master by validated year/make/model combinations
year_mask = golden_df["year"].astype(int) == year
make_mask = golden_df["make"] == make  
model_mask = golden_df["model"].isin(refined_models)
car_ids = golden_df[year_mask & make_mask & model_mask]["car_id"].unique()
```

#### 2. Tag Generation from Golden Master
```python
# Extract pre-formatted YEAR_MAKE_MODEL tags from car_id field
vehicle_tags = [car_id for car_id in car_ids if car_id]  # car_id already in YEAR_MAKE_MODEL format
```

#### 3. Context Filtering for Pass 2
```python
# Filter golden master to provide accurate context for AI Pass 2
filtered_golden = golden_df[
    (golden_df["year"].astype(int) >= year_min) & 
    (golden_df["year"].astype(int) <= year_max) &
    (golden_df["make"] == make)
]
valid_models = filtered_golden["model"].unique()
```

### Key Benefits of Two-Pass Approach

1. **Accuracy**: Pass 2 uses filtered golden master context instead of full dataset
2. **Validation**: All generated tags are validated against authoritative golden master
3. **Performance**: Filtered context reduces AI token usage and improves accuracy
4. **Reliability**: No random tag generation - all tags come from golden master car_id field
5. **Consistency**: YEAR_MAKE_MODEL format is pre-formatted in golden master dataset

### Vehicle Tag Generation Strategy

#### Tag Format Requirements
- **ONLY Vehicle-Specific Tags**: Tags should contain ONLY YEAR_MAKE_MODEL format tags
- **Format**: "1939_Hupmobile_Skylark, 1940_Hupmobile_Skylark, 1941_Hupmobile_Skylark"
- **Source**: Extract from golden master car_id field (pre-formatted)
- **No Descriptive Tags**: Remove all generic tags like "Automotive", "Parts", "Replacement"

#### Golden Master car_id Field
- **Pre-Formatted**: Golden master contains car_id field with YEAR_MAKE_MODEL format
- **Authoritative**: All valid vehicle combinations are in golden master dataset
- **No Manual Formatting**: Use car_id values directly, no string manipulation needed
- **Validation**: Every tag must exist in golden master car_id field

#### Example Tag Generation Process
```python
# Pass 1: Extract initial vehicle info
initial_data = {
    'year_min': 1939,
    'year_max': 1941, 
    'make': 'Hupmobile',
    'model': 'ALL (4-Door)'
}

# Pass 2: Filter golden master and extract car_ids
filtered_golden = golden_df[
    (golden_df["year"].astype(int) >= 1939) & 
    (golden_df["year"].astype(int) <= 1941) &
    (golden_df["make"] == "Hupmobile")
]

# Get refined models using AI with filtered context
refined_models = get_refined_models_from_ai(..., valid_models=filtered_golden["model"].unique())

# Extract car_ids (already in YEAR_MAKE_MODEL format)
for year in range(1939, 1942):
    year_mask = filtered_golden["year"].astype(int) == year
    model_mask = filtered_golden["model"].isin(refined_models)
    car_ids = filtered_golden[year_mask & model_mask]["car_id"].unique()
    vehicle_tags.extend(car_ids.tolist())

# Result: ["1939_Hupmobile_Skylark", "1940_Hupmobile_Skylark", "1941_Hupmobile_Skylark"]
```

## Phase 5: Final Format Generation (65-Column Shopify Compliance)

### Objective
Generate all 65 columns required for Shopify import in the exact format specified by the product import requirements.

### Test Cases to Implement

#### Test 5.1: Column Mapping
```python
def test_shopify_column_mapping():
    """Test mapping of processed data to Shopify column requirements"""
    # Test cases:
    # - Map all 65 required columns correctly
    # - Handle required vs optional columns
    # - Generate appropriate default values
    # - Validate column order and naming
    # - Ensure data type compliance
```

#### Test 5.2: Product Variants
```python
def test_product_variants():
    """Test handling of single vs multi-variant products"""
    # Test cases:
    # - Generate single variant products correctly
    # - Handle multi-variant products (if applicable)
    # - Set appropriate variant fields
    # - Validate pricing and inventory data
    # - Handle variant-specific attributes
```

#### Test 5.3: SEO and Content Generation
```python
def test_seo_content_generation():
    """Test generation of SEO-optimized content"""
    # Test cases:
    # - Generate meta titles within character limits
    # - Create meta descriptions with proper length
    # - Generate HTML product descriptions
    # - Handle special characters and formatting
    # - Validate content quality and relevance
```

### Implementation Requirements

#### Shopify Format Generator
```python
class ShopifyFormatGenerator:
    def __init__(self, column_requirements_path: str)
    def load_column_requirements(self) -> Dict[str, Any]
    def generate_shopify_format(self, df: pd.DataFrame) -> pd.DataFrame
    def validate_column_compliance(self, df: pd.DataFrame) -> Dict[str, Any]
    def generate_seo_content(self, product_data: ProductData) -> Dict[str, str]
```

## Phase 6: Integration Testing and Batch Processing

### Objective
Test the complete pipeline integration and implement efficient batch processing capabilities.

### Test Cases to Implement

#### Test 6.1: End-to-End Pipeline
```python
def test_complete_pipeline():
    """Test the complete data processing pipeline"""
    # Test cases:
    # - Process sample data through entire pipeline
    # - Validate output format and quality
    # - Check processing time and resource usage
    # - Test error handling and recovery
    # - Validate against golden master throughout
```

#### Test 6.2: Batch Processing
```python
def test_batch_processing():
    """Test batch processing capabilities"""
    # Test cases:
    # - Process large datasets efficiently
    # - Handle OpenAI batch API integration
    # - Implement progress tracking and logging
    # - Test resume functionality for interrupted processes
    # - Validate batch vs single processing consistency
```

#### Test 6.3: Performance and Cost Optimization
```python
def test_performance_optimization():
    """Test performance and cost optimization features"""
    # Test cases:
    # - Measure processing speed (items per minute)
    # - Track API usage and costs
    # - Test token optimization strategies
    # - Validate memory usage and scalability
    # - Generate performance reports
```

### Implementation Requirements

#### Pipeline Orchestrator
```python
class IncompleteDataPipeline:
    def __init__(self, config: Dict[str, Any])
    def process_complete_pipeline(self, input_path: str, output_path: str) -> Dict[str, Any]
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame
    def generate_processing_report(self) -> Dict[str, Any]
    def estimate_processing_cost(self, df: pd.DataFrame) -> Dict[str, float]
```

#### Batch Processor
```python
class BatchProcessor:
    def __init__(self, client: OpenAI, batch_size: int = 100)
    def create_batch_tasks(self, df: pd.DataFrame) -> List[Dict]
    def submit_batch_job(self, tasks: List[Dict]) -> str
    def monitor_batch_progress(self, batch_id: str) -> Dict[str, Any]
    def process_batch_results(self, batch_id: str) -> pd.DataFrame
```

## Testing Strategy

### Test Structure
```
tests/
├── test_data_loader.py                 # Phase 1 tests
├── test_ai_extraction.py               # Phase 2 tests
├── test_golden_master_validation.py    # Phase 3 tests
├── test_model_refinement.py            # Phase 4 tests
├── test_shopify_format.py              # Phase 5 tests
├── test_integration.py                 # Phase 6 tests
├── fixtures/
│   ├── sample_raw_data.csv
│   ├── sample_golden_master.csv
│   └── expected_outputs/
└── utils/
    ├── test_helpers.py
    └── mock_openai_responses.py
```

### Test Execution Commands
```bash
# Run tests by phase
pytest tests/test_data_loader.py -v
pytest tests/test_ai_extraction.py -v
pytest tests/test_golden_master_validation.py -v
pytest tests/test_model_refinement.py -v
pytest tests/test_shopify_format.py -v
pytest tests/test_integration.py -v

# Run all tests with coverage
pytest tests/ --cov=utils --cov-report=html --cov-report=term-missing

# Run performance tests
pytest tests/ -v -m performance

# Run AI integration tests (requires API key)
pytest tests/ -v -m ai
```

## Cost Management and Optimization

### Token Optimization Strategies
1. **Efficient Prompting**: Use structured prompts that minimize input tokens
2. **Batch Processing**: Use OpenAI Batch API for 50% cost reduction
3. **Model Selection**: Use gpt-4.1-mini for extraction, gpt-4o for refinement
4. **Caching**: Cache AI responses for similar products
5. **Preprocessing**: Clean and optimize data before AI processing

### Cost Estimation Tools
```python
# Example cost estimation
def estimate_processing_cost(df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns:
    {
        'input_tokens': 150000,
        'output_tokens': 75000,
        'total_cost': 45.50,
        'cost_per_item': 0.091
    }
    """
```

## Error Handling and Recovery

### Error Categories
1. **Data Quality Errors**: Missing fields, malformed data
2. **API Errors**: Rate limits, timeouts, authentication
3. **Validation Errors**: Invalid vehicle combinations
4. **Processing Errors**: Memory issues, file I/O problems

### Recovery Strategies
1. **Graceful Degradation**: Process what's possible, log what fails
2. **Retry Logic**: Exponential backoff for API calls
3. **Fallback Processing**: Template-based processing when AI fails
4. **Detailed Logging**: Comprehensive error reporting and debugging

## Success Criteria

### Quality Metrics
- **Golden Master Validation**: 95%+ accuracy for vehicle compatibility
- **Data Completeness**: All 65 Shopify columns populated
- **Processing Success Rate**: 98%+ successful transformations
- **Cost Efficiency**: Token optimization achieving target cost per item

### Performance Metrics
- **Processing Speed**: Target based on AI model and batch size
- **Memory Usage**: Efficient handling of large datasets
- **Error Rate**: <2% processing failures
- **API Usage**: Optimal token usage and batch processing

## Implementation Checklist

### Phase 1: Foundation ✓
- [ ] Raw data loader with validation
- [ ] AI-friendly format converter
- [ ] Token estimation and cost calculation
- [ ] Basic error handling

### Phase 2: AI Extraction ✓
- [ ] Product data extraction with OpenAI
- [ ] Pydantic model validation
- [ ] Error handling and retry logic
- [ ] Structured output processing

### Phase 3: Validation ✓
- [ ] Golden master dataset integration
- [ ] Vehicle compatibility validation
- [ ] Invalid combination handling
- [ ] Validation reporting

### Phase 4: Refinement ✓
- [ ] Body type processing logic
- [ ] AI model refinement
- [ ] Car ID expansion
- [ ] Historical accuracy validation

### Phase 5: Final Format ✓
- [ ] 65-column Shopify compliance
- [ ] SEO content generation
- [ ] Product variant handling
- [ ] Format validation

### Phase 6: Integration ✓
- [ ] Complete pipeline testing
- [ ] Batch processing implementation
- [ ] Performance optimization
- [ ] Production deployment

## Usage Example

```python
# Initialize pipeline for new data source
pipeline = IncompleteDataPipeline({
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'golden_master_path': 'shared/data/master_ultimate_golden.csv',
    'batch_size': 100,
    'model': 'gpt-4.1-mini'
})

# Process raw data
results = pipeline.process_complete_pipeline(
    input_path='data/raw/new_vendor_data.csv',
    output_path='data/processed/shopify_import.csv'
)

# Review processing report
print(f"Processed {results['total_items']} items")
print(f"Success rate: {results['success_rate']:.1%}")
print(f"Total cost: ${results['total_cost']:.2f}")
```

This comprehensive guide provides the foundation for implementing robust, test-driven data processing pipelines for incomplete fitment data sources using AI enhancement and golden master validation.