import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from Steele.utils.steele_data_transformer import AIVehicleTagGenerator, SteeleDataTransformer

# Import product import requirements for testing
sys.path.append(str(project_root / "shared" / "data" / "product_import"))
try:
    # Import column requirements
    exec(open(str(project_root / "shared" / "data" / "product_import" / "product_import-column-requirements.py")).read())
except Exception as e:
    print(f"Warning: Could not load product import requirements: {e}")
    # Fallback minimal columns for testing
    cols_list = ["Title", "Body HTML", "Vendor", "Tags", "Variant SKU", "Variant Price", "Custom Collections"]

class TestAIVehicleTagGeneration:
    """Test suite specifically for AI vehicle tag generation functionality"""
    
    @pytest.fixture
    def ai_tag_generator(self):
        """Create AI tag generator for testing"""
        return AIVehicleTagGenerator()
    
    @pytest.fixture
    def sample_vehicle_data(self):
        """Sample vehicle data matching Steele format"""
        return [
            {"year": "1928", "make": "Stutz", "model": "Stutz", "product": "Accelerator Pedal"},
            {"year": "1930", "make": "Durant", "model": "Model 6-14", "product": "Brake Pad"},
            {"year": "1929", "make": "Chrysler", "model": "Series 65", "product": "Headlight"},
        ]
    
    def test_ai_tag_generator_initialization(self, ai_tag_generator):
        """Test that AI tag generator initializes properly"""
        assert ai_tag_generator.ai_available is True
        assert ai_tag_generator.golden_df is not None
        assert len(ai_tag_generator.golden_df) > 300000  # Should have full golden dataset
        print(f"✅ Golden master loaded: {len(ai_tag_generator.golden_df)} records")
    
    def test_golden_master_loading(self, ai_tag_generator):
        """Test that golden master dataset loads correctly"""
        df = ai_tag_generator.golden_df
        
        # Check required columns exist
        required_cols = ['Year', 'Make', 'Model']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert df['Year'].min() >= 1800, "Year values seem invalid"
        assert df['Year'].max() <= 2030, "Year values seem invalid"
        assert df['Make'].notna().sum() > 100000, "Too many missing Make values"
        assert df['Model'].notna().sum() > 100000, "Too many missing Model values"
        
        print(f"✅ Golden master validation passed")
        print(f"   Years: {df['Year'].min()} - {df['Year'].max()}")
        print(f"   Makes: {df['Make'].nunique()} unique")
        print(f"   Models: {df['Model'].nunique()} unique")
    
    @pytest.mark.ai
    def test_step1_year_make_extraction(self, ai_tag_generator, sample_vehicle_data):
        """Test Step 1: AI extracts/confirms Year and Make"""
        
        for vehicle in sample_vehicle_data:
            year, make = ai_tag_generator._extract_year_and_make(
                vehicle["year"], vehicle["make"], vehicle["model"], vehicle["product"]
            )
            
            # Validate outputs
            assert len(year) == 4, f"Year should be 4 digits: {year}"
            assert year.isdigit(), f"Year should be numeric: {year}"
            assert len(make) > 0, f"Make should not be empty: {make}"
            assert make != "NONE", f"Make should be valid: {make}"
            
            print(f"✅ Step 1 - {vehicle['year']} {vehicle['make']} → {year} {make}")
    
    def test_step2_real_model_lookup(self, ai_tag_generator):
        """Test Step 2: Query golden master for real models"""
        
        test_cases = [
            ("1928", "Stutz"),
            ("1930", "Durant"), 
            ("1929", "Chrysler"),
            ("1965", "Ford"),  # Should have many models
            ("1999", "Toyota"), # Modern vehicle
        ]
        
        for year, make in test_cases:
            models = ai_tag_generator._get_real_models_for_year_make(year, make)
            
            # Validate results
            assert isinstance(models, list), "Should return a list"
            print(f"✅ Step 2 - {year} {make}: {len(models)} models found")
            
            if len(models) > 0:
                print(f"   Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                
                # Validate model quality
                for model in models:
                    assert isinstance(model, str), "Models should be strings"
                    assert len(model) > 0, "Models should not be empty"
                    assert model != "nan", "Models should not be 'nan'"
    
    @pytest.mark.ai
    def test_step3_model_selection(self, ai_tag_generator):
        """Test Step 3: AI selects best model from real options"""
        
        # Test with known models from golden master
        test_cases = [
            {
                "year": "1928", "make": "Stutz", "original_model": "Stutz",
                "product": "Accelerator Pedal", "real_models": ["Series BB"]
            },
            {
                "year": "1930", "make": "Durant", "original_model": "Model 6-14", 
                "product": "Brake Pad", "real_models": ["Model 614", "Model 617"]
            },
            {
                "year": "1929", "make": "Chrysler", "original_model": "Series 65",
                "product": "Headlight", "real_models": ["Series 65", "Series 66", "Series 70"]
            }
        ]
        
        for case in test_cases:
            selected = ai_tag_generator._select_best_model_from_options(
                case["year"], case["make"], case["original_model"],
                case["product"], case["real_models"]
            )
            
            # Validate selection
            assert selected in case["real_models"], f"Selected model must be from real options: {selected}"
            print(f"✅ Step 3 - {case['original_model']} → {selected} (from {case['real_models']})")
    
    @pytest.mark.ai 
    def test_complete_vehicle_tag_generation(self, ai_tag_generator, sample_vehicle_data):
        """Test complete end-to-end vehicle tag generation"""
        
        for vehicle in sample_vehicle_data:
            tag = ai_tag_generator.generate_accurate_vehicle_tag(
                vehicle["year"], vehicle["make"], vehicle["model"], vehicle["product"]
            )
            
            # Validate tag format
            assert isinstance(tag, str), "Tag should be a string"
            assert len(tag) > 0, "Tag should not be empty"
            assert tag.count('_') >= 2, "Tag should have format YEAR_MAKE_MODEL"
            
            # Validate tag structure
            parts = tag.split('_')
            assert len(parts) >= 3, f"Tag should have at least 3 parts: {tag}"
            assert parts[0].isdigit(), f"First part should be year: {parts[0]}"
            assert len(parts[0]) == 4, f"Year should be 4 digits: {parts[0]}"
            
            print(f"✅ Complete tag generation: {vehicle['year']} {vehicle['make']} {vehicle['model']} → {tag}")
    
    def test_ai_disabled_fallback(self):
        """Test fallback behavior when AI is disabled"""
        
        # Create transformer with AI disabled
        transformer = SteeleDataTransformer(use_ai=False)
        
        # Test fallback tag generation
        product_data = type('ProductData', (), {
            'year_min': '1928',
            'make': 'Stutz', 
            'model': 'Stutz',
            'title': 'Test Product'
        })()
        
        tag = transformer._generate_vehicle_tag(product_data)
        
        # Should use simple format
        expected = "1928_Stutz_Stutz"
        assert tag == expected, f"Fallback tag should be simple format: {tag}"
        print(f"✅ AI disabled fallback: {tag}")
    
    def test_invalid_vehicle_data_handling(self, ai_tag_generator):
        """Test handling of invalid or incomplete vehicle data"""
        
        test_cases = [
            {"year": "", "make": "Ford", "model": "Mustang"},
            {"year": "1965", "make": "", "model": "Mustang"},
            {"year": "1965", "make": "Ford", "model": ""},
            {"year": "invalid", "make": "Ford", "model": "Mustang"},
        ]
        
        for case in test_cases:
            tag = ai_tag_generator.generate_accurate_vehicle_tag(
                case["year"], case["make"], case["model"], "Test Product"
            )
            
            # Should handle gracefully
            assert isinstance(tag, str), "Should return string even for invalid data"
            print(f"✅ Invalid data handling: {case} → {tag}")
    
    def test_real_steele_data_integration(self):
        """Test with actual Steele sample data"""
        
        transformer = SteeleDataTransformer(use_ai=True)
        
        # Load small sample
        steele_df = transformer.load_sample_data("data/samples/steele_sample.csv")
        steele_sample = steele_df.head(3)  # Test with just 3 records
        
        # Transform to standard format
        validation_df = transformer.validate_against_golden_dataset(steele_sample)
        standard_products = transformer.transform_to_standard_format(steele_sample, validation_df)
        
        # Test vehicle tag generation
        for product in standard_products:
            tag = transformer._generate_vehicle_tag(product)
            
            assert isinstance(tag, str), "Tag should be string"
            if product.make != "NONE" and product.model != "NONE":
                assert len(tag) > 0, "Tag should not be empty for valid vehicles"
                print(f"✅ Real data: {product.year_min} {product.make} {product.model} → {tag}")
    
    def test_performance_benchmark(self, ai_tag_generator):
        """Test performance of vehicle tag generation"""
        import time
        
        test_vehicles = [
            ("1928", "Stutz", "Stutz", "Test Product"),
            ("1930", "Durant", "Model 6-14", "Test Product"),
            ("1929", "Chrysler", "Series 65", "Test Product"),
        ] * 3  # Test 9 total
        
        start_time = time.time()
        
        for year, make, model, product in test_vehicles:
            tag = ai_tag_generator.generate_accurate_vehicle_tag(year, make, model, product)
            assert len(tag) > 0, "Should generate valid tag"
        
        end_time = time.time()
        duration = end_time - start_time
        rate = len(test_vehicles) / duration
        
        print(f"✅ Performance: {len(test_vehicles)} tags in {duration:.2f}s ({rate:.1f} tags/sec)")
        
        # Should be reasonable performance
        assert rate > 0.1, f"Performance too slow: {rate} tags/sec"
    
    def test_golden_master_coverage(self, ai_tag_generator):
        """Test coverage of golden master dataset"""
        
        df = ai_tag_generator.golden_df
        
        # Test various makes that should exist
        expected_makes = ['Ford', 'Chevrolet', 'Chrysler', 'Stutz', 'Durant']
        
        for make in expected_makes:
            make_data = df[df['Make'].str.upper() == make.upper()]
            assert len(make_data) > 0, f"Make '{make}' should exist in golden master"
            
            # Check year range
            years = make_data['Year'].unique()
            print(f"✅ {make}: {len(make_data)} records, years {min(years)}-{max(years)}")
    
    @pytest.mark.ai
    def test_ai_consistency(self, ai_tag_generator):
        """Test that AI gives consistent results for same input"""
        
        # Test same vehicle multiple times
        test_vehicle = ("1928", "Stutz", "Stutz", "Accelerator Pedal")
        
        results = []
        for i in range(3):
            tag = ai_tag_generator.generate_accurate_vehicle_tag(*test_vehicle)
            results.append(tag)
        
        # Should be consistent (AI temperature is low with gpt-4.1-mini)
        assert len(set(results)) <= 2, f"Results should be consistent: {results}"
        print(f"✅ Consistency test: {results}")
    
    def test_generic_model_detection_multi_tags(self, ai_tag_generator):
        """Test that generic models generate tags for ALL variants"""
        
        test_cases = [
            {
                "year": "1933", "make": "Stutz", "model": "Stutz", 
                "product": "Axle Rebound Pad",
                "expected_models": ["Model DV-32", "Model LAA", "Model SV-16"]  # From golden master
            },
            {
                "year": "1930", "make": "Durant", "model": "Durant",
                "product": "Brake Pad", 
                "expected_min_models": 2  # Should have multiple Durant models
            }
        ]
        
        for case in test_cases:
            # First verify the real models exist
            real_models = ai_tag_generator._get_real_models_for_year_make(case["year"], case["make"])
            
            if "expected_models" in case:
                # Check that expected models are found
                for expected_model in case["expected_models"]:
                    assert expected_model in real_models, f"Expected model {expected_model} not found in golden master"
            
            # Test generic model detection
            selected = ai_tag_generator._select_best_model_from_options(
                case["year"], case["make"], case["model"], case["product"], real_models
            )
            
            # Should return a list when generic model detected
            assert isinstance(selected, list), f"Generic model should return list, got: {type(selected)}"
            assert len(selected) > 1, f"Generic model should return multiple models, got: {len(selected)}"
            
            # Test complete tag generation
            tag_result = ai_tag_generator.generate_accurate_vehicle_tag(
                case["year"], case["make"], case["model"], case["product"]
            )
            
            # Should contain multiple tags separated by commas
            assert "," in tag_result, f"Generic model should generate multiple tags: {tag_result}"
            
            individual_tags = tag_result.split(", ")
            assert len(individual_tags) > 1, f"Should have multiple individual tags: {individual_tags}"
            
            # Each tag should be valid format
            for tag in individual_tags:
                parts = tag.split("_")
                assert len(parts) >= 3, f"Each tag should have YEAR_MAKE_MODEL format: {tag}"
                assert parts[0] == case["year"], f"Year should match: {parts[0]} vs {case['year']}"
                assert case["make"].replace(' ', '_') in tag, f"Make should be in tag: {tag}"
            
            print(f"✅ Generic model test: {case['year']} {case['make']} {case['model']}")
            print(f"   Generated {len(individual_tags)} tags: {tag_result}")
    
    def test_specific_model_single_tag(self, ai_tag_generator):
        """Test that specific models still generate single tags"""
        
        # Test with a specific model that should match exactly
        test_cases = [
            {
                "year": "1929", "make": "Chrysler", "model": "Series 65",
                "product": "Headlight", "expected_single": True
            }
        ]
        
        for case in test_cases:
            real_models = ai_tag_generator._get_real_models_for_year_make(case["year"], case["make"])
            
            # Should have Series 65 as one of the models
            assert "Series 65" in real_models, f"Series 65 should be in golden master"
            
            # Test specific model selection (should return single model)
            selected = ai_tag_generator._select_best_model_from_options(
                case["year"], case["make"], case["model"], case["product"], real_models
            )
            
            # Should return string (single model) for specific model
            assert isinstance(selected, str), f"Specific model should return string, got: {type(selected)}"
            assert selected == "Series 65", f"Should select exact match: {selected}"
            
            # Test complete tag generation
            tag_result = ai_tag_generator.generate_accurate_vehicle_tag(
                case["year"], case["make"], case["model"], case["product"]
            )
            
            # Should be single tag
            assert "," not in tag_result, f"Specific model should generate single tag: {tag_result}"
            assert "1929_Chrysler_Series_65" == tag_result, f"Tag should match exactly: {tag_result}"
            
            print(f"✅ Specific model test: {case['year']} {case['make']} {case['model']} → {tag_result}")
    
    def test_product_consolidation_by_sku(self):
        """Test consolidation of products by unique SKU with combined tags"""
        
        # Create transformer for testing
        transformer = SteeleDataTransformer(use_ai=False)  # Use simple tags for testing
        
        # Create test DataFrame with duplicate SKUs but different vehicle tags
        test_data = {
            'Title': ['Test Product A', 'Test Product A', 'Test Product B', 'Test Product B', 'Test Product C'],
            'Variant SKU': ['SKU-001', 'SKU-001', 'SKU-002', 'SKU-002', 'SKU-003'],
            'Tags': ['1930_Ford_Model_A', '1931_Ford_Model_A', '1965_Ford_Mustang', '1966_Ford_Mustang', '1955_Chevy_Bel_Air'],
            'Body HTML': ['Description A short', 'Description A longer version', 'Description B', 'Description B', 'Description C'],
            'Custom Collections': ['Engine', 'Engine', 'Brakes', 'Suspension', 'Electrical'],
            'Vendor': ['Steele', 'Steele', 'Steele', 'Steele', 'Steele'],
            'Variant Price': [10.00, 10.00, 20.00, 20.00, 30.00]
        }
        
        # Add all other required columns with default values
        for col in cols_list:
            if col not in test_data:
                test_data[col] = [''] * 5
        
        test_df = pd.DataFrame(test_data)
        
        # Test consolidation
        consolidated_df = transformer.consolidate_products_by_unique_id(test_df)
        
        # Validate results
        assert len(consolidated_df) == 3, f"Should have 3 unique products, got {len(consolidated_df)}"
        
        # Check SKU-001 consolidation (2 rows → 1 row)
        sku_001_rows = consolidated_df[consolidated_df['Variant SKU'] == 'SKU-001']
        assert len(sku_001_rows) == 1, "Should have exactly 1 row for SKU-001"
        
        sku_001_tags = sku_001_rows.iloc[0]['Tags']
        expected_tags = '1930_Ford_Model_A, 1931_Ford_Model_A'
        assert sku_001_tags == expected_tags, f"Tags should be combined: {sku_001_tags}"
        
        # Should use longer description
        sku_001_desc = sku_001_rows.iloc[0]['Body HTML']
        assert 'longer' in sku_001_desc, "Should use longer description"
        
        # Check SKU-002 consolidation (2 rows → 1 row)
        sku_002_rows = consolidated_df[consolidated_df['Variant SKU'] == 'SKU-002']
        assert len(sku_002_rows) == 1, "Should have exactly 1 row for SKU-002"
        
        sku_002_tags = sku_002_rows.iloc[0]['Tags']
        expected_tags_002 = '1965_Ford_Mustang, 1966_Ford_Mustang'
        assert sku_002_tags == expected_tags_002, f"Tags should be combined: {sku_002_tags}"
        
        # Should combine collections
        sku_002_collections = sku_002_rows.iloc[0]['Custom Collections']
        assert 'Brakes' in sku_002_collections, "Should include Brakes collection"
        assert 'Suspension' in sku_002_collections, "Should include Suspension collection"
        
        # Check SKU-003 (single row, unchanged)
        sku_003_rows = consolidated_df[consolidated_df['Variant SKU'] == 'SKU-003']
        assert len(sku_003_rows) == 1, "Should have exactly 1 row for SKU-003"
        assert sku_003_rows.iloc[0]['Tags'] == '1955_Chevy_Bel_Air', "Single product tags should be unchanged"
        
        print("✅ Product consolidation test passed")
        print(f"   Consolidated {len(test_df)} → {len(consolidated_df)} products")
        print(f"   SKU-001 tags: {sku_001_tags}")
        print(f"   SKU-002 tags: {sku_002_tags}")
    
    def test_consolidation_with_real_multi_tags(self):
        """Test consolidation with real multi-tag output from AI system"""
        
        # Create test data that simulates what we get from the AI multi-tag system
        test_data = {
            'Title': ['Axle Rebound Pad', 'Brake Pad Set'],
            'Variant SKU': ['10-0002-35', '10-0004-35'],
            'Tags': [
                '1933_Stutz_Model_DV-32, 1933_Stutz_Model_LAA, 1933_Stutz_Model_SV-16',  # Multi-tag from AI
                '1930_Durant_Model_614, 1930_Durant_Model_617'  # Another multi-tag
            ],
            'Body HTML': ['Axle pad description', 'Brake pad description'],
            'Custom Collections': ['Suspension', 'Brakes'],
            'Vendor': ['Steele', 'Steele'],
            'Variant Price': [127.79, 89.99]
        }
        
        # Add all other required columns
        for col in cols_list:
            if col not in test_data:
                test_data[col] = [''] * 2
        
        test_df = pd.DataFrame(test_data)
        transformer = SteeleDataTransformer(use_ai=False)
        
        # Test consolidation (should not change anything since SKUs are unique)
        consolidated_df = transformer.consolidate_products_by_unique_id(test_df)
        
        # Should have same number of products (no duplicates)
        assert len(consolidated_df) == 2, f"Should have 2 products, got {len(consolidated_df)}"
        
        # Multi-tags should be preserved exactly
        stutz_tags = consolidated_df[consolidated_df['Variant SKU'] == '10-0002-35'].iloc[0]['Tags']
        expected_stutz = '1933_Stutz_Model_DV-32, 1933_Stutz_Model_LAA, 1933_Stutz_Model_SV-16'
        assert stutz_tags == expected_stutz, f"Multi-tags should be preserved: {stutz_tags}"
        
        durant_tags = consolidated_df[consolidated_df['Variant SKU'] == '10-0004-35'].iloc[0]['Tags']
        expected_durant = '1930_Durant_Model_614, 1930_Durant_Model_617'
        assert durant_tags == expected_durant, f"Multi-tags should be preserved: {durant_tags}"
        
        print("✅ Multi-tag consolidation test passed")
        print(f"   Stutz multi-tags preserved: {stutz_tags}")
        print(f"   Durant multi-tags preserved: {durant_tags}") 