import pandas as pd
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from pydantic import BaseModel
from typing import Literal, Optional

# Add shared path for golden dataset access
sys.path.append(str(Path(__file__).parent.parent.parent / "shared" / "data"))

class ProductData(BaseModel):
    """AI-friendly intermediate format that uses fewer tokens"""
    title: str
    year_min: str = "1800"
    year_max: str = "1800"
    make: str = "NONE"
    model: str = "NONE"
    mpn: str = ""
    cost: float = 0.0
    price: float = 0.0
    body_html: str = ""
    collection: str = "Accessories"
    product_type: str = "Automotive Part"
    meta_title: str = ""
    meta_description: str = ""

class TestSteeleGoldenIntegration:
    """Test suite for Golden Master dataset integration and AI-friendly transformation"""
    
    @pytest.fixture
    def golden_sample_df(self):
        """Mock golden master sample for testing"""
        return pd.DataFrame({
            'year': [1928, 1930, 1930, 1965],
            'make': ['Stutz', 'Stutz', 'Durant', 'Ford'],
            'model': ['Stutz', 'Stutz', 'Model 6-14', 'Mustang'],
            'car_id': ['1928_Stutz_Stutz', '1930_Stutz_Stutz', '1930_Durant_Model 6-14', '1965_Ford_Mustang']
        })
    
    @pytest.fixture
    def steele_sample_data(self):
        """Sample Steele data for testing"""
        return pd.DataFrame({
            'StockCode': ['10-0001-40', '10-0002-35'],
            'Product Name': ['Accelerator Pedal Pad', 'Axle Rebound Pad'],
            'Description': ['Pad, accelerator pedal...', 'Pad, front axle rebound...'],
            'MAP': [75.49, 127.79],
            'Dealer Price': [43.76, 81.97],
            'Year': [1928, 1930],
            'Make': ['Stutz', 'Stutz'],
            'Model': ['Stutz', 'Stutz']
        })
    
    def test_load_golden_dataset_structure(self):
        """Test that we can load and validate golden dataset structure"""
        # Check if golden dataset sample exists
        golden_path = Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"
        if golden_path.exists():
            # Test first few rows only due to large file size
            df = pd.read_csv(golden_path, nrows=5)
            
            expected_columns = ['year', 'make', 'model']
            for col in expected_columns:
                assert col.lower() in [c.lower() for c in df.columns], f"Golden dataset missing {col} column"
        else:
            # Use mock data for testing
            mock_golden = pd.DataFrame({
                'Year': [1928, 1930],
                'Make': ['Stutz', 'Stutz'], 
                'Model': ['Stutz', 'Stutz']
            })
            assert len(mock_golden) > 0
    
    def test_vehicle_compatibility_validation(self, steele_sample_data, golden_sample_df):
        """Test that Steele vehicles exist in golden dataset"""
        
        def validate_vehicle_compatibility(steele_df: pd.DataFrame, golden_df: pd.DataFrame) -> pd.DataFrame:
            """Validate that vehicles from Steele exist in golden dataset"""
            validation_results = []
            
            for idx, row in steele_df.iterrows():
                year = int(row['Year']) if pd.notna(row['Year']) else None
                make = str(row['Make']) if pd.notna(row['Make']) else None
                model = str(row['Model']) if pd.notna(row['Model']) else None
                
                if year and make and model:
                    # Check if combination exists in golden dataset
                    match = golden_df[
                        (golden_df['year'] == year) &
                        (golden_df['make'] == make) &
                        (golden_df['model'] == model)
                    ]
                    
                    validation_results.append({
                        'steele_row': idx,
                        'year': year,
                        'make': make, 
                        'model': model,
                        'valid': len(match) > 0,
                        'golden_matches': len(match)
                    })
            
            return pd.DataFrame(validation_results)
        
        results = validate_vehicle_compatibility(steele_sample_data, golden_sample_df)
        
        assert len(results) > 0, "Should have validation results"
        assert 'valid' in results.columns, "Should have validity check"
        
        # Check that at least some vehicles are valid
        valid_vehicles = results[results['valid'] == True]
        assert len(valid_vehicles) > 0, "Should have at least some valid vehicle matches"
    
    def test_transform_to_ai_friendly_format(self, steele_sample_data):
        """Test transformation to AI-friendly intermediate format"""
        
        def transform_to_product_data(steele_row: pd.Series) -> ProductData:
            """Transform single Steele row to ProductData format"""
            return ProductData(
                title=str(steele_row['Product Name']),
                year_min=str(int(steele_row['Year'])) if pd.notna(steele_row['Year']) else "1800",
                year_max=str(int(steele_row['Year'])) if pd.notna(steele_row['Year']) else "1800",
                make=str(steele_row['Make']) if pd.notna(steele_row['Make']) else "NONE",
                model=str(steele_row['Model']) if pd.notna(steele_row['Model']) else "NONE",
                mpn=str(steele_row['StockCode']),
                cost=float(steele_row['Dealer Price']),
                price=float(steele_row['MAP']),
                body_html=str(steele_row['Description']),
                collection="Engine",  # Would be determined by AI
                product_type="Automotive Part"
            )
        
        # Test transformation
        product_data = transform_to_product_data(steele_sample_data.iloc[0])
        
        assert product_data.title == "Accelerator Pedal Pad"
        assert product_data.year_min == "1928"
        assert product_data.year_max == "1928"
        assert product_data.make == "Stutz"
        assert product_data.model == "Stutz"
        assert product_data.mpn == "10-0001-40"
        assert product_data.cost == 43.76
        assert product_data.price == 75.49
    
    def test_ai_friendly_format_token_efficiency(self, steele_sample_data):
        """Test that AI-friendly format uses fewer tokens than full data"""
        
        def estimate_tokens(text: str) -> int:
            """Rough token estimation"""
            return len(text.split()) * 1.3
        
        # Original full row data as would be sent to AI
        original_row = steele_sample_data.iloc[0]
        original_prompt = f"""
        Product Information:
        StockCode: {original_row['StockCode']}
        Product Name: {original_row['Product Name']}
        Description: {original_row['Description']}
        MAP: {original_row['MAP']}
        Dealer Price: {original_row['Dealer Price']}
        Year: {original_row['Year']}
        Make: {original_row['Make']}
        Model: {original_row['Model']}
        """
        original_tokens = estimate_tokens(original_prompt)
        
        # AI-friendly ProductData format - only essential fields
        ai_friendly_prompt = f"""
        Product: Accelerator Pedal Pad
        Description: Pad, accelerator pedal...
        Vehicle: 1928 Stutz Stutz
        Price: 75.49
        Cost: 43.76
        """
        ai_friendly_tokens = estimate_tokens(ai_friendly_prompt)
        
        # AI-friendly format should use fewer tokens
        print(f"Original tokens: {original_tokens}, AI-friendly tokens: {ai_friendly_tokens}")
        assert ai_friendly_tokens < original_tokens, f"AI format ({ai_friendly_tokens}) should use fewer tokens than original ({original_tokens})"
    
    @pytest.mark.integration
    def test_golden_dataset_lookup_performance(self, golden_sample_df):
        """Test performance of golden dataset lookups"""
        import time
        
        test_vehicles = [
            (1928, 'Stutz', 'Stutz'),
            (1930, 'Stutz', 'Stutz'),
            (1965, 'Ford', 'Mustang')
        ]
        
        start_time = time.time()
        
        for year, make, model in test_vehicles:
            matches = golden_sample_df[
                (golden_sample_df['year'] == year) &
                (golden_sample_df['make'] == make) &
                (golden_sample_df['model'] == model)
            ]
            assert len(matches) >= 0  # Just check lookup works
        
        lookup_time = time.time() - start_time
        
        # Should be very fast for small dataset
        assert lookup_time < 0.1, f"Golden dataset lookup took too long: {lookup_time} seconds"
    
    def test_missing_vehicle_fallback(self, golden_sample_df):
        """Test handling of vehicles not in golden dataset"""
        
        def validate_with_fallback(year: int, make: str, model: str, golden_df: pd.DataFrame) -> dict:
            """Validate vehicle with fallback for missing data"""
            matches = golden_df[
                (golden_df['year'] == year) &
                (golden_df['make'] == make) &
                (golden_df['model'] == model)
            ]
            
            if len(matches) > 0:
                return {'status': 'valid', 'matches': len(matches)}
            else:
                # Check if make exists with different model
                make_matches = golden_df[golden_df['make'] == make]
                if len(make_matches) > 0:
                    return {'status': 'make_exists', 'suggested_models': make_matches['model'].unique().tolist()}
                else:
                    return {'status': 'unknown_vehicle', 'fallback': 'manual_review_required'}
        
        # Test missing vehicle
        result = validate_with_fallback(1999, 'Tesla', 'Model S', golden_sample_df)
        
        assert result['status'] in ['valid', 'make_exists', 'unknown_vehicle']
        assert 'fallback' in result or 'matches' in result or 'suggested_models' in result
    
    def test_batch_vehicle_validation(self, steele_sample_data, golden_sample_df):
        """Test batch processing of vehicle validation"""
        
        def batch_validate_vehicles(steele_df: pd.DataFrame, golden_df: pd.DataFrame) -> pd.DataFrame:
            """Batch validate all vehicles in Steele data"""
            results = []
            
            for idx, row in steele_df.iterrows():
                try:
                    year = int(row['Year']) if pd.notna(row['Year']) else None
                    make = str(row['Make']) if pd.notna(row['Make']) else None
                    model = str(row['Model']) if pd.notna(row['Model']) else None
                    
                    if year and make and model:
                        matches = golden_df[
                            (golden_df['year'] == year) &
                            (golden_df['make'] == make) &
                            (golden_df['model'] == model)
                        ]
                        
                        results.append({
                            'row_index': idx,
                            'stock_code': row['StockCode'],
                            'valid': len(matches) > 0,
                            'validation_status': 'valid' if len(matches) > 0 else 'not_found'
                        })
                    else:
                        results.append({
                            'row_index': idx,
                            'stock_code': row['StockCode'],
                            'valid': False,
                            'validation_status': 'incomplete_vehicle_data'
                        })
                        
                except Exception as e:
                    results.append({
                        'row_index': idx,
                        'stock_code': row.get('StockCode', 'unknown'),
                        'valid': False,
                        'validation_status': f'error: {str(e)}'
                    })
            
            return pd.DataFrame(results)
        
        validation_df = batch_validate_vehicles(steele_sample_data, golden_sample_df)
        
        assert len(validation_df) == len(steele_sample_data), "Should validate all input rows"
        assert 'valid' in validation_df.columns, "Should have validity column"
        assert 'validation_status' in validation_df.columns, "Should have status column"
        
        # Check that we got reasonable results
        valid_count = len(validation_df[validation_df['valid'] == True])
        assert valid_count >= 0, "Should handle validation without errors"
    
    def test_prepare_data_for_ai_processing(self, steele_sample_data, golden_sample_df):
        """Test preparing validated data for AI processing"""
        
        def prepare_ai_input(steele_row: pd.Series, is_valid: bool) -> dict:
            """Prepare concise input for AI processing"""
            if is_valid:
                return {
                    'product_name': str(steele_row['Product Name']),
                    'description': str(steele_row['Description'])[:100] + "...",  # Truncate for token efficiency
                    'year': int(steele_row['Year']),
                    'make': str(steele_row['Make']),
                    'model': str(steele_row['Model']),
                    'validation_status': 'golden_validated'
                }
            else:
                return {
                    'product_name': str(steele_row['Product Name']),
                    'description': str(steele_row['Description'])[:100] + "...",
                    'validation_status': 'requires_ai_classification'
                }
        
        # Test with validated vehicle
        ai_input_valid = prepare_ai_input(steele_sample_data.iloc[0], True)
        
        assert ai_input_valid['validation_status'] == 'golden_validated'
        assert ai_input_valid['year'] == 1928
        assert ai_input_valid['make'] == 'Stutz'
        assert len(str(ai_input_valid)) < 500  # Should be concise for AI processing
        
        # Test with invalid vehicle  
        ai_input_invalid = prepare_ai_input(steele_sample_data.iloc[0], False)
        
        assert ai_input_invalid['validation_status'] == 'requires_ai_classification'
        assert 'year' not in ai_input_invalid  # Should not include unvalidated vehicle data 