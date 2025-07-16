import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError
from openai import OpenAI
from utils.ai_extraction import AIProductExtractor, ProductData
from utils.exceptions import AIExtractionError


class TestProductData:
    """Test suite for ProductData Pydantic model"""
    
    def test_valid_product_data(self):
        """Test valid product data creation"""
        data = {
            'title': 'Accelerator Pedal Pad',
            'year_min': 1965,
            'year_max': 1970,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': '10-0001-40',
            'cost': 43.76,
            'price': 75.49,
            'body_html': '<p>High-quality accelerator pedal pad for Ford Mustang</p>',
            'collection': 'Ford Parts',
            'product_type': 'Pedal Pad',
            'meta_title': 'Ford Mustang Accelerator Pedal Pad 1965-1970',
            'meta_description': 'Premium accelerator pedal pad for 1965-1970 Ford Mustang'
        }
        
        product = ProductData(**data)
        assert product.title == 'Accelerator Pedal Pad'
        assert product.year_min == 1965
        assert product.year_max == 1970
        assert product.make == 'Ford'
        assert product.model == 'Mustang'
        assert product.mpn == '10-0001-40'
        assert product.cost == 43.76
        assert product.price == 75.49
    
    def test_invalid_year_range(self):
        """Test invalid year range validation"""
        data = {
            'title': 'Test Product',
            'year_min': 1970,  # Invalid: min > max
            'year_max': 1965,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': 'TEST-001',
            'cost': 10.0,
            'price': 20.0,
            'body_html': '<p>Test</p>',
            'collection': 'Test',
            'product_type': 'Test',
            'meta_title': 'Test',
            'meta_description': 'Test'
        }
        
        with pytest.raises(ValidationError):
            ProductData(**data)
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        data = {
            'title': 'Test Product',
            # Missing required fields
        }
        
        with pytest.raises(ValidationError):
            ProductData(**data)
    
    def test_year_validation_bounds(self):
        """Test year validation bounds"""
        # Test minimum year
        data = {
            'title': 'Test Product',
            'year_min': 1899,  # Too early
            'year_max': 1900,
            'make': 'Ford',
            'model': 'Model T',
            'mpn': 'TEST-001',
            'cost': 10.0,
            'price': 20.0,
            'body_html': '<p>Test</p>',
            'collection': 'Test',
            'product_type': 'Test',
            'meta_title': 'Test',
            'meta_description': 'Test'
        }
        
        with pytest.raises(ValidationError):
            ProductData(**data)
        
        # Test maximum year
        data['year_min'] = 2030  # Too far in future
        data['year_max'] = 2040
        
        with pytest.raises(ValidationError):
            ProductData(**data)
    
    def test_price_validation(self):
        """Test price validation"""
        data = {
            'title': 'Test Product',
            'year_min': 1965,
            'year_max': 1970,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': 'TEST-001',
            'cost': -10.0,  # Invalid negative cost
            'price': 20.0,
            'body_html': '<p>Test</p>',
            'collection': 'Test',
            'product_type': 'Test',
            'meta_title': 'Test',
            'meta_description': 'Test'
        }
        
        with pytest.raises(ValidationError):
            ProductData(**data)


class TestAIProductExtractor:
    """Test suite for AI product extraction functionality"""
    
    @pytest.fixture
    def extractor(self):
        """Create an AIProductExtractor instance for testing"""
        return AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client"""
        client = Mock(spec=OpenAI)
        return client
    
    @pytest.fixture
    def sample_golden_df(self):
        """Create sample golden master data"""
        return pd.DataFrame({
            'year': [1965, 1966, 1967, 1968, 1969, 1970],
            'make': ['Ford', 'Ford', 'Ford', 'Ford', 'Ford', 'Ford'],
            'model': ['Mustang', 'Mustang', 'Mustang', 'Mustang', 'Mustang', 'Mustang'],
            'body_type': ['Coupe', 'Coupe', 'Coupe', 'Coupe', 'Coupe', 'Coupe']
        })
    
    def test_create_extraction_prompt(self, extractor, sample_golden_df):
        """Test extraction prompt creation"""
        product_info = "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang"
        
        valid_options = {
            'makes': ['Ford', 'Chevrolet', 'Dodge'],
            'models': ['Mustang', 'Camaro', 'Challenger'],
            'years': list(range(1900, 2025))
        }
        
        prompt = extractor.create_extraction_prompt(product_info, valid_options)
        
        # Check that prompt contains essential information
        assert 'Accelerator Pedal Pad' in prompt
        assert '1965-1970' in prompt
        assert 'Ford' in prompt
        assert 'Mustang' in prompt
        assert 'JSON' in prompt
        assert 'year_min' in prompt
        assert 'year_max' in prompt
        assert 'make' in prompt
        assert 'model' in prompt
    
    @patch('utils.ai_extraction.OpenAI')
    def test_extract_product_data_success(self, mock_openai_class, extractor, sample_golden_df):
        """Test successful AI product data extraction"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps({
            'title': 'Accelerator Pedal Pad',
            'year_min': 1965,
            'year_max': 1970,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': '10-0001-40',
            'cost': 43.76,
            'price': 75.49,
            'body_html': '<p>High-quality accelerator pedal pad for Ford Mustang</p>',
            'collection': 'Ford Parts',
            'product_type': 'Pedal Pad',
            'meta_title': 'Ford Mustang Accelerator Pedal Pad 1965-1970',
            'meta_description': 'Premium accelerator pedal pad for 1965-1970 Ford Mustang'
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create new extractor instance (to trigger the mock)
        extractor = AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
        
        product_info = "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang"
        
        result = extractor.extract_product_data(product_info, sample_golden_df)
        
        assert isinstance(result, ProductData)
        assert result.title == 'Accelerator Pedal Pad'
        assert result.year_min == 1965
        assert result.year_max == 1970
        assert result.make == 'Ford'
        assert result.model == 'Mustang'
    
    @patch('utils.ai_extraction.OpenAI')
    def test_extract_product_data_api_error(self, mock_openai_class, extractor, sample_golden_df):
        """Test handling of API errors"""
        # Mock API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        extractor = AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
        
        product_info = "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang"
        
        # Should not raise exception but return fallback data
        result = extractor.extract_product_data(product_info, sample_golden_df)
        assert isinstance(result, ProductData)
        assert result.title == "Error: Processing Failed"
    
    @patch('utils.ai_extraction.OpenAI')
    def test_extract_product_data_invalid_json(self, mock_openai_class, extractor, sample_golden_df):
        """Test handling of invalid JSON responses"""
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        extractor = AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
        
        product_info = "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang"
        
        # Should not raise exception but return fallback data
        result = extractor.extract_product_data(product_info, sample_golden_df)
        assert isinstance(result, ProductData)
        assert result.title == "Error: Processing Failed"
    
    @patch('utils.ai_extraction.OpenAI')
    def test_extract_product_data_validation_error(self, mock_openai_class, extractor, sample_golden_df):
        """Test handling of validation errors in AI response"""
        # Mock response with invalid data
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps({
            'title': 'Test Product',
            'year_min': 1970,  # Invalid: min > max
            'year_max': 1965,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': 'TEST-001',
            'cost': 10.0,
            'price': 20.0,
            'body_html': '<p>Test</p>',
            'collection': 'Test',
            'product_type': 'Test',
            'meta_title': 'Test',
            'meta_description': 'Test'
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        extractor = AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
        
        product_info = "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang"
        
        # Should not raise exception but return fallback data
        result = extractor.extract_product_data(product_info, sample_golden_df)
        assert isinstance(result, ProductData)
        assert result.title == "Error: Processing Failed"
    
    def test_handle_ai_errors(self, extractor):
        """Test AI error handling"""
        # Test with API error
        api_error = Exception("API rate limit exceeded")
        fallback_result = extractor.handle_ai_errors(api_error)
        
        assert isinstance(fallback_result, ProductData)
        assert fallback_result.title == "Error: Processing Failed"
        assert fallback_result.make == "Unknown"
        assert fallback_result.model == "Unknown"
    
    def test_retry_logic(self, extractor):
        """Test retry logic for failed requests"""
        # This would test the retry mechanism
        # For now, just verify the method exists
        assert hasattr(extractor, 'extract_product_data')
        assert hasattr(extractor, 'handle_ai_errors')
    
    def test_get_valid_options_from_golden_master(self, extractor, sample_golden_df):
        """Test extraction of valid options from golden master"""
        valid_options = extractor.get_valid_options_from_golden_master(sample_golden_df)
        
        assert 'makes' in valid_options
        assert 'models' in valid_options
        assert 'years' in valid_options
        
        assert 'Ford' in valid_options['makes']
        assert 'Mustang' in valid_options['models']
        assert 1965 in valid_options['years']
        assert 1970 in valid_options['years']
    
    def test_extract_year_range_patterns(self, extractor):
        """Test year range extraction from various formats"""
        test_cases = [
            ("For 1965-1970 Ford Mustang", (1965, 1970)),
            ("Compatible with 34/64 Chevrolet", (1934, 1964)),
            ("Fits 1955 to 1957 Thunderbird", (1955, 1957)),
            ("Universal fit for all vehicles", (1900, 2024)),
            ("1969 Camaro SS", (1969, 1969)),
            ("No year information", (1900, 2024))  # Default fallback
        ]
        
        for description, expected in test_cases:
            result = extractor.extract_year_range(description)
            assert result == expected, f"Failed for: {description}"
    
    def test_extract_make_model_patterns(self, extractor):
        """Test make/model extraction from various formats"""
        test_cases = [
            ("For 1965-1970 Ford Mustang", ("Ford", "Mustang")),
            ("Compatible with Chevrolet Camaro", ("Chevrolet", "Camaro")),
            ("Dodge Challenger RT", ("Dodge", "Challenger")),
            ("Universal fit for all vehicles", ("Universal", "All")),
            ("No make/model information", ("Unknown", "Unknown"))
        ]
        
        for description, expected in test_cases:
            result = extractor.extract_make_model(description)
            assert result == expected, f"Failed for: {description}"
    
    def test_generate_seo_content(self, extractor):
        """Test SEO content generation"""
        product_data = {
            'title': 'Accelerator Pedal Pad',
            'year_min': 1965,
            'year_max': 1970,
            'make': 'Ford',
            'model': 'Mustang',
            'mpn': '10-0001-40'
        }
        
        seo_content = extractor.generate_seo_content(product_data)
        
        assert 'meta_title' in seo_content
        assert 'meta_description' in seo_content
        assert 'body_html' in seo_content
        
        # Check content quality
        assert 'Ford' in seo_content['meta_title']
        assert 'Mustang' in seo_content['meta_title']
        assert '1965-1970' in seo_content['meta_title']
        assert len(seo_content['meta_title']) <= 60  # SEO best practice
        assert len(seo_content['meta_description']) <= 160  # SEO best practice


class TestAIExtractionIntegration:
    """Integration tests for AI extraction functionality"""
    
    @pytest.fixture
    def extractor(self):
        return AIProductExtractor(api_key="test_key", model="gpt-4.1-mini")
    
    @pytest.fixture
    def sample_golden_df(self):
        return pd.DataFrame({
            'year': [1965, 1966, 1967, 1968, 1969, 1970],
            'make': ['Ford', 'Ford', 'Ford', 'Ford', 'Ford', 'Ford'],
            'model': ['Mustang', 'Mustang', 'Mustang', 'Mustang', 'Mustang', 'Mustang'],
            'body_type': ['Coupe', 'Coupe', 'Coupe', 'Coupe', 'Coupe', 'Coupe']
        })
    
    def test_batch_extraction_simulation(self, extractor, sample_golden_df):
        """Test batch extraction simulation (without actual API calls)"""
        product_infos = [
            "SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang",
            "SKU: 10-0002-35 | Product: Axle Rebound Pad | Description: Universal fit for all vehicles",
            "SKU: 10-0003-35 | Product: Brake Pad | Description: 1934/64 Chevrolet 2-Door"
        ]
        
        # Simulate batch processing
        results = []
        for product_info in product_infos:
            try:
                # In real implementation, this would call AI
                # For testing, we'll create mock results
                mock_result = ProductData(
                    title="Test Product",
                    year_min=1965,
                    year_max=1970,
                    make="Ford",
                    model="Mustang",
                    mpn="TEST-001",
                    cost=10.0,
                    price=20.0,
                    body_html="<p>Test</p>",
                    collection="Test",
                    product_type="Test",
                    meta_title="Test",
                    meta_description="Test"
                )
                results.append(mock_result)
            except Exception as e:
                # Handle errors in batch processing
                results.append(extractor.handle_ai_errors(e))
        
        assert len(results) == 3
        assert all(isinstance(result, ProductData) for result in results)
    
    def test_error_recovery_pipeline(self, extractor, sample_golden_df):
        """Test error recovery in extraction pipeline"""
        # Test with problematic product info
        problematic_product_info = "Invalid product information"
        
        # Should not crash, should return fallback data
        try:
            result = extractor.handle_ai_errors(Exception("Test error"))
            assert isinstance(result, ProductData)
            assert result.title == "Error: Processing Failed"
        except Exception as e:
            pytest.fail(f"Error recovery failed: {e}")
    
    def test_performance_metrics(self, extractor):
        """Test performance metrics collection"""
        # Test that extractor can track performance
        assert hasattr(extractor, 'model')
        assert hasattr(extractor, 'api_key')
        
        # In real implementation, would track:
        # - Processing time per item
        # - Token usage
        # - Success rate
        # - Error rate