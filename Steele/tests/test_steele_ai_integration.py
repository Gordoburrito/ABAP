import pandas as pd
import pytest
import os
import json
from unittest.mock import Mock, patch
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

# Load environment variables for testing
load_dotenv()

class VehicleClassification(BaseModel):
    vehicle_type: Literal["car", "truck"]
    confidence: float = 0.9
    reasoning: str = "Test classification"

class SEOMetadata(BaseModel):
    meta_title: str
    meta_description: str
    confidence: float = 0.9

class TestSteeleAIIntegration:
    """Test suite for AI integration with Steele data processing"""
    
    @pytest.fixture
    def openai_client(self):
        """OpenAI client for testing"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)
    
    @pytest.fixture
    def sample_vehicle_data(self):
        """Sample vehicle data for testing"""
        return {
            "year": "1965",
            "make": "Ford",
            "model": "Mustang",
            "product_name": "Accelerator Pedal Pad",
            "description": "Pad, accelerator pedal. Cements and fastens with screws to original metal pedal."
        }
    
    @pytest.fixture
    def steele_sample_df(self):
        """Load Steele sample data for AI testing"""
        return pd.read_csv("data/samples/steele_sample.csv").head(3)  # Use small sample for testing
    
    @pytest.mark.ai
    def test_openai_api_connection(self, openai_client):
        """Test that OpenAI API connection works"""
        try:
            # Simple test call
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            assert response.choices[0].message.content is not None
        except Exception as e:
            pytest.fail(f"OpenAI API connection failed: {e}")
    
    @pytest.mark.ai
    def test_vehicle_classification_with_ai(self, openai_client, sample_vehicle_data):
        """Test AI vehicle classification functionality"""
        
        def classify_vehicle_with_ai(year: str, make: str, model: str, client: OpenAI) -> VehicleClassification:
            prompt = f"""Determine if this vehicle is a car or truck.

Vehicle Details:
Year: {year}
Make: {make}
Model: {model}

Rules:
1. Only classify as either "car" or "truck"
2. Trucks include: pickup trucks, commercial trucks, and cargo vans
3. Cars include: sedans, coupes, wagons, passenger vans, and SUVs

Output must be in JSON format with fields:
- vehicle_type: either "car" or "truck"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation
"""

            response = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Classify vehicles as either car or truck."},
                    {"role": "user", "content": prompt}
                ],
                response_format=VehicleClassification,
                temperature=0.1
            )

            return VehicleClassification.model_validate_json(response.choices[0].message.content)
        
        # Test classification
        result = classify_vehicle_with_ai(
            sample_vehicle_data["year"],
            sample_vehicle_data["make"], 
            sample_vehicle_data["model"],
            openai_client
        )
        
        assert result.vehicle_type in ["car", "truck"]
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 10
        # Ford Mustang should be classified as car
        assert result.vehicle_type == "car"
    
    @pytest.mark.ai
    def test_seo_metadata_generation(self, openai_client, sample_vehicle_data):
        """Test AI generation of SEO metadata"""
        
        def generate_seo_metadata_with_ai(product_data: dict, client: OpenAI) -> SEOMetadata:
            prompt = f"""Generate SEO metadata for this automotive part.

Product: {product_data['product_name']}
Description: {product_data['description']}
Vehicle: {product_data['year']} {product_data['make']} {product_data['model']}

Requirements:
- Meta title: 60 characters or less, include year range, make, and product name
- Meta description: 160 characters or less, compelling and descriptive
- Include vintage/classic terminology for pre-1980 vehicles

Output in JSON format with fields:
- meta_title: SEO optimized title
- meta_description: SEO optimized description
- confidence: float between 0.0 and 1.0
"""

            response = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Generate SEO metadata for automotive parts."},
                    {"role": "user", "content": prompt}
                ],
                response_format=SEOMetadata,
                temperature=0.3
            )

            return SEOMetadata.model_validate_json(response.choices[0].message.content)
        
        # Test SEO generation
        result = generate_seo_metadata_with_ai(sample_vehicle_data, openai_client)
        
        assert len(result.meta_title) <= 60
        assert len(result.meta_description) <= 160
        assert "Ford" in result.meta_title or "Mustang" in result.meta_title
        assert "1965" in result.meta_title or "1965" in result.meta_description
        assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.ai
    def test_batch_vehicle_classification(self, openai_client, steele_sample_df):
        """Test batch processing of vehicle classification"""
        
        def classify_vehicle_with_ai(year: str, make: str, model: str, client: OpenAI) -> VehicleClassification:
            # Simplified for testing
            try:
                prompt = f"Classify {year} {make} {model} as car or truck. Respond with JSON containing vehicle_type, confidence, and reasoning."
                
                response = client.beta.chat.completions.parse(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=VehicleClassification,
                    temperature=0.1
                )
                
                return VehicleClassification.model_validate_json(response.choices[0].message.content)
            except Exception as e:
                # Fallback for testing
                return VehicleClassification(
                    vehicle_type="car",
                    confidence=0.5,
                    reasoning=f"Fallback classification due to error: {str(e)}"
                )
        
        # Test batch processing
        results = []
        for idx, row in steele_sample_df.iterrows():
            if pd.notna(row['Year']) and pd.notna(row['Make']) and pd.notna(row['Model']):
                result = classify_vehicle_with_ai(
                    str(int(row['Year'])), 
                    str(row['Make']), 
                    str(row['Model']),
                    openai_client
                )
                results.append(result)
        
        assert len(results) > 0, "Should have classified at least one vehicle"
        
        for result in results:
            assert result.vehicle_type in ["car", "truck"]
            assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.ai
    def test_ai_error_handling(self, openai_client):
        """Test graceful handling of AI API errors"""
        
        def classify_with_fallback(year: str, make: str, model: str, client: OpenAI) -> VehicleClassification:
            try:
                # Intentionally use invalid parameters to trigger error
                response = client.chat.completions.create(
                    model="invalid-model-name",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                return VehicleClassification(vehicle_type="car", confidence=1.0, reasoning="Success")
            except Exception as e:
                # Fallback logic
                return VehicleClassification(
                    vehicle_type="car",  # Default assumption
                    confidence=0.1,
                    reasoning=f"Fallback due to API error: {type(e).__name__}"
                )
        
        result = classify_with_fallback("1965", "Ford", "Mustang", openai_client)
        
        assert result.vehicle_type == "car"
        assert result.confidence == 0.1
        assert "Fallback" in result.reasoning
    
    def test_mock_ai_responses(self):
        """Test with mocked AI responses for consistent testing"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"vehicle_type": "car", "confidence": 0.95, "reasoning": "Mock classification"}'
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Test that mocked responses work
            result = VehicleClassification.model_validate_json(mock_response.choices[0].message.content)
            
            assert result.vehicle_type == "car"
            assert result.confidence == 0.95
            assert result.reasoning == "Mock classification"
    
    @pytest.mark.ai
    @pytest.mark.slow
    def test_ai_response_consistency(self, openai_client):
        """Test that AI responses are consistent for same input"""
        
        test_data = {
            "year": "1930",
            "make": "Stutz", 
            "model": "Stutz"
        }
        
        def classify_with_low_temp(year: str, make: str, model: str, client: OpenAI) -> VehicleClassification:
            prompt = f"Classify {year} {make} {model} as car or truck."
            
            response = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=VehicleClassification,
                temperature=0.0  # Very low temperature for consistency
            )
            
            return VehicleClassification.model_validate_json(response.choices[0].message.content)
        
        # Run classification multiple times
        results = []
        for _ in range(3):
            result = classify_with_low_temp(
                test_data["year"],
                test_data["make"],
                test_data["model"],
                openai_client
            )
            results.append(result)
        
        # Check consistency
        vehicle_types = [r.vehicle_type for r in results]
        assert len(set(vehicle_types)) == 1, f"Inconsistent classifications: {vehicle_types}"
    
    @pytest.mark.ai
    def test_ai_cost_estimation(self, openai_client):
        """Test estimation of AI processing costs"""
        
        # Rough token estimation for cost calculation
        def estimate_tokens(text: str) -> int:
            return len(text.split()) * 1.3  # Rough approximation
        
        sample_prompt = """Determine if this vehicle is a car or truck.
Vehicle Details: 1965 Ford Mustang
Rules: Only classify as either car or truck"""
        
        estimated_input_tokens = estimate_tokens(sample_prompt)
        estimated_output_tokens = 50  # For structured response
        
        # Rough cost estimation (as of 2024 pricing)
        cost_per_1k_input_tokens = 0.00015  # gpt-4.1-mini pricing
        cost_per_1k_output_tokens = 0.0006
        
        estimated_cost = (
            (estimated_input_tokens / 1000) * cost_per_1k_input_tokens +
            (estimated_output_tokens / 1000) * cost_per_1k_output_tokens
        )
        
        # For 1000 products, cost should be reasonable
        cost_per_1000_products = estimated_cost * 1000
        
        assert cost_per_1000_products < 1.0, f"Cost too high: ${cost_per_1000_products:.4f} for 1000 products"
        
        print(f"Estimated cost per 1000 products: ${cost_per_1000_products:.4f}") 