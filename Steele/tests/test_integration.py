import pytest
import pandas as pd
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
from utils.integration_pipeline import IncompleteDataPipeline
from utils.batch_processor import BatchProcessor
from utils.exceptions import DataProcessingError


class TestIncompleteDataPipeline:
    """Test suite for complete pipeline integration"""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration for testing"""
        return {
            'openai_api_key': 'test_key',
            'golden_master_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"),
            'column_requirements_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"),
            'batch_size': 10,
            'model': 'gpt-4.1-mini',
            'enable_ai': True,
            'max_retries': 3
        }
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create pipeline instance for testing"""
        return IncompleteDataPipeline(pipeline_config)
    
    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for testing"""
        return pd.DataFrame({
            'StockCode': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'Product Name': ['Accelerator Pedal Pad', 'Brake Pad Set', 'Universal Mirror'],
            'Description': ['For 1965-1970 Ford Mustang', 'For 1969-1970 Chevrolet Camaro', 'Universal fit for all vehicles'],
            'MAP': [75.49, 127.79, 45.69],
            'Dealer Price': [43.76, 81.97, 30.87],
            'StockUom': ['EA', 'EA', 'EA'],
            'UPC Code': ['123456789', '987654321', '456789123']
        })
    
    def test_pipeline_initialization(self, pipeline, pipeline_config):
        """Test pipeline initialization"""
        assert pipeline.config == pipeline_config
        assert pipeline.batch_size == 10
        assert pipeline.model == 'gpt-4.1-mini'
        assert pipeline.enable_ai is True
        
        # Check that components are initialized
        assert pipeline.data_loader is not None
        assert pipeline.ai_converter is not None
        assert pipeline.golden_validator is not None
        assert pipeline.shopify_generator is not None
    
    @patch('utils.ai_extraction.OpenAI')
    def test_process_complete_pipeline_success(self, mock_openai_class, pipeline, sample_input_data, tmp_path):
        """Test successful complete pipeline processing"""
        # Mock AI responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''{
            "title": "Accelerator Pedal Pad",
            "year_min": 1965,
            "year_max": 1970,
            "make": "Ford",
            "model": "Mustang",
            "mpn": "10-0001-40",
            "cost": 43.76,
            "price": 75.49,
            "body_html": "<p>High-quality accelerator pedal pad</p>",
            "collection": "Ford Parts",
            "product_type": "Pedal Pad",
            "meta_title": "Ford Mustang Accelerator Pedal Pad 1965-1970",
            "meta_description": "Premium accelerator pedal pad for 1965-1970 Ford Mustang"
        }'''
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create input file
        input_path = tmp_path / "input.csv"
        sample_input_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "output.csv"
        
        # Process pipeline
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        # Check results
        assert isinstance(results, dict)
        assert 'total_items' in results
        assert 'processed_items' in results
        assert 'success_rate' in results
        assert 'processing_time' in results
        assert 'total_cost' in results
        
        # Check output file
        assert output_path.exists()
        output_df = pd.read_csv(output_path)
        assert len(output_df) > 0
        assert len(output_df.columns) == 65  # Shopify format
    
    def test_process_complete_pipeline_file_not_found(self, pipeline):
        """Test pipeline with non-existent input file"""
        with pytest.raises(FileNotFoundError):
            pipeline.process_complete_pipeline("nonexistent.csv", "output.csv")
    
    @patch('utils.ai_extraction.OpenAI')
    def test_estimate_processing_cost(self, mock_openai_class, pipeline, sample_input_data):
        """Test processing cost estimation"""
        cost_estimate = pipeline.estimate_processing_cost(sample_input_data)
        
        assert isinstance(cost_estimate, dict)
        assert 'total_input_tokens' in cost_estimate
        assert 'estimated_output_tokens' in cost_estimate
        assert 'total_cost' in cost_estimate
        assert 'cost_per_item' in cost_estimate
        assert 'model' in cost_estimate
        
        # Check values are reasonable
        assert cost_estimate['total_cost'] >= 0
        assert cost_estimate['cost_per_item'] >= 0
        assert cost_estimate['total_input_tokens'] > 0
    
    def test_generate_processing_report(self, pipeline):
        """Test processing report generation"""
        # Mock processing results
        pipeline.processing_stats = {
            'total_items': 100,
            'processed_items': 95,
            'failed_items': 5,
            'processing_time': 120.5,
            'total_cost': 2.50,
            'average_processing_time': 1.2,
            'success_rate': 0.95
        }
        
        report = pipeline.generate_processing_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'performance_metrics' in report
        assert 'cost_analysis' in report
        assert 'quality_metrics' in report
        
        # Check report content
        assert report['summary']['total_items'] == 100
        assert report['summary']['success_rate'] == 0.95
        assert report['cost_analysis']['total_cost'] == 2.50
    
    @patch('utils.ai_extraction.OpenAI')
    def test_process_batch_with_errors(self, mock_openai_class, pipeline, tmp_path):
        """Test batch processing with some errors"""
        # Create problematic input data
        problematic_data = pd.DataFrame({
            'StockCode': ['GOOD-001', None, 'GOOD-003'],  # Missing stock code
            'Product Name': ['Good Product', 'Bad Product', ''],  # Empty name
            'Description': ['Valid description', None, 'Another valid'],  # Missing description
            'MAP': [10.0, 'invalid', 20.0],  # Invalid price
            'Dealer Price': [5.0, 8.0, None]  # Missing cost
        })
        
        # Mock AI responses (some successful, some failing)
        def mock_ai_response(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = '''{
                "title": "Test Product",
                "year_min": 1965,
                "year_max": 1970,
                "make": "Ford",
                "model": "Mustang",
                "mpn": "TEST-001",
                "cost": 5.0,
                "price": 10.0,
                "body_html": "<p>Test product</p>",
                "collection": "Test Parts",
                "product_type": "Test",
                "meta_title": "Test Product",
                "meta_description": "Test description"
            }'''
            return mock_response
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = mock_ai_response
        mock_openai_class.return_value = mock_client
        
        # Create input file
        input_path = tmp_path / "problematic_input.csv"
        problematic_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "output.csv"
        
        # Process pipeline - should handle errors gracefully
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        # Should process all items with some failures
        assert results['total_items'] == 3
        assert results['success_rate'] < 1.0  # Some failures expected
        assert output_path.exists()
    
    def test_pipeline_performance_metrics(self, pipeline, sample_input_data):
        """Test performance metrics collection"""
        start_time = time.time()
        
        # Simulate processing
        cost_estimate = pipeline.estimate_processing_cost(sample_input_data)
        
        elapsed_time = time.time() - start_time
        
        # Performance should be reasonable
        assert elapsed_time < 5.0  # Should complete quickly for small dataset
        assert cost_estimate['total_cost'] < 1.0  # Should be low cost for 3 items


class TestBatchProcessor:
    """Test suite for batch processing functionality"""
    
    @pytest.fixture
    def batch_processor(self):
        """Create batch processor for testing"""
        mock_client = Mock()
        return BatchProcessor(mock_client, batch_size=5)
    
    @pytest.fixture
    def sample_batch_data(self):
        """Create sample batch data"""
        return pd.DataFrame({
            'product_info': [
                'SKU: 10-0001-40 | Product: Accelerator Pedal Pad | Description: For 1965-1970 Ford Mustang',
                'SKU: 10-0002-35 | Product: Brake Pad Set | Description: For 1969-1970 Chevrolet Camaro',
                'SKU: 10-0003-35 | Product: Universal Mirror | Description: Universal fit for all vehicles'
            ],
            'stock_code': ['10-0001-40', '10-0002-35', '10-0003-35'],
            'cost': [43.76, 81.97, 30.87],
            'price': [75.49, 127.79, 45.69]
        })
    
    def test_create_batch_tasks(self, batch_processor, sample_batch_data):
        """Test creation of batch tasks"""
        tasks = batch_processor.create_batch_tasks(sample_batch_data)
        
        assert isinstance(tasks, list)
        assert len(tasks) == len(sample_batch_data)
        
        # Check task structure
        for task in tasks:
            assert isinstance(task, dict)
            assert 'custom_id' in task
            assert 'method' in task
            assert 'url' in task
            assert 'body' in task
    
    def test_submit_batch_job(self, batch_processor, sample_batch_data):
        """Test batch job submission"""
        tasks = batch_processor.create_batch_tasks(sample_batch_data)
        
        # Mock batch API response
        mock_batch = Mock()
        mock_batch.id = 'batch_123456'
        mock_batch.status = 'validating'
        
        batch_processor.client.batches.create.return_value = mock_batch
        
        batch_id = batch_processor.submit_batch_job(tasks)
        
        assert batch_id == 'batch_123456'
        batch_processor.client.batches.create.assert_called_once()
    
    def test_monitor_batch_progress(self, batch_processor):
        """Test batch progress monitoring"""
        batch_id = 'batch_123456'
        
        # Mock batch status response
        mock_batch = Mock()
        mock_batch.id = batch_id
        mock_batch.status = 'completed'
        mock_batch.request_counts = Mock()
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 10
        mock_batch.request_counts.failed = 0
        
        batch_processor.client.batches.retrieve.return_value = mock_batch
        
        progress = batch_processor.monitor_batch_progress(batch_id)
        
        assert isinstance(progress, dict)
        assert progress['status'] == 'completed'
        assert progress['completed'] == 10
        assert progress['total'] == 10
        assert progress['failed'] == 0
    
    def test_process_batch_results(self, batch_processor, tmp_path):
        """Test processing of batch results"""
        batch_id = 'batch_123456'
        
        # Create mock results file
        results_content = '''{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "{\\"title\\": \\"Test Product\\", \\"make\\": \\"Ford\\", \\"model\\": \\"Mustang\\"}"}}]}}}
{"id": "batch_req_2", "custom_id": "request-2", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "{\\"title\\": \\"Test Product 2\\", \\"make\\": \\"Chevrolet\\", \\"model\\": \\"Camaro\\"}"}}]}}}'''
        
        results_file = tmp_path / "batch_results.jsonl"
        results_file.write_text(results_content)
        
        # Mock batch API response
        mock_batch = Mock()
        mock_batch.output_file_id = 'file_123'
        
        batch_processor.client.batches.retrieve.return_value = mock_batch
        batch_processor.client.files.content.return_value.content = results_content.encode()
        
        results_df = batch_processor.process_batch_results(batch_id)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2
        assert 'response_data' in results_df.columns
    
    def test_get_error_report(self, batch_processor):
        """Test error report generation"""
        # Simulate some errors
        batch_processor.errors = [
            {'index': 0, 'error': 'API timeout', 'product_info': 'Test product 1'},
            {'index': 2, 'error': 'Invalid response', 'product_info': 'Test product 3'}
        ]
        batch_processor.results = ['success'] * 5  # 5 total results
        
        error_report = batch_processor.get_error_report()
        
        assert isinstance(error_report, dict)
        assert error_report['total_errors'] == 2
        assert error_report['error_rate'] == 0.4  # 2/5
        assert len(error_report['errors']) == 2


class TestIntegrationPerformance:
    """Test suite for integration performance and scalability"""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            'openai_api_key': 'test_key',
            'golden_master_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"),
            'column_requirements_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"),
            'batch_size': 50,
            'model': 'gpt-4.1-mini',
            'enable_ai': False,  # Disable AI for performance tests
            'max_retries': 1
        }
    
    def test_large_dataset_processing(self, pipeline_config, tmp_path):
        """Test processing of large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'StockCode': [f'TEST-{i:04d}' for i in range(1000)],
            'Product Name': [f'Test Product {i}' for i in range(1000)],
            'Description': ['Universal fit for all vehicles'] * 1000,
            'MAP': [20.0] * 1000,
            'Dealer Price': [10.0] * 1000,
            'StockUom': ['EA'] * 1000,
            'UPC Code': [f'{i:09d}' for i in range(1000)]
        })
        
        # Create input file
        input_path = tmp_path / "large_input.csv"
        large_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "large_output.csv"
        
        # Create pipeline
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        start_time = time.time()
        
        # Process pipeline
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        elapsed_time = time.time() - start_time
        
        # Performance assertions
        assert elapsed_time < 60.0  # Should complete within 1 minute
        assert results['total_items'] == 1000
        assert results['success_rate'] > 0.95  # High success rate expected
        assert output_path.exists()
        
        # Check output
        output_df = pd.read_csv(output_path)
        assert len(output_df) == 1000
        assert len(output_df.columns) == 65
    
    def test_memory_efficiency(self, pipeline_config):
        """Test memory efficiency with large datasets"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create pipeline and load large dataset simulation
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        # Simulate processing multiple batches
        for i in range(10):
            batch_data = pd.DataFrame({
                'StockCode': [f'BATCH{i}-{j:03d}' for j in range(100)],
                'Product Name': [f'Batch {i} Product {j}' for j in range(100)],
                'Description': ['Test description'] * 100,
                'MAP': [20.0] * 100,
                'Dealer Price': [10.0] * 100
            })
            
            # Estimate cost (lightweight operation)
            pipeline.estimate_processing_cost(batch_data)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100  # Less than 100MB increase
    
    def test_concurrent_processing_simulation(self, pipeline_config):
        """Test simulation of concurrent processing capabilities"""
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        # Create multiple small datasets
        datasets = []
        for i in range(5):
            data = pd.DataFrame({
                'StockCode': [f'CONCURRENT-{i}-{j}' for j in range(20)],
                'Product Name': [f'Concurrent Product {i}-{j}' for j in range(20)],
                'Description': ['Test description'] * 20,
                'MAP': [15.0] * 20,
                'Dealer Price': [8.0] * 20
            })
            datasets.append(data)
        
        # Process all datasets and measure performance
        start_time = time.time()
        
        results = []
        for data in datasets:
            cost_estimate = pipeline.estimate_processing_cost(data)
            results.append(cost_estimate)
        
        elapsed_time = time.time() - start_time
        
        # Should process quickly
        assert elapsed_time < 10.0  # Less than 10 seconds for 5 small datasets
        assert len(results) == 5
        
        # All results should be valid
        for result in results:
            assert result['total_cost'] >= 0
            assert result['cost_per_item'] >= 0


class TestIntegrationErrorHandling:
    """Test suite for integration error handling and recovery"""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            'openai_api_key': 'test_key',
            'golden_master_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "master_ultimate_golden.csv"),
            'column_requirements_path': str(Path(__file__).parent.parent.parent / "shared" / "data" / "product_import" / "product_import-column-requirements.py"),
            'batch_size': 10,
            'model': 'gpt-4.1-mini',
            'enable_ai': True,
            'max_retries': 2
        }
    
    @patch('utils.ai_extraction.OpenAI')
    def test_api_error_recovery(self, mock_openai_class, pipeline_config, tmp_path):
        """Test recovery from API errors"""
        # Mock API errors
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client
        
        # Create test data
        test_data = pd.DataFrame({
            'StockCode': ['ERROR-001'],
            'Product Name': ['Error Test Product'],
            'Description': ['This will cause an API error'],
            'MAP': [25.0],
            'Dealer Price': [15.0]
        })
        
        # Create input file
        input_path = tmp_path / "error_input.csv"
        test_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "error_output.csv"
        
        # Create pipeline
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        # Process pipeline - should handle errors gracefully
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        # Should complete with fallback processing
        assert results['total_items'] == 1
        assert results['success_rate'] < 1.0  # Some failures expected
        assert output_path.exists()
        
        # Output should still have correct format
        output_df = pd.read_csv(output_path)
        assert len(output_df.columns) == 65
    
    def test_malformed_data_handling(self, pipeline_config, tmp_path):
        """Test handling of malformed input data"""
        # Create malformed data
        malformed_data = pd.DataFrame({
            'StockCode': ['GOOD-001', None, '', 'GOOD-004'],
            'Product Name': ['Good Product', '', None, 'Another Good Product'],
            'Description': [None, 'Valid description', '', 'Another valid description'],
            'MAP': ['invalid', 20.0, None, 25.0],
            'Dealer Price': [10.0, 'invalid', 15.0, None]
        })
        
        # Create input file
        input_path = tmp_path / "malformed_input.csv"
        malformed_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "malformed_output.csv"
        
        # Create pipeline
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        # Process pipeline - should handle malformed data gracefully
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        # Should process all rows
        assert results['total_items'] == 4
        assert output_path.exists()
        
        # Output should have correct structure
        output_df = pd.read_csv(output_path)
        assert len(output_df) == 4
        assert len(output_df.columns) == 65
    
    def test_partial_processing_recovery(self, pipeline_config, tmp_path):
        """Test recovery from partial processing failures"""
        # This test simulates a scenario where processing is interrupted
        # and needs to be resumed
        
        # Create test data
        test_data = pd.DataFrame({
            'StockCode': [f'PARTIAL-{i:03d}' for i in range(50)],
            'Product Name': [f'Partial Test Product {i}' for i in range(50)],
            'Description': ['Test description'] * 50,
            'MAP': [20.0] * 50,
            'Dealer Price': [12.0] * 50
        })
        
        # Create input file
        input_path = tmp_path / "partial_input.csv"
        test_data.to_csv(input_path, index=False)
        
        # Create output path
        output_path = tmp_path / "partial_output.csv"
        
        # Create pipeline
        pipeline = IncompleteDataPipeline(pipeline_config)
        
        # Simulate partial processing
        results = pipeline.process_complete_pipeline(str(input_path), str(output_path))
        
        # Should complete successfully
        assert results['total_items'] == 50
        assert output_path.exists()
        
        # Verify output integrity
        output_df = pd.read_csv(output_path)
        assert len(output_df) == 50
        assert not output_df['Title'].isna().all()  # Should have some valid titles