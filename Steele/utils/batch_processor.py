import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from openai import OpenAI
import logging
from utils.exceptions import BatchProcessingError

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles OpenAI batch processing for large datasets"""
    
    def __init__(self, client: OpenAI, batch_size: int = 100):
        self.client = client
        self.batch_size = batch_size
        self.results = []
        self.errors = []
        
    def create_batch_tasks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create batch tasks for OpenAI batch API
        
        Args:
            df: DataFrame with product information
            
        Returns:
            List[Dict]: Batch tasks
        """
        tasks = []
        
        for index, row in df.iterrows():
            # Create individual task for batch API
            task = {
                "custom_id": f"request-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert automotive parts data extractor."
                        },
                        {
                            "role": "user",
                            "content": self._create_extraction_prompt(row)
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            }
            tasks.append(task)
        
        return tasks
    
    def _create_extraction_prompt(self, row: pd.Series) -> str:
        """Create extraction prompt for batch processing"""
        product_info = row.get('product_info', '')
        
        prompt = f"""
Extract structured automotive product data from: {product_info}

Return ONLY a JSON object with this structure:
{{
    "title": "Product title",
    "year_min": 1965,
    "year_max": 1970,
    "make": "Ford",
    "model": "Mustang",
    "mpn": "Part number",
    "cost": 43.76,
    "price": 75.49,
    "body_html": "<p>HTML description</p>",
    "collection": "Product collection",
    "product_type": "Part type",
    "meta_title": "SEO title",
    "meta_description": "SEO description"
}}
"""
        return prompt
    
    def submit_batch_job(self, tasks: List[Dict]) -> str:
        """
        Submit batch job to OpenAI
        
        Args:
            tasks: List of batch tasks
            
        Returns:
            str: Batch job ID
        """
        try:
            # Create batch file content
            batch_content = "\n".join([json.dumps(task) for task in tasks])
            
            # Create file for batch processing
            file_response = self.client.files.create(
                file=batch_content.encode(),
                purpose="batch"
            )
            
            # Submit batch job
            batch_response = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Submitted batch job {batch_response.id} with {len(tasks)} tasks")
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {str(e)}")
            raise BatchProcessingError(f"Failed to submit batch job: {str(e)}")
    
    def monitor_batch_progress(self, batch_id: str) -> Dict[str, Any]:
        """
        Monitor batch processing progress
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Dict: Progress information
        """
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            progress = {
                'batch_id': batch_id,
                'status': batch.status,
                'created_at': batch.created_at,
                'in_progress_at': getattr(batch, 'in_progress_at', None),
                'completed_at': getattr(batch, 'completed_at', None),
                'failed_at': getattr(batch, 'failed_at', None),
                'cancelled_at': getattr(batch, 'cancelled_at', None),
                'expires_at': getattr(batch, 'expires_at', None)
            }
            
            # Add request counts if available
            if hasattr(batch, 'request_counts'):
                progress.update({
                    'total': batch.request_counts.total,
                    'completed': batch.request_counts.completed,
                    'failed': batch.request_counts.failed
                })
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to monitor batch progress: {str(e)}")
            raise BatchProcessingError(f"Failed to monitor batch progress: {str(e)}")
    
    def wait_for_completion(self, batch_id: str, max_wait_time: int = 3600) -> Dict[str, Any]:
        """
        Wait for batch completion with timeout
        
        Args:
            batch_id: Batch job ID
            max_wait_time: Maximum wait time in seconds
            
        Returns:
            Dict: Final batch status
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.monitor_batch_progress(batch_id)
            status = progress['status']
            
            logger.info(f"Batch {batch_id} status: {status}")
            
            if status in ['completed', 'failed', 'cancelled', 'expired']:
                return progress
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
        
        raise BatchProcessingError(f"Batch {batch_id} did not complete within {max_wait_time} seconds")
    
    def process_batch_results(self, batch_id: str) -> pd.DataFrame:
        """
        Process and parse batch results
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            pd.DataFrame: Processed results
        """
        try:
            # Get batch information
            batch = self.client.batches.retrieve(batch_id)
            
            if batch.status != 'completed':
                raise BatchProcessingError(f"Batch {batch_id} not completed. Status: {batch.status}")
            
            if not batch.output_file_id:
                raise BatchProcessingError(f"No output file for batch {batch_id}")
            
            # Download results file
            results_content = self.client.files.content(batch.output_file_id)
            results_text = results_content.content.decode('utf-8')
            
            # Parse results
            results = []
            for line in results_text.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    results.append(self._parse_batch_result(result))
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Failed to process batch results: {str(e)}")
            raise BatchProcessingError(f"Failed to process batch results: {str(e)}")
    
    def _parse_batch_result(self, result: Dict) -> Dict:
        """Parse individual batch result"""
        try:
            custom_id = result.get('custom_id', 'unknown')
            response = result.get('response', {})
            
            if response.get('status_code') == 200:
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    
                    # Try to parse JSON response
                    try:
                        parsed_data = json.loads(content)
                        return {
                            'custom_id': custom_id,
                            'status': 'success',
                            'response_data': parsed_data
                        }
                    except json.JSONDecodeError:
                        return {
                            'custom_id': custom_id,
                            'status': 'parse_error',
                            'response_data': {'error': 'Invalid JSON response', 'content': content}
                        }
                else:
                    return {
                        'custom_id': custom_id,
                        'status': 'no_choices',
                        'response_data': {'error': 'No choices in response'}
                    }
            else:
                return {
                    'custom_id': custom_id,
                    'status': 'api_error',
                    'response_data': {'error': f"API error: {response.get('status_code')}", 'response': response}
                }
                
        except Exception as e:
            return {
                'custom_id': result.get('custom_id', 'unknown'),
                'status': 'processing_error',
                'response_data': {'error': str(e)}
            }
    
    def get_error_report(self) -> Dict[str, Any]:
        """
        Generate error report for batch processing
        
        Returns:
            Dict: Error report
        """
        total_results = len(self.results)
        error_count = len(self.errors)
        
        return {
            'total_errors': error_count,
            'error_rate': error_count / total_results if total_results > 0 else 0,
            'errors': self.errors,
            'total_processed': total_results
        }
    
    def process_large_dataset(self, df: pd.DataFrame, progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Process large dataset using batch API with progress tracking
        
        Args:
            df: DataFrame to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            pd.DataFrame: Processed results
        """
        try:
            logger.info(f"Starting batch processing for {len(df)} items")
            
            # Split into batches
            batches = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
            all_results = []
            
            for batch_idx, batch_df in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_df)} items)")
                
                # Create tasks
                tasks = self.create_batch_tasks(batch_df)
                
                # Submit batch
                batch_id = self.submit_batch_job(tasks)
                
                # Wait for completion
                final_status = self.wait_for_completion(batch_id)
                
                if final_status['status'] == 'completed':
                    # Process results
                    batch_results = self.process_batch_results(batch_id)
                    all_results.append(batch_results)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress = {
                            'batch': batch_idx + 1,
                            'total_batches': len(batches),
                            'batch_size': len(batch_df),
                            'status': 'completed'
                        }
                        progress_callback(progress)
                else:
                    logger.error(f"Batch {batch_id} failed with status: {final_status['status']}")
                    # Add error entries for failed batch
                    error_results = pd.DataFrame([{
                        'custom_id': f'error-{i}',
                        'status': 'batch_failed',
                        'response_data': {'error': f"Batch processing failed: {final_status['status']}"}
                    } for i in range(len(batch_df))])
                    all_results.append(error_results)
            
            # Combine all results
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                logger.info(f"Batch processing completed. Processed {len(combined_results)} items")
                return combined_results
            else:
                logger.warning("No results from batch processing")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise BatchProcessingError(f"Batch processing failed: {str(e)}")
    
    def estimate_batch_cost(self, df: pd.DataFrame, input_token_estimate: int = 500, output_token_estimate: int = 300) -> Dict[str, float]:
        """
        Estimate cost for batch processing
        
        Args:
            df: DataFrame to process
            input_token_estimate: Estimated input tokens per item
            output_token_estimate: Estimated output tokens per item
            
        Returns:
            Dict: Cost estimation
        """
        total_items = len(df)
        total_input_tokens = total_items * input_token_estimate
        total_output_tokens = total_items * output_token_estimate
        
        # Batch pricing (50% discount)
        input_cost_per_million = 0.075  # $0.075 per 1M input tokens (50% off regular price)
        output_cost_per_million = 0.300  # $0.300 per 1M output tokens (50% off regular price)
        
        input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost
        
        return {
            'total_items': total_items,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'cost_per_item': total_cost / total_items if total_items > 0 else 0,
            'batch_discount': '50%'
        }