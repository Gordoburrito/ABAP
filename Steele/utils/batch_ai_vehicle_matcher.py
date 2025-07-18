import pandas as pd
import os
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class ModelMatchResult(BaseModel):
    """Pydantic model for AI model matching responses"""
    selected_car_ids: List[str]
    confidence: float
    match_type: str = "ai_model_match"

class BatchAIVehicleMatcher:
    """
    Batch-enabled AI vehicle matching for Steele data transformation.
    Collects all AI requests and processes them using OpenAI's batch API.
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize batch AI vehicle matcher.
        
        Args:
            use_ai: Whether to use AI for matching
        """
        self.use_ai = use_ai
        self.client = None
        
        # Batch processing state
        self.batch_tasks = []
        self.batch_results = {}
        self.current_batch_id = None
        
        # Cost tracking for GPT-4o (batch API uses different pricing)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls_made = 0
        
        # GPT-4o batch pricing (per 1M tokens) - 50% discount
        self.input_cost_per_1m = 2.50  # $2.50 per 1M input tokens (50% off)
        self.output_cost_per_1m = 10.00  # $10.00 per 1M output tokens (50% off)
        
        if self.use_ai:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print("✅ Batch AI matching enabled with OpenAI API")
            else:
                print("⚠️  OPENAI_API_KEY not found, AI matching disabled")
                self.use_ai = False
        else:
            print("ℹ️  AI matching disabled by configuration")
    
    def add_model_matching_task(
        self, 
        task_id: str,
        year_make_matches: pd.DataFrame,
        input_model: str,
        input_submodel: Optional[str] = None,
        input_type: Optional[str] = None,
        input_doors: Optional[str] = None,
        input_body_type: Optional[str] = None
    ) -> None:
        """
        Add a model matching task to the batch queue.
        
        Args:
            task_id: Unique identifier for this task
            year_make_matches: DataFrame with car_ids from golden dataset
            input_model: Model string from Steele data
            input_submodel: Submodel string (optional)
            input_type: Type string (optional)
            input_doors: Doors string (optional)
            input_body_type: Body type string (optional)
        """
        if not self.use_ai or self.client is None:
            return
        
        # Extract unique models and car_ids from year_make_matches
        unique_models = year_make_matches['model'].unique().tolist()
        car_id_model_map = {}
        
        for _, row in year_make_matches.iterrows():
            car_id = row['car_id']
            model = row['model']
            if car_id not in car_id_model_map:
                car_id_model_map[car_id] = model
        
        # Create prompt
        prompt = self._create_model_matching_prompt(
            input_model, input_submodel, input_type, input_doors, input_body_type,
            unique_models, car_id_model_map
        )
        
        # Create batch task
        task = {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are an automotive expert. Select the best matching car_ids based on vehicle specifications. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        
        self.batch_tasks.append(task)
        print(f"📝 Added model matching task: {task_id}")
    
    def add_make_matching_task(
        self,
        task_id: str,
        year_matches: pd.DataFrame,
        input_make: str
    ) -> None:
        """
        Add a make matching task to the batch queue.
        
        Args:
            task_id: Unique identifier for this task
            year_matches: DataFrame with all year matches from golden dataset
            input_make: Make string from Steele data
        """
        if not self.use_ai or self.client is None:
            return
        
        unique_makes = year_matches['make'].unique().tolist()
        
        prompt = f"""Match the input automotive make to the best option from the available makes.

INPUT MAKE: "{input_make}"

AVAILABLE MAKES:
{chr(10).join([f"- {make}" for make in unique_makes])}

TASK:
1. Find the best matching make from the available options
2. Consider common abbreviations (e.g., AMC = American Motors)
3. Consider partial matches and alternative names
4. Return the exact make name from the available list

RULES:
- Must return exactly one make from the available list
- If no good match exists, return "NO_MATCH"
- Consider automotive industry abbreviations and alternate names

RESPONSE: Just the make name or "NO_MATCH" (no explanation needed)"""

        # Create batch task
        task = {
            "custom_id": task_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 50,
                "messages": [
                    {"role": "system", "content": "You are an automotive expert specializing in vehicle make identification and abbreviations."},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        
        self.batch_tasks.append(task)
        print(f"📝 Added make matching task: {task_id}")
    
    def process_batch(self, batch_name: str = "steele_vehicle_matching") -> str:
        """
        Submit all queued tasks to OpenAI's batch API.
        
        Args:
            batch_name: Name prefix for the batch files
            
        Returns:
            Batch job ID for tracking
        """
        if not self.batch_tasks:
            print("⚠️  No batch tasks to process")
            return None
        
        if not self.use_ai or self.client is None:
            print("❌ AI is not available for batch processing")
            return None
        
        print(f"🚀 Processing batch with {len(self.batch_tasks)} tasks...")
        
        # Ensure data directory exists
        data_dir = Path("/Users/gordonlewis/ABAP/Steele/data/batch")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSONL file with all tasks
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = data_dir / f"{batch_name}_{timestamp}.jsonl"
        
        with open(file_name, "w") as file:
            for task in self.batch_tasks:
                file.write(json.dumps(task) + "\n")
        
        print(f"💾 Saved {len(self.batch_tasks)} tasks to: {file_name}")
        
        try:
            # Upload file for batch processing
            print("📤 Uploading batch file to OpenAI...")
            batch_file = self.client.files.create(
                file=open(file_name, "rb"), 
                purpose="batch"
            )
            
            # Create batch job
            print("🔄 Creating batch job...")
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            self.current_batch_id = batch_job.id
            
            print(f"✅ Batch job created: {batch_job.id}")
            print(f"📊 Status: {batch_job.status}")
            print(f"⏳ Completion window: 24 hours")
            print(f"💰 Expected 50% cost savings with batch API")
            
            return batch_job.id
            
        except Exception as e:
            print(f"❌ Failed to create batch job: {e}")
            return None
    
    def check_batch_status(self, batch_id: str) -> str:
        """
        Check the status of a batch job.
        
        Args:
            batch_id: ID of the batch job to check
            
        Returns:
            Current status of the batch job
        """
        if not self.use_ai or self.client is None:
            return "ai_disabled"
        
        try:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status
            
            print(f"📊 Batch {batch_id} status: {status}")
            
            if hasattr(batch_job, 'request_counts') and batch_job.request_counts:
                counts = batch_job.request_counts
                print(f"   📈 Requests - Total: {counts.total}, Completed: {counts.completed}, Failed: {counts.failed}")
            
            return status
            
        except Exception as e:
            print(f"❌ Failed to check batch status: {e}")
            return "error"
    
    def retrieve_batch_results(self, batch_id: str) -> bool:
        """
        Retrieve and process results from a completed batch job.
        
        Args:
            batch_id: ID of the completed batch job
            
        Returns:
            True if results were successfully retrieved and processed
        """
        if not self.use_ai or self.client is None:
            return False
        
        try:
            print(f"📥 Retrieving results for batch: {batch_id}")
            
            # Get batch job details
            batch_job = self.client.batches.retrieve(batch_id)
            
            if batch_job.status != "completed":
                print(f"❌ Batch not completed yet. Status: {batch_job.status}")
                return False
            
            # Get results file
            if not batch_job.output_file_id:
                print("❌ No output file available")
                return False
            
            result_content = self.client.files.content(batch_job.output_file_id).content
            
            # Save results file
            data_dir = Path("/Users/gordonlewis/ABAP/Steele/data/batch")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file_name = data_dir / f"batch_results_{timestamp}.jsonl"
            
            with open(result_file_name, "wb") as file:
                file.write(result_content)
            
            print(f"💾 Results saved to: {result_file_name}")
            
            # Process results
            processed_count = 0
            error_count = 0
            
            with open(result_file_name, "r") as file:
                for line_number, line in enumerate(file, 1):
                    try:
                        result = json.loads(line.strip())
                        task_id = result["custom_id"]
                        
                        # Check if there's an error field and it's not None
                        if result.get("error") is not None:
                            print(f"❌ Task {task_id} failed: {result['error']}")
                            error_count += 1
                            continue
                        
                        # Check if response exists and has successful status
                        if "response" not in result:
                            print(f"❌ Task {task_id} failed: No response")
                            error_count += 1
                            continue
                        
                        if result["response"]["status_code"] != 200:
                            print(f"❌ Task {task_id} failed: Status {result['response']['status_code']}")
                            error_count += 1
                            continue
                        
                        # Extract response content
                        try:
                            response_content = result["response"]["body"]["choices"][0]["message"]["content"]
                        except (KeyError, IndexError) as e:
                            print(f"❌ Task {task_id} failed: Invalid response structure - {e}")
                            error_count += 1
                            continue
                        
                        # Store result for later retrieval
                        self.batch_results[task_id] = {
                            "content": response_content,
                            "usage": result["response"]["body"].get("usage", {})
                        }
                        
                        # Track token usage
                        if "usage" in result["response"]["body"]:
                            usage = result["response"]["body"]["usage"]
                            self.total_input_tokens += usage.get("prompt_tokens", 0)
                            self.total_output_tokens += usage.get("completion_tokens", 0)
                            self.api_calls_made += 1
                        
                        processed_count += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing result at line {line_number}: {e}")
                        error_count += 1
                        continue
            
            print(f"✅ Processed {processed_count} results successfully")
            if error_count > 0:
                print(f"⚠️  {error_count} results had errors")
            
            # Show cost summary
            self.print_cost_report()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to retrieve batch results: {e}")
            return False
    
    def get_model_match_result(self, task_id: str, year_make_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Get the processed result for a model matching task.
        
        Args:
            task_id: ID of the task to get results for
            year_make_matches: Original DataFrame to filter based on AI results
            
        Returns:
            Filtered DataFrame with AI-selected matches
        """
        if task_id not in self.batch_results:
            print(f"❌ No result found for task: {task_id}")
            return pd.DataFrame()
        
        try:
            content = self.batch_results[task_id]["content"]
            
            # Parse JSON response
            try:
                result_data = json.loads(content)
                result = ModelMatchResult(**result_data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse model match result for {task_id}: {e}")
                return pd.DataFrame()
            
            print(f"🤖 AI Result for {task_id}:")
            print(f"   Selected car_ids: {result.selected_car_ids}")
            print(f"   Confidence: {result.confidence}")
            
            # 🔧 ADD DEBUG CODE FOR EMPTY RESULTS
            if len(result.selected_car_ids) == 0:
                print(f"   ⚠️  EMPTY RESULT - Debugging...")
                print(f"   Available car_ids in year_make_matches:")
                for idx, row in year_make_matches.iterrows():
                    print(f"      • {row['car_id']}: {row['model']}")
                print(f"   This suggests the AI prompt needs improvement or confidence threshold is too high")
                
                # Show AI response for debugging
                print(f"   Raw AI response: {content}")
            
            # Filter year_make_matches to only selected car_ids
            selected_matches = year_make_matches[year_make_matches['car_id'].isin(result.selected_car_ids)]
            
            if len(selected_matches) == 0 and len(result.selected_car_ids) > 0:
                print(f"⚠️  AI selected car_ids not found in year_make_matches: {result.selected_car_ids}")
                return pd.DataFrame()
            
            if len(selected_matches) == 0:
                print(f"✋ Found 0 matches for {task_id} (AI confidence: {result.confidence})")
                return pd.DataFrame()
            
            print(f"✅ Found {len(selected_matches)} matches for {task_id}")
            return selected_matches
            
        except Exception as e:
            print(f"❌ Error processing model match result for {task_id}: {e}")
            return pd.DataFrame()
    
    def get_make_match_result(self, task_id: str) -> Optional[str]:
        """
        Get the processed result for a make matching task.
        
        Args:
            task_id: ID of the task to get results for
            
        Returns:
            Corrected make string if found, None otherwise
        """
        if task_id not in self.batch_results:
            print(f"❌ No result found for task: {task_id}")
            return None
        
        try:
            content = self.batch_results[task_id]["content"].strip()
            
            print(f"🤖 AI Make Result for {task_id}: '{content}'")
            
            if content == "NO_MATCH":
                return None
            
            return content
            
        except Exception as e:
            print(f"❌ Error processing make match result for {task_id}: {e}")
            return None
    
    def wait_for_completion(self, batch_id: str, max_wait_time: int = 3600) -> bool:
        """
        Wait for a batch job to complete.
        
        Args:
            batch_id: ID of the batch job to wait for
            max_wait_time: Maximum time to wait in seconds (default: 1 hour)
            
        Returns:
            True if batch completed successfully, False otherwise
        """
        if not self.use_ai or self.client is None:
            return False
        
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        print(f"⏳ Waiting for batch {batch_id} to complete...")
        print(f"   Max wait time: {max_wait_time} seconds")
        print(f"   Check interval: {check_interval} seconds")
        
        while time.time() - start_time < max_wait_time:
            status = self.check_batch_status(batch_id)
            
            if status == "completed":
                print("✅ Batch completed successfully!")
                return True
            elif status in ["failed", "expired", "cancelled"]:
                print(f"❌ Batch failed with status: {status}")
                return False
            elif status == "error":
                print("❌ Error checking batch status")
                return False
            
            print(f"   Still processing... (elapsed: {int(time.time() - start_time)}s)")
            time.sleep(check_interval)
        
        print(f"⏰ Timeout after {max_wait_time} seconds")
        return False
    
    def _create_model_matching_prompt(
        self,
        input_model: str,
        input_submodel: Optional[str],
        input_type: Optional[str], 
        input_doors: Optional[str],
        input_body_type: Optional[str],
        available_models: List[str],
        car_id_model_map: Dict[str, str]
    ) -> str:
        """Create optimized prompt for AI model matching"""
        
        # Build input specifications
        input_specs = [f"Model: {input_model}"]
        if input_submodel:
            input_specs.append(f"Submodel: {input_submodel}")
        if input_type:
            input_specs.append(f"Type: {input_type}")
        if input_doors:
            input_specs.append(f"Doors: {input_doors}")
        if input_body_type:
            input_specs.append(f"Body Type: {input_body_type}")
        
        # Build available options
        options_text = []
        for car_id, model in car_id_model_map.items():
            options_text.append(f"- {car_id}: {model}")
        
        prompt = f"""Match vehicle to car_ids. Be VERY flexible with model naming variations.

INPUT: {' | '.join(input_specs)}
OPTIONS: {' | '.join(options_text)}

MATCHING RULES - Be generous:
- "Model DC" = "Series DC" = "Type DC" (same core identifier)
- "Model 6-14" = "Series 614" (number variations)  
- Focus on letters/numbers, ignore prefixes like Model/Series/Type/Class
- Consider body type and doors as secondary factors
- When in doubt, include the match

EXAMPLES:
- Input "Model DC" → Select "Series DC" car_ids (confidence 0.8+)
- Input "Type 350" → Select "Model 350", "Series 350" car_ids
- Input "Version XL" → Select "Class XL", "Grade XL" car_ids

JSON response:
{{
    "selected_car_ids": ["car_id1", "car_id2"],
    "confidence": 0.8,
    "match_type": "ai_model_match"
}}

Rules: Use exact car_ids from options. Confidence 0.5-1.0. Include all reasonable matches."""
        
        return prompt
    
    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Calculate current cost based on token usage (with batch discount).
        
        Returns:
            Dictionary with cost breakdown
        """
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'api_calls': self.api_calls_made,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'discount': '50% (Batch API)'
        }
    
    def print_cost_report(self):
        """Print a formatted cost report."""
        current = self.get_cost_estimate()
        
        print("💰 BATCH AI COST REPORT (GPT-4o with 50% Batch Discount)")
        print("=" * 50)
        print(f"API Calls Made: {current['api_calls']:,}")
        print(f"Input Tokens: {current['input_tokens']:,}")
        print(f"Output Tokens: {current['output_tokens']:,}")
        print(f"Input Cost: ${current['input_cost']:.4f}")
        print(f"Output Cost: ${current['output_cost']:.4f}")
        print(f"Total Cost: ${current['total_cost']:.4f}")
        print(f"Savings: 50% discount vs standard API")
        print("=" * 50)
        
        if current['api_calls'] > 0:
            cost_per_call = current['total_cost'] / current['api_calls']
            print(f"Cost per API call: ${cost_per_call:.4f}")
        print()
    
    def clear_batch_queue(self):
        """Clear all queued batch tasks."""
        self.batch_tasks = []
        print("🗑️  Batch queue cleared")
    
    def get_queue_size(self) -> int:
        """Get the number of tasks in the batch queue."""
        return len(self.batch_tasks)