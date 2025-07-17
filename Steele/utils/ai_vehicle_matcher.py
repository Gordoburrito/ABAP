import pandas as pd
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelMatchResult(BaseModel):
    """Pydantic model for AI model matching responses"""
    selected_car_ids: List[str]
    confidence: float
    match_type: str = "ai_model_match"

class MatchConfidence(BaseModel):
    """Pydantic model for match confidence scoring"""
    score: float
    reasoning: str

class AIVehicleMatcher:
    """
    AI-powered vehicle matching utilities for Steele data transformation.
    Handles model matching when year+make combinations exist in golden dataset.
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize AI vehicle matcher.
        
        Args:
            use_ai: Whether to use AI for matching (can be disabled for testing)
        """
        self.use_ai = use_ai
        self.client = None
        
        # Make lookup cache for common abbreviations
        self.make_lookup_cache = {
            'amc': 'American Motors',
            'american motors corporation': 'American Motors',
            'american motors corp': 'American Motors',
            'gm': 'General Motors',
            'general motors corporation': 'General Motors',
            'ford motor company': 'Ford',
            'ford motor co': 'Ford',
            'chrysler corporation': 'Chrysler',
            'chrysler corp': 'Chrysler',
            # Add more as needed
        }
        
        # Cost tracking for GPT-4.1-mini
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls_made = 0
        
        # GPT-4.1-mini pricing (per 1M tokens)
        self.input_cost_per_1m = 0.40  # $0.40 per 1M input tokens
        self.output_cost_per_1m = 1.60  # $1.60 per 1M output tokens
        
        if self.use_ai:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print("✅ AI matching enabled with OpenAI API")
            else:
                print("⚠️  OPENAI_API_KEY not found, AI matching disabled")
                self.use_ai = False
        else:
            print("ℹ️  AI matching disabled by configuration")
    
    def ai_match_models_for_year_make(
        self, 
        year_make_matches: pd.DataFrame,
        input_model: str,
        input_submodel: Optional[str] = None,
        input_type: Optional[str] = None,
        input_doors: Optional[str] = None,
        input_body_type: Optional[str] = None,
        confidence_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Use AI to select the best car_ids from year+make matches based on model and other attributes.
        
        Args:
            year_make_matches: DataFrame with car_ids from golden dataset for specific year+make
            input_model: Model string from Steele data
            input_submodel: Submodel string (optional)
            input_type: Type string (optional)
            input_doors: Doors string (optional)
            input_body_type: Body type string (optional)
            confidence_threshold: Minimum confidence to accept AI selection
            
        Returns:
            Filtered DataFrame with AI-selected matches
        """
        if not self.use_ai or self.client is None:
            # NO FALLBACK - Require AI for model matching
            print("❌ AI is required for model matching but not available")
            return pd.DataFrame()
        
        try:
            print(f"🤖 AI MODEL MATCHING DEBUG:")
            print(f"   Input model: '{input_model}'")
            print(f"   Input submodel: '{input_submodel}'")
            print(f"   Input type: '{input_type}'")
            print(f"   Input doors: '{input_doors}'")
            print(f"   Input body_type: '{input_body_type}'")
            print(f"   Year+make matches count: {len(year_make_matches)}")
            
            # Extract unique models and car_ids from year_make_matches
            unique_models = year_make_matches['model'].unique().tolist()
            car_id_model_map = {}
            
            for _, row in year_make_matches.iterrows():
                car_id = row['car_id']
                model = row['model']
                if car_id not in car_id_model_map:
                    car_id_model_map[car_id] = model
            
            print(f"   Available models: {unique_models}")
            print(f"   Car ID to model mapping: {car_id_model_map}")
            
            # Create token-optimized prompt
            prompt = self._create_model_matching_prompt(
                input_model, input_submodel, input_type, input_doors, input_body_type,
                unique_models, car_id_model_map
            )
            
            # Call OpenAI API
            response = self.client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an automotive expert. Select the best matching car_ids based on vehicle specifications."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ModelMatchResult,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Track token usage for cost calculation
            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.api_calls_made += 1
                print(f"   💰 Tokens: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
            
            result = ModelMatchResult.model_validate_json(response.choices[0].message.content)
            
            print(f"   🤖 AI Response:")
            print(f"      Selected car_ids: {result.selected_car_ids}")
            print(f"      Confidence: {result.confidence}")
            print(f"      Match type: {result.match_type}")
            
            # Validate confidence threshold
            if result.confidence < confidence_threshold:
                print(f"⚠️  AI confidence {result.confidence} below threshold {confidence_threshold}")
                print(f"❌ No fallback - AI must handle this case")
                return pd.DataFrame()
            
            # Filter year_make_matches to only selected car_ids
            selected_matches = year_make_matches[year_make_matches['car_id'].isin(result.selected_car_ids)]
            
            if len(selected_matches) == 0:
                print("⚠️  AI selected no valid car_ids")
                print("❌ No fallback - AI must handle this case") 
                return pd.DataFrame()
            
            print(f"✅ AI selected {len(selected_matches)} car_ids with confidence {result.confidence}")
            print(f"   Selected matches: {selected_matches[['car_id', 'model']].to_dict('records')}")
            return selected_matches
            
        except Exception as e:
            print(f"⚠️  AI model matching failed: {e}")
            print(f"❌ No fallback available - AI is required for model matching")
            return pd.DataFrame()
    
    def fuzzy_match_make_for_year(
        self, 
        year_matches: pd.DataFrame,
        input_make: str,
        similarity_threshold: float = 0.6
    ) -> Optional[str]:
        """
        Use fuzzy matching to find the best make from year matches.
        
        Args:
            year_matches: DataFrame with all year matches from golden dataset
            input_make: Make string from Steele data
            similarity_threshold: Minimum similarity score
            
        Returns:
            Corrected make string if found, None otherwise
        """
        print(f"🔍 ENHANCED MAKE MATCHING DEBUG:")
        print(f"   Input make: '{input_make}'")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Year matches count: {len(year_matches)}")
        
        if len(year_matches) == 0 or not input_make:
            print(f"   ❌ Early exit: empty year_matches or input_make")
            return None
        
        # Step 1: Check lookup cache first
        input_make_lower = input_make.lower().strip()
        if input_make_lower in self.make_lookup_cache:
            cached_make = self.make_lookup_cache[input_make_lower]
            print(f"   💾 Found in lookup cache: '{input_make}' -> '{cached_make}'")
            
            # Verify the cached make exists in year_matches
            unique_makes = year_matches['make'].unique()
            if cached_make in unique_makes:
                print(f"   ✅ Cached make '{cached_make}' found in golden dataset")
                return cached_make
            else:
                print(f"   ⚠️  Cached make '{cached_make}' not in golden dataset: {list(unique_makes)}")
        
        # Step 2: Try fuzzy matching
        
        def normalize_make_string(make_str):
            """Normalize make string for comparison"""
            if pd.isna(make_str):
                return ""
            return str(make_str).lower().replace(" ", "").replace("-", "").replace("_", "")
        
        def calculate_make_similarity(str1, str2):
            """Calculate similarity between two make strings"""
            str1_norm = normalize_make_string(str1)
            str2_norm = normalize_make_string(str2)
            
            print(f"         Normalized: '{str1}' -> '{str1_norm}', '{str2}' -> '{str2_norm}'")
            
            # Exact match after normalization
            if str1_norm == str2_norm:
                print(f"         Exact match!")
                return 1.0
            
            # Substring match for makes (improved for partial matches)
            if len(str1_norm) >= 3 and len(str2_norm) >= 3:
                if str1_norm in str2_norm:
                    # Input is contained in golden make (e.g., "american" in "americanmotors")
                    overlap_ratio = len(str1_norm) / len(str2_norm)
                    similarity = 0.8 + (0.1 * overlap_ratio)  # 0.8-0.9 range
                    print(f"         Input contained in golden: {similarity:.3f}")
                    return similarity
                elif str2_norm in str1_norm:
                    # Golden make is contained in input (e.g., "american" in "americanmotors")
                    overlap_ratio = len(str2_norm) / len(str1_norm)
                    similarity = 0.8 + (0.1 * overlap_ratio)  # 0.8-0.9 range
                    print(f"         Golden contained in input: {similarity:.3f}")
                    return similarity
            
            # Word-based matching for multi-word makes
            str1_words = str1.lower().split()
            str2_words = str2.lower().split()
            
            if len(str1_words) > 1 or len(str2_words) > 1:
                word_matches = 0
                total_words = max(len(str1_words), len(str2_words))
                
                for word1 in str1_words:
                    for word2 in str2_words:
                        if word1 == word2 and len(word1) >= 3:  # Significant word match
                            word_matches += 1
                            break
                
                if word_matches > 0:
                    word_similarity = 0.7 + (0.2 * word_matches / total_words)
                    print(f"         Word-based match: {word_similarity:.3f} ({word_matches}/{total_words} words)")
                    return word_similarity
            
            # Character-based similarity (fallback)
            max_len = max(len(str1_norm), len(str2_norm))
            if max_len == 0:
                return 0.0
            
            matches = 0
            i = j = 0
            while i < len(str1_norm) and j < len(str2_norm):
                if str1_norm[i] == str2_norm[j]:
                    matches += 1
                    i += 1
                    j += 1
                else:
                    i += 1
            
            similarity = matches / max_len
            print(f"         Character-based: {similarity:.3f}")
            return similarity if similarity >= 0.5 else 0.0
        
        # Get unique makes from year matches
        unique_makes = year_matches['make'].unique()
        print(f"   Available makes in golden dataset: {list(unique_makes)}")
        
        make_similarities = []
        
        print(f"   🔍 Testing similarities:")
        for golden_make in unique_makes:
            similarity = calculate_make_similarity(input_make, golden_make)
            print(f"      '{input_make}' vs '{golden_make}' = {similarity:.3f}")
            
            if similarity >= similarity_threshold:
                make_similarities.append({
                    'make': golden_make,
                    'similarity': similarity
                })
                print(f"         ✅ Above threshold ({similarity_threshold})")
        
        print(f"   Matches above threshold: {len(make_similarities)}")
        
        # Sort by similarity and return best match
        if make_similarities:
            # Show all matches above threshold
            sorted_matches = sorted(make_similarities, key=lambda x: x['similarity'], reverse=True)
            print(f"   All matches: {[(m['make'], round(m['similarity'], 3)) for m in sorted_matches]}")
            
            best_match = sorted_matches[0]
            print(f"✅ Fuzzy matched make '{input_make}' to '{best_match['make']}' (similarity: {best_match['similarity']})")
            return best_match['make']
        
        print(f"⚠️  No fuzzy make match found for '{input_make}' above threshold {similarity_threshold}")
        
        # Step 3: Try AI make matching if fuzzy fails and AI is available
        if self.use_ai and self.client:
            print(f"   🤖 Trying AI make matching...")
            ai_make = self._ai_match_make(year_matches, input_make)
            if ai_make:
                # Cache the successful AI match for future use
                self.make_lookup_cache[input_make_lower] = ai_make
                print(f"   💾 Cached AI result: '{input_make}' -> '{ai_make}'")
                return ai_make
        
        return None
    
    def _ai_match_make(self, year_matches: pd.DataFrame, input_make: str) -> Optional[str]:
        """
        Use AI to match make when fuzzy matching fails.
        
        Args:
            year_matches: DataFrame with all year matches from golden dataset
            input_make: Make string from Steele data
            
        Returns:
            Corrected make string if found, None otherwise
        """
        try:
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

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an automotive expert specializing in vehicle make identification and abbreviations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            # Track token usage for cost calculation
            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.api_calls_made += 1
                print(f"      💰 Tokens: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
            
            ai_result = response.choices[0].message.content.strip()
            print(f"      🤖 AI suggested: '{ai_result}'")
            
            if ai_result == "NO_MATCH":
                return None
            
            # Verify AI result is in available makes
            if ai_result in unique_makes:
                print(f"      ✅ AI match verified: '{input_make}' -> '{ai_result}'")
                return ai_result
            else:
                print(f"      ❌ AI returned invalid make: '{ai_result}' not in {unique_makes}")
                return None
                
        except Exception as e:
            print(f"      ⚠️  AI make matching failed: {e}")
            return None
    
    def validate_with_corrected_make(
        self,
        golden_df: pd.DataFrame,
        year: int,
        corrected_make: str,
        original_model: str,
        input_submodel: Optional[str] = None,
        input_type: Optional[str] = None,
        input_doors: Optional[str] = None,
        input_body_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Re-run validation with corrected make, potentially leading to AI model matching.
        
        Args:
            golden_df: Golden dataset DataFrame
            year: Vehicle year
            corrected_make: Fuzzy-matched make
            original_model: Original model from Steele data
            input_submodel: Submodel string (optional)
            input_type: Type string (optional)
            input_doors: Doors string (optional)
            input_body_type: Body type string (optional)
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Step 1: Try exact match with corrected make
            exact_matches = golden_df[
                (golden_df['year'] == year) &
                (golden_df['make'] == corrected_make) &
                (golden_df['model'] == original_model)
            ]
            
            if len(exact_matches) > 0:
                return {
                    'golden_validated': True,
                    'golden_matches': len(exact_matches),
                    'car_ids': exact_matches['car_id'].unique().tolist(),
                    'match_type': 'exact_with_corrected_make',
                    'corrected_make': corrected_make
                }
            
            # Step 2: Try year + corrected_make match, then AI model matching
            year_make_matches = golden_df[
                (golden_df['year'] == year) &
                (golden_df['make'] == corrected_make)
            ]
            
            if len(year_make_matches) > 0:
                # Special case: if corrected_make == model, return all matches
                if corrected_make == original_model:
                    return {
                        'golden_validated': True,
                        'golden_matches': len(year_make_matches),
                        'car_ids': year_make_matches['car_id'].unique().tolist(),
                        'match_type': 'make_equals_model_corrected',
                        'corrected_make': corrected_make
                    }
                else:
                    # Use AI model matching with corrected make
                    ai_matches = self.ai_match_models_for_year_make(
                        year_make_matches,
                        original_model,
                        input_submodel,
                        input_type,
                        input_doors,
                        input_body_type
                    )
                    
                    if len(ai_matches) > 0:
                        return {
                            'golden_validated': True,
                            'golden_matches': len(ai_matches),
                            'car_ids': ai_matches['car_id'].unique().tolist(),
                            'match_type': 'ai_model_match_with_corrected_make',
                            'corrected_make': corrected_make
                        }
            
            # No matches found even with corrected make
            return {
                'golden_validated': False,
                'golden_matches': 0,
                'car_ids': [],
                'match_type': 'no_match_with_corrected_make',
                'corrected_make': corrected_make
            }
            
        except Exception as e:
            return {
                'golden_validated': False,
                'golden_matches': 0,
                'car_ids': [],
                'error': f"Re-validation failed: {str(e)}",
                'corrected_make': corrected_make
            }
    
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
        
        # Note: available_models list is used for reference but car_id_model_map is more detailed
        
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
    
    def _fallback_fuzzy_model_match(self, year_make_matches: pd.DataFrame, input_model: str) -> pd.DataFrame:
        """
        Fallback to existing fuzzy matching when AI is unavailable.
        This mirrors the fuzzy_match_car_id logic from the main transformer.
        """
        print(f"🔄 FALLBACK FUZZY MODEL MATCHING:")
        print(f"   Input model: '{input_model}'")
        print(f"   Available matches: {len(year_make_matches)}")
        
        if len(year_make_matches) == 0 or not input_model:
            print(f"   ❌ Early exit: empty data")
            return pd.DataFrame()
        
        def normalize_model_string(model_str):
            if pd.isna(model_str):
                return ""
            return str(model_str).lower().replace(" ", "").replace("-", "").replace("_", "")
        
        def calculate_similarity(str1, str2):
            str1_norm = normalize_model_string(str1)
            str2_norm = normalize_model_string(str2)
            
            print(f"         Comparing: '{str1}' -> '{str1_norm}' vs '{str2}' -> '{str2_norm}'")
            
            if str1_norm == str2_norm:
                print(f"         Exact normalized match!")
                return 1.0
            
            # Handle common model prefixes (Model/Series/Type etc.) - KEY FIX for "Model DC" vs "Series DC"
            def extract_core_identifier(model_str):
                """Extract core identifier after removing common prefixes"""
                import re
                # Split by spaces and non-alphanumeric chars
                parts = re.split(r'[^a-zA-Z0-9]+', model_str.lower())
                parts = [p for p in parts if p]  # Remove empty strings
                
                common_prefixes = ['model', 'series', 'type', 'class', 'version', 'grade']
                
                if len(parts) >= 2 and parts[0] in common_prefixes:
                    return ''.join(parts[1:])  # Everything after the prefix
                return ''.join(parts)
            
            core1 = extract_core_identifier(str1)
            core2 = extract_core_identifier(str2)
            
            print(f"         Core identifiers: '{core1}' vs '{core2}'")
            
            if core1 == core2 and len(core1) >= 2:  # Require at least 2 chars
                print(f"         🎯 CORE MATCH! (Model/Series equivalence)")
                return 0.95  # Very high confidence for core match
            
            # Existing substring matching
            min_len = min(len(str1_norm), len(str2_norm))
            if min_len >= 3 and (str1_norm in str2_norm or str2_norm in str1_norm):
                print(f"         Substring match")
                return 0.9
            
            # Number matching logic
            import re
            numbers1 = re.findall(r'\d+', str1_norm)
            numbers2 = re.findall(r'\d+', str2_norm)
            if numbers1 and numbers2:
                joined1 = ''.join(numbers1)
                joined2 = ''.join(numbers2)
                if joined1 == joined2:
                    print(f"         Number sequence match: {joined1}")
                    return 0.95
            
            # Character-based similarity (fallback)
            max_len = max(len(str1_norm), len(str2_norm))
            if max_len == 0:
                return 0.0
            
            matches = 0
            i = j = 0
            while i < len(str1_norm) and j < len(str2_norm):
                if str1_norm[i] == str2_norm[j]:
                    matches += 1
                    i += 1
                    j += 1
                else:
                    i += 1
            
            similarity = matches / max_len
            print(f"         Character-based: {similarity:.3f}")
            
            # Lowered threshold from 0.5 to 0.4 for more flexibility
            return similarity if similarity >= 0.4 else 0.0
        
        print(f"   🔍 Testing model similarities:")
        similarities = []
        for idx, row in year_make_matches.iterrows():
            golden_model = row['model']
            similarity = calculate_similarity(input_model, golden_model)
            print(f"      '{input_model}' vs '{golden_model}' = {similarity:.3f}")
            
            if similarity >= 0.5:  # Lowered from 0.6 to 0.5 for more flexibility
                similarities.append({
                    'idx': idx,
                    'similarity': similarity,
                    'car_id': row['car_id'],
                    'model': golden_model
                })
                print(f"         ✅ Above threshold (0.5)")
            else:
                print(f"         ❌ Below threshold (0.5)")
        
        print(f"   Model matches above threshold: {len(similarities)}")
        
        if similarities:
            sorted_similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
            print(f"   All model matches: {[(s['model'], round(s['similarity'], 3)) for s in sorted_similarities]}")
            
            best_score = sorted_similarities[0]['similarity']
            best_matches_idx = [s['idx'] for s in similarities if s['similarity'] == best_score]
            result = year_make_matches.loc[best_matches_idx].copy()
            
            print(f"✅ Fuzzy model matching found {len(result)} matches with score {best_score}")
            return result
        else:
            print(f"⚠️  No fuzzy model matches found above threshold 0.5")
            return pd.DataFrame()
    
    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Calculate current cost based on token usage.
        
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
            'total_cost': total_cost
        }
    
    def estimate_batch_cost(self, num_products: int) -> Dict[str, float]:
        """
        Estimate cost for processing a batch of products.
        
        Args:
            num_products: Number of products to process
            
        Returns:
            Dictionary with cost estimates
        """

        # Average tokens per AI call based on our optimized prompts
        avg_input_tokens_model_match = 180  # Optimized prompt
        avg_output_tokens_model_match = 25  # No reasoning, just car_ids
        avg_input_tokens_make_match = 100   # Shorter make matching prompt  
        avg_output_tokens_make_match = 10   # Just make name
        
        # Estimate usage patterns:
        # - 30% of products need model matching (year+make found)
        # - 15% of products need make matching (year found, make fuzzy fails)
        # - 55% exact matches or cached results (no AI calls)
        
        model_matches = int(num_products * 0.30)
        make_matches = int(num_products * 0.15)
        
        total_input_tokens = (
            (model_matches * avg_input_tokens_model_match) +
            (make_matches * avg_input_tokens_make_match)
        )
        
        total_output_tokens = (
            (model_matches * avg_output_tokens_model_match) +
            (make_matches * avg_output_tokens_make_match)
        )
        
        input_cost = (total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (total_output_tokens / 1_000_000) * self.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        api_calls = model_matches + make_matches
        
        return {
            'products': num_products,
            'estimated_api_calls': api_calls,
            'estimated_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'estimated_input_cost': input_cost,
            'estimated_output_cost': output_cost,
            'estimated_total_cost': total_cost,
            'cost_per_product': total_cost / num_products if num_products > 0 else 0
        }
    
    def print_cost_report(self):
        
        """Print a formatted cost report."""

        current = self.get_cost_estimate()
        
        print("💰 AI COST REPORT (GPT-4.1-mini)")
        print("=" * 40)
        print(f"API Calls Made: {current['api_calls']:,}")
        print(f"Input Tokens: {current['input_tokens']:,}")
        print(f"Output Tokens: {current['output_tokens']:,}")
        print(f"Input Cost: ${current['input_cost']:.4f}")
        print(f"Output Cost: ${current['output_cost']:.4f}")
        print(f"Total Cost: ${current['total_cost']:.4f}")
        print("=" * 40)
        
        # Show batch estimates
        batch_sizes = [100, 1000, 10000]
        print("\nBATCH COST ESTIMATES:")
        for size in batch_sizes:
            estimate = self.estimate_batch_cost(size)
            print(f"{size:,} products: ${estimate['estimated_total_cost']:.2f} "
                  f"(${estimate['cost_per_product']:.4f} per product)")
        print()