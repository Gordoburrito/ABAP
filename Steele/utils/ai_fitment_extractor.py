import pandas as pd
import openai
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
steele_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class FitmentExtraction(BaseModel):
    """Model for AI-extracted fitment data"""
    years: List[str] = []
    make: str = "UNKNOWN"
    model: str = "UNKNOWN"  # Can be "UNKNOWN", "ALL", or specific model name
    confidence: float = 0.0
    reasoning: str = ""
    error: Optional[str] = None
    vehicle_tags: List[str] = []  # Generated vehicle tags from golden master

class SteeleAIFitmentExtractor:
    """
    AI-powered fitment extractor for Steele automotive parts data.
    Extracts vehicle fitment information from product descriptions.
    Implements two-pass approach: First extract basic fitment, then expand "ALL" values using golden master.
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the extractor with OpenAI API key"""
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        # Load Golden Master data for validation and model expansion
        self.golden_master_df = self.load_golden_master_data()
    
    def load_golden_master_data(self) -> pd.DataFrame:
        """Load Golden Master Ultimate CSV for validation and model expansion"""
        golden_master_path = project_root / "shared" / "data" / "master_ultimate_golden.csv"  # Updated path
        
        if not golden_master_path.exists():
            print(f"Warning: Golden Master not found at {golden_master_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(golden_master_path)
            print(f"Loaded Golden Master with {len(df)} records for validation and model expansion")
            return df
        except Exception as e:
            print(f"Error loading Golden Master: {e}")
            return pd.DataFrame()
    
    def extract_fitment_from_description(self, product_name: str, description: str, stock_code: str = "") -> FitmentExtraction:
        """
        FIRST PASS: Extract vehicle fitment information from product description using AI.
        Now supports returning "ALL" for make/model when appropriate.
        """
        
        # Enhanced prompt to handle "ALL" cases and vague descriptions
        prompt = f"""
You are an automotive parts fitment expert. Extract vehicle fitment information from this product description.

Product: {product_name}
Description: {description}
Stock Code: {stock_code}

CRITICAL YEAR EXTRACTION RULES:
1. ALWAYS extract years EXACTLY as mentioned in the description
2. Convert year ranges to individual years: "(1920-1929)" becomes ["1920", "1921", "1922", "1923", "1924", "1925", "1926", "1927", "1928", "1929"]
3. NEVER guess or change years - use ONLY what's explicitly stated

MAKE EXTRACTION RULES:
4. For MAKE: 
   - If specific manufacturer names are mentioned, use them (e.g., "Stutz", "Ford", "Chrysler")
   - For vague descriptions like "Independent (1920-1929) automobile manufacturers" or "models built by Independent", use "ALL"
   - For generic descriptions like "automobile manufacturers" without specific brands, use "ALL"
   - For "Street Rod or Custom Build", use "ALL"

MODEL EXTRACTION RULES:
5. For MODEL: 
   - If specific models are mentioned, use them
   - If make and years are clear but NO specific models mentioned, use "ALL"
   - For vague descriptions, use "ALL"
   - If nothing is clear, use "UNKNOWN"

CONFIDENCE AND REASONING:
6. CONFIDENCE: 0.0-1.0 based on clarity of fitment info
7. Provide clear REASONING for your extraction

CORRECT EXAMPLES:
- "compatible with models built by Independent (1920 - 1929) automobile manufacturers" → years: ["1920", "1921", "1922", "1923", "1924", "1925", "1926", "1927", "1928", "1929"], make: "ALL", model: "ALL"
- "fits Independent vehicle models produced from 1920 - 1924" → years: ["1920", "1921", "1922", "1923", "1924"], make: "ALL", model: "ALL"
- "fits 1912-1915 Sterns-Knight models" → years: ["1912", "1913", "1914", "1915"], make: "Sterns-Knight", model: "ALL"
- "1938 Rolls-Royce Phantom III" → years: ["1938"], make: "Rolls-Royce", model: "Phantom III"
- "Street Rod or Custom Build project" → years: [], make: "ALL", model: "ALL" (no specific years mentioned)

VINTAGE CAR KNOWLEDGE:
- Independent Motor Car Company: Real manufacturer (1916-1924)
- Sterns-Knight: Real manufacturer (1912-1924)
- Minerva Motors: Belgian luxury car manufacturer (1902-1938)
- Many vintage manufacturers made limited models, so "ALL" is often appropriate

Return JSON format:
{{
    "years": ["year1", "year2", ...],
    "make": "manufacturer_name_or_ALL",
    "model": "model_name_or_ALL_or_UNKNOWN",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of extraction logic"
}}
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o (latest GPT-4 model available)
                messages=[
                    {"role": "system", "content": "You are an expert automotive parts fitment analyzer specializing in vintage and classic cars. Handle vague descriptions by using 'ALL' when appropriate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content.strip()
            
            # Handle potential markdown formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {stock_code}: {e}")
                print(f"Raw response: {response_text}")
                
                # Try to handle multi-extraction responses (like 10-0138-67)
                if "},{" in response_text:
                    print("Attempting to parse as multi-extraction response...")
                    # Take the first extraction only
                    try:
                        first_extraction = response_text.split("},{")[0] + "}"
                        extracted_data = json.loads(first_extraction)
                        print("Successfully parsed first extraction from multi-response")
                    except json.JSONDecodeError:
                        return FitmentExtraction(
                            error=f"JSON parsing failed: {str(e)}",
                            reasoning="Failed to parse AI response as JSON"
                        )
                else:
                    return FitmentExtraction(
                        error=f"JSON parsing failed: {str(e)}",
                        reasoning="Failed to parse AI response as JSON"
                    )
            
            # Validate and create FitmentExtraction object
            return self._validate_extraction(extracted_data)
            
        except Exception as e:
            print(f"Error extracting fitment for {stock_code}: {str(e)}")
            return FitmentExtraction(
                error=str(e),
                reasoning="AI extraction failed due to API error"
            )
    
    def _validate_extraction(self, extracted_data: Dict) -> FitmentExtraction:
        """Validate and clean extracted fitment data"""
        try:
            years = extracted_data.get('years', [])
            make = str(extracted_data.get('make', 'UNKNOWN')).strip().upper()
            model = str(extracted_data.get('model', 'UNKNOWN')).strip()
            confidence = float(extracted_data.get('confidence', 0.0))
            reasoning = str(extracted_data.get('reasoning', ''))
            
            # Validate years
            valid_years = []
            if isinstance(years, list):
                for year in years:
                    try:
                        year_int = int(str(year).strip())
                        if 1900 <= year_int <= 2030:  # Reasonable year range
                            valid_years.append(str(year_int))
                    except (ValueError, TypeError):
                        continue
            
            # Normalize make name
            if make and make != 'UNKNOWN':
                # Keep original casing for proper names, but ensure it's clean
                make = extracted_data.get('make', 'UNKNOWN').strip()
                if not make or make.upper() == 'UNKNOWN':
                    make = 'UNKNOWN'
            
            # Handle model - can be specific model, "ALL", or "UNKNOWN"
            if model and model.upper() not in ['UNKNOWN', 'ALL']:
                # Keep original model name casing
                model = extracted_data.get('model', 'UNKNOWN').strip()
            elif model and model.upper() == 'ALL':
                model = 'ALL'
            else:
                model = 'UNKNOWN'
            
            # Validate confidence
            confidence = max(0.0, min(1.0, confidence))
            
            return FitmentExtraction(
                years=valid_years,
                make=make,
                model=model,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return FitmentExtraction(
                error=f"Validation failed: {str(e)}",
                reasoning="Failed to validate extracted data"
            )
    
    def expand_all_makes(self, years: List[str]) -> List[str]:
        """
        SECOND PASS: Expand "ALL" makes to actual makes available in the golden master for given years.
        """
        try:
            if not self.golden_master_df.empty and years:
                # Convert years to integers for comparison
                year_ints = [int(year) for year in years if year.isdigit()]
                
                if not year_ints:
                    print(f"No valid years found: {years}")
                    return ["ALL"]
                
                # Filter golden master by years
                year_mask = self.golden_master_df['Year'].isin(year_ints)
                makes = set()
                
                if year_mask.any():
                    filtered_df = self.golden_master_df[year_mask]
                    makes = set(filtered_df['Make'].dropna().unique())
                
                if makes:
                    sorted_makes = sorted(list(makes))
                    print(f"Expanded ALL makes for years {years}: {sorted_makes}")
                    return sorted_makes
                else:
                    print(f"No makes found for years {years}")
                    return ["ALL"]
                    
        except Exception as e:
            print(f"Error expanding makes for years {years}: {e}")
            return ["ALL"]

    def expand_all_models(self, make: str, years: List[str]) -> List[str]:
        """
        SECOND PASS: Expand "ALL" models to actual models available in the golden master for given make and years.
        """
        try:
            if not self.golden_master_df.empty and years and make != "UNKNOWN":
                # Convert years to integers for comparison
                year_ints = [int(year) for year in years if year.isdigit()]
                
                if not year_ints:
                    print(f"No valid years found: {years}")
                    return ["ALL"]
                
                # Filter golden master by make and years
                make_mask = self.golden_master_df['Make'] == make
                year_mask = self.golden_master_df['Year'].isin(year_ints)
                models = set()
                
                combined_mask = make_mask & year_mask
                if combined_mask.any():
                    filtered_df = self.golden_master_df[combined_mask]
                    models = set(filtered_df['Model'].dropna().unique())
                
                if models:
                    sorted_models = sorted(list(models))
                    print(f"Expanded {make} models for years {years}: {sorted_models}")
                    return sorted_models
                else:
                    print(f"No models found for {make} in years {years}")
                    return ["ALL"]
                    
        except Exception as e:
            print(f"Error expanding models for {make}: {e}")
            return ["ALL"]

    def expand_fitment_extraction(self, extraction: FitmentExtraction) -> FitmentExtraction:
        """
        SECOND PASS: Expand fitment extraction using golden master data to generate vehicle tags.
        This follows the REM pattern of generating specific vehicle tags like "1964_Plymouth_Belvedere".
        """
        if extraction.error:
            return extraction
            
        expanded_extraction = FitmentExtraction(
            years=extraction.years,
            make=extraction.make,
            model=extraction.model,
            confidence=extraction.confidence,
            reasoning=extraction.reasoning,
            error=extraction.error
        )
        
        # Generate vehicle tags using the golden master
        vehicle_tags = self._generate_vehicle_tags_from_extraction(extraction)
        expanded_extraction.vehicle_tags = vehicle_tags
        
        # Update reasoning with tag generation info
        if vehicle_tags and vehicle_tags != ['0_Unknown_UNKNOWN']:
            expanded_extraction.reasoning += f" | Generated {len(vehicle_tags)} vehicle tags from golden master"
        else:
            expanded_extraction.reasoning += " | Could not generate vehicle tags - using Unknown"
        
        return expanded_extraction

    def process_unknown_skus_batch_with_expansion(self, unknown_skus_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of unknown SKUs with TWO-PASS approach:
        1. Extract basic fitment with "ALL" support
        2. Expand "ALL" values using golden master
        """
        results = []
        total_skus = len(unknown_skus_df)
        
        print(f"\n🚀 Starting TWO-PASS AI fitment extraction for {total_skus} Unknown SKUs...")
        print("=" * 60)
        
        for idx, row in unknown_skus_df.iterrows():
            sku_num = idx + 1
            stock_code = row.get('StockCode', row.get('Variant SKU', ''))
            product_name = row.get('Product Name', row.get('Title', ''))
            description = row.get('Body HTML', row.get('Description', ''))
            
            print(f"\n[{sku_num}/{total_skus}] Processing SKU: {stock_code}")
            print(f"Product: {product_name}")
            print(f"Description: {description[:100]}...")
            
            # FIRST PASS: Extract fitment using AI
            extraction = self.extract_fitment_from_description(
                product_name=product_name,
                description=description,
                stock_code=stock_code
            )
            
            print(f"✅ First pass: Make={extraction.make}, Model={extraction.model}, Years={extraction.years}")
            
            # SECOND PASS: Expand "ALL" values using golden master
            expanded_extraction = self.expand_fitment_extraction(extraction)
            
            if expanded_extraction.make != extraction.make or expanded_extraction.model != extraction.model:
                print(f"🔄 Second pass expanded: Make={expanded_extraction.make}, Model={expanded_extraction.model}")
            
            # Use vehicle tags from expanded extraction
            final_tags = expanded_extraction.vehicle_tags if expanded_extraction.vehicle_tags else ['0_Unknown_UNKNOWN']
            
            # Store results - use ORIGINAL extraction for display, expanded for tags
            result_data = row.to_dict()
            
            # Replace Tags column if it's 0_Unknown_UNKNOWN and we have generated tags
            original_tags = row.get('Tags', '')
            if original_tags == '0_Unknown_UNKNOWN' and final_tags and final_tags != ['0_Unknown_UNKNOWN']:
                result_data['Tags'] = ', '.join(final_tags)
                print(f"✅ Replaced Tags column with {len(final_tags)} generated tags")
            
            result_data.update({
                'ai_extracted_years': ', '.join(extraction.years) if extraction.years else '',
                'ai_extracted_make': extraction.make,
                'ai_extracted_model': extraction.model,
                'ai_confidence': extraction.confidence,
                'ai_reasoning': extraction.reasoning,
                'generated_tags': ', '.join(final_tags) if final_tags else '0_Unknown_UNKNOWN',
                'extraction_error': expanded_extraction.error or extraction.error or ''
            })
            
            results.append(result_data)
            print(f"🏷️  Generated tags: {result_data['generated_tags']}")
        
        print("\n" + "=" * 60)
        print("✅ Two-pass AI fitment extraction complete!")
        
        return pd.DataFrame(results)

    def _generate_vehicle_tags_from_extraction(self, extraction: FitmentExtraction) -> List[str]:
        """
        Generate vehicle tags from fitment extraction using year-by-year Golden Master lookups.
        This follows the REM pattern exactly - for each individual year, check what makes/models 
        are actually available to prevent impossible combinations like "1920_Toyota_Camry".
        """
        if extraction.error or not extraction.years:
            return ['0_Unknown_UNKNOWN']
        
        try:
            # Check if golden master is loaded
            if self.golden_master_df.empty:
                print("Warning: Golden master is empty")
                return ['0_Unknown_UNKNOWN']
            
            # Parse years into integers
            years = []
            for year_str in extraction.years:
                try:
                    years.append(int(year_str))
                except ValueError:
                    continue
            
            if not years:
                return ['0_Unknown_UNKNOWN']
            
            # Generate vehicle tags using year-by-year Golden Master lookups
            vehicle_tags = []
            
            # Process each year individually to ensure valid year/make/model combinations
            for year in years:
                print(f"  🔍 Processing year {year}")
                
                # STEP 1: Determine which makes to process for this specific year
                makes_to_process = []
                
                if extraction.make == "ALL":
                    # Get all makes that actually existed in this specific year
                    year_mask = self.golden_master_df['Year'] == float(year)
                    available_makes = self.golden_master_df[year_mask]['Make'].dropna().unique()
                    makes_to_process = available_makes.tolist()
                    print(f"    📋 Year {year} has {len(makes_to_process)} makes available")
                
                elif extraction.make != "UNKNOWN":
                    # Use specific makes, but verify they existed in this year
                    specified_makes = [make.strip() for make in extraction.make.split(',')]
                    year_mask = self.golden_master_df['Year'] == float(year)
                    available_makes = set(self.golden_master_df[year_mask]['Make'].dropna().unique())
                    
                    for make in specified_makes:
                        if make in available_makes:
                            makes_to_process.append(make)
                        else:
                            print(f"    ⚠️  Make '{make}' not found in {year}, skipping")
                
                else:
                    continue  # Skip UNKNOWN makes
                
                # STEP 2: For each valid make in this year, determine models
                for make in makes_to_process:
                    print(f"    🚗 Processing {year} {make}")
                    
                    # Get available models for this specific year/make combination
                    year_mask = self.golden_master_df['Year'] == float(year)
                    make_mask = self.golden_master_df['Make'] == make
                    available_models = self.golden_master_df[year_mask & make_mask]['Model'].dropna().unique()
                    
                    if len(available_models) == 0:
                        print(f"      ⚠️  No models found for {year} {make}")
                        continue
                    
                    models_to_process = []
                    
                    if extraction.model == "ALL":
                        # Use all models available for this year/make
                        models_to_process = available_models.tolist()
                        print(f"      📋 Using all {len(models_to_process)} models for {year} {make}")
                    
                    elif extraction.model != "UNKNOWN":
                        # Use specific models, but only if they existed for this year/make
                        specified_models = [model.strip() for model in extraction.model.split(',')]
                        available_models_set = set(available_models)
                        
                        for model in specified_models:
                            if model in available_models_set:
                                models_to_process.append(model)
                            else:
                                print(f"      ⚠️  Model '{model}' not found for {year} {make}")
                    
                    else:
                        # Model is UNKNOWN, generate with UNKNOWN  
                        models_to_process = ["UNKNOWN"]
                    
                    # STEP 3: Generate tags for valid year/make/model combinations
                    for model in models_to_process:
                        # Final verification: does this exact combination exist in golden master?
                        if model != "UNKNOWN":
                            year_mask = self.golden_master_df['Year'] == float(year)
                            make_mask = self.golden_master_df['Make'] == make
                            model_mask = self.golden_master_df['Model'] == model
                            
                            if len(self.golden_master_df[year_mask & make_mask & model_mask]) > 0:
                                # Use golden master format: only replace spaces in make, preserve spaces in model
                                tag = f"{year}_{make.replace(' ', '_')}_{model}"
                                vehicle_tags.append(tag)
                            else:
                                print(f"      ❌ Combination {year}/{make}/{model} not found in golden master")
                        else:
                            tag = f"{year}_{make.replace(' ', '_')}_UNKNOWN"
                            vehicle_tags.append(tag)
            
            # Remove duplicates and sort
            vehicle_tags = sorted(list(set(vehicle_tags)))
            
            if vehicle_tags:
                print(f"🏷️  Generated {len(vehicle_tags)} validated vehicle tags from golden master")
                return vehicle_tags
            else:
                print("❌ No valid vehicle tags generated")
                return ['0_Unknown_UNKNOWN']
                
        except Exception as e:
            print(f"Error generating vehicle tags: {e}")
            import traceback
            traceback.print_exc()
            return ['0_Unknown_UNKNOWN']

def load_golden_master_for_validation():
    """Load Golden Master data for validation purposes"""
    golden_master_path = project_root / "shared" / "data" / "master_ultimate_golden.csv"  # Updated path
    
    if golden_master_path.exists():
        return pd.read_csv(golden_master_path)
    else:
        print(f"Golden Master not found at: {golden_master_path}")
        return pd.DataFrame()

def main():
    """Main function to run AI fitment extraction on unknown SKUs"""
    
    # Check for unknown SKUs file
    unknown_skus_file = steele_root / "data" / "samples" / "unknown_skus_sample.csv"
    
    if not unknown_skus_file.exists():
        print(f"❌ Unknown SKUs file not found: {unknown_skus_file}")
        print("Please run extract_unknown_skus.py first to generate the unknown SKUs sample.")
        return
    
    # Load unknown SKUs
    print(f"📖 Loading unknown SKUs from: {unknown_skus_file}")
    unknown_skus_df = pd.read_csv(unknown_skus_file)
    print(f"Found {len(unknown_skus_df)} unknown SKUs to process")
    
    # Initialize AI extractor
    try:
        extractor = SteeleAIFitmentExtractor()
        print("✅ AI Fitment Extractor initialized successfully")
    except ValueError as e:
        print(f"❌ Error initializing extractor: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Process unknown SKUs
    results_df = extractor.process_unknown_skus_batch_with_expansion(unknown_skus_df)
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = steele_root / "data" / "results" / f"ai_extracted_fitment_{timestamp}.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📊 Total records generated: {len(results_df)} (from {unknown_skus_df['StockCode'].nunique()} original SKUs)")
    
    # Show sample of results
    if len(results_df) > 0:
        print("\n📋 SAMPLE RESULTS:")
        print("-" * 80)
        sample_df = results_df[['Stock Code', 'ai_extracted_make', 'ai_extracted_model', 'ai_confidence', 'extraction_error']].head(10)
        print(sample_df.to_string(index=False))

if __name__ == "__main__":
    main() 