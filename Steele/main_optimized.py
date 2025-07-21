from utils.optimized_batch_steele_transformer import OptimizedBatchSteeleTransformer
import time
import os
import sys
import json
import pandas as pd
from pathlib import Path

def main(submit_only=False):
    """
    Ultra-optimized main entry point for Steele data transformation.
    Uses pattern deduplication to eliminate 2+ hour queuing time.
    """
    try:
        # Get the directory where this script is located (Steele directory)
        steele_dir = Path(__file__).parent

        if submit_only:
            print("=" * 80)
            print("🚀 STEELE ULTRA-OPTIMIZED BATCH SUBMISSION")
            print("   Pattern deduplication • 50% batch API savings")
            print("   Submit optimized batch - get results later")
            print("=" * 80)
            
            # Initialize optimized transformer
            transformer = OptimizedBatchSteeleTransformer(use_ai=True)

            print("\n📊 PHASE 1: ULTRA-FAST DATA ANALYSIS")
            # steele_df = transformer.load_sample_data("data/processed/steele_processed_complete.csv")
            steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
            print(f"✅ Loaded {len(steele_df):,} products")

            # Analyze patterns for deduplication
            pattern_groups, unique_patterns = transformer.analyze_steele_patterns(steele_df)
            
            print("\n🔄 PHASE 2: PATTERN OPTIMIZATION")
            pattern_mapping = transformer.create_pattern_mapping(steele_df, pattern_groups)
            
            print("\n🔍 PHASE 3: SMART VALIDATION")
            exact_match_patterns, ai_needed_patterns = transformer.validate_patterns_against_golden(pattern_mapping)
            
            if len(ai_needed_patterns) == 0:
                print("🎉 ALL PATTERNS HAVE EXACT MATCHES - NO AI NEEDED!")
                print("✅ Processing can complete immediately")
                return "no_batch_needed"

            print(f"\n🚀 PHASE 4: OPTIMIZED BATCH CREATION")
            batch_size = transformer.create_optimized_batch_queue(ai_needed_patterns)
            
            print(f"\n⚡ PHASE 5: BATCH SUBMISSION")
            print(f"   🔥 Reduced {len(steele_df):,} products to {batch_size:,} AI calls")
            print(f"   💰 Saved {len(steele_df) - batch_size:,} API calls with deduplication")
            
            batch_id = transformer.batch_ai_matcher.process_batch("steele_ultra_optimized")
            
            if batch_id:
                print(f"\n✅ ULTRA-OPTIMIZED BATCH SUBMITTED!")
                print(f"📋 Batch ID: {batch_id}")
                print(f"⏳ Processing time: Up to 24 hours")
                print(f"💰 Cost savings: 50% batch API + {transformer.optimization_stats['deduplication_factor']:.1f}x deduplication")
                print("")
                print("📝 To retrieve results later:")
                print(f"   python main_optimized.py --retrieve {batch_id}")
                
                # Save batch info with optimization stats
                batch_info_file = steele_dir / "data" / "batch" / f"optimized_batch_info_{batch_id}.txt"
                os.makedirs(batch_info_file.parent, exist_ok=True)
                
                with open(batch_info_file, "w") as f:
                    f.write(f"Batch ID: {batch_id}\n")
                    f.write(f"Submitted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Products: {transformer.optimization_stats['total_products']:,}\n")
                    f.write(f"Unique Patterns: {transformer.optimization_stats['unique_patterns']:,}\n")
                    f.write(f"Deduplication Factor: {transformer.optimization_stats['deduplication_factor']:.1f}x\n")
                    f.write(f"AI Calls: {batch_size:,}\n")
                    f.write(f"AI Calls Saved: {transformer.optimization_stats['ai_calls_saved']:,}\n")
                    f.write(f"Input File: data/processed/steele_processed_complete.csv\n")
                
                print(f"💾 Batch info saved to: {batch_info_file}")
                
                return batch_id
            else:
                print("❌ Failed to submit batch")
                return None
        else:
            print("=" * 80)
            print("🚀 STEELE ULTRA-OPTIMIZED TRANSFORMATION")
            print("   Pattern deduplication • 50% batch API savings")
            print("   Complete processing with optimization")
            print("=" * 80)

            # Initialize optimized transformer
            transformer = OptimizedBatchSteeleTransformer(use_ai=True)

            # Process with ultra-optimization
            print("\n🔍 DEBUG - Starting complete processing pipeline...")
            final_df = transformer.process_ultra_optimized_pipeline("data/processed/steele_processed_complete.csv")
            
            # DEBUG: Check final results from complete pipeline
            print(f"\n🔍 DEBUG - Complete Pipeline Final Results:")
            print(f"   Type: {type(final_df)}")
            
            if hasattr(final_df, 'columns'):
                print(f"   Columns: {list(final_df.columns)}")
                print(f"   Shape: {final_df.shape}")
                if 'car_id' in final_df.columns:
                    car_id_count = final_df['car_id'].notna().sum()
                    print(f"   Products with car_id: {car_id_count:,}")
                    if car_id_count > 0:
                        sample_car_ids = final_df[final_df['car_id'].notna()].head(3)
                        print(f"   Sample car_ids: {list(sample_car_ids['car_id'].values)}")
                else:
                    print(f"   ❌ car_id column NOT FOUND in final results!")
                    
                # Check for any car_id related columns
                car_related_cols = [col for col in final_df.columns if 'car' in col.lower() or 'vehicle' in col.lower()]
                if car_related_cols:
                    print(f"   Car-related columns found: {car_related_cols}")
                    for col in car_related_cols:
                        non_null_count = final_df[col].notna().sum()
                        print(f"     {col}: {non_null_count:,} non-null values")
            elif isinstance(final_df, list):
                print(f"   Length: {len(final_df)}")
                if len(final_df) > 0 and isinstance(final_df[0], dict):
                    print(f"   Sample item keys: {list(final_df[0].keys())}")
                    if 'car_id' in final_df[0]:
                        print(f"   ✅ Sample item has car_id: {final_df[0]['car_id']}")
                    else:
                        print(f"   ❌ Sample item missing car_id!")
            else:
                print(f"   Unknown data structure!")

            # Save results with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = steele_dir / "data" / "transformed" / f"steele_ultra_optimized_{timestamp}.csv"
            os.makedirs(output_file.parent, exist_ok=True)

            # Handle saving based on data type
            if hasattr(final_df, 'to_csv'):
                # It's a DataFrame
                final_df.to_csv(output_file, index=False)
                print(f"\n💾 DataFrame results saved to: {output_file}")
            elif isinstance(final_df, list):
                # It's a list, convert to DataFrame first
                import pandas as pd
                if len(final_df) > 0 and isinstance(final_df[0], dict):
                    df_to_save = pd.DataFrame(final_df)
                    df_to_save.to_csv(output_file, index=False)
                    print(f"\n💾 List converted to DataFrame and saved to: {output_file}")
                else:
                    print(f"\n❌ Cannot save list data - items are not dictionaries!")
                    return None
            else:
                print(f"\n❌ Cannot save data - unknown format: {type(final_df)}")
                return None

            # Display summary
            final_count = len(final_df) if hasattr(final_df, '__len__') else 0
            print("\n" + "=" * 80)
            print("✅ ULTRA-OPTIMIZED TRANSFORMATION COMPLETE")
            print(f"   📊 Products processed: {final_count:,}")
            print(f"   📊 Deduplication: {transformer.optimization_stats['deduplication_factor']:.1f}x")
            print(f"   📊 AI calls saved: {transformer.optimization_stats['ai_calls_saved']:,}")
            print(f"   📊 Output file: {output_file}")
            print(f"   💰 Cost savings: Massive with deduplication + 50% batch API")
            print("=" * 80)

            return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def retrieve_optimized_results(batch_id):
    """
    Retrieve results from a completed optimized batch and finish processing.
    
    Args:
        batch_id: The completed batch ID
    """
    try:
        steele_dir = Path(__file__).parent

        print("=" * 80)
        print("🚀 STEELE ULTRA-OPTIMIZED RESULTS RETRIEVAL")
        print(f"   Retrieving results for batch: {batch_id}")
        print("=" * 80)

        # Initialize optimized transformer
        transformer = OptimizedBatchSteeleTransformer(use_ai=True)
        
        print("\n📥 PHASE 1: BATCH RESULTS RETRIEVAL")
        if not transformer.batch_ai_matcher.retrieve_batch_results(batch_id):
            print("❌ Failed to retrieve batch results")
            return None
            
        # DEBUG: Check what we got from batch retrieval
        print(f"\n🔍 DEBUG - Retrieved Batch Results:")
        if hasattr(transformer.batch_ai_matcher, 'batch_results') and transformer.batch_ai_matcher.batch_results:
            batch_results = transformer.batch_ai_matcher.batch_results
            print(f"   Total batch results: {len(batch_results)}")
            # Show first few results in detail
            sample_results = list(batch_results.items())[:3]
            for i, (key, result) in enumerate(sample_results):
                print(f"   Sample {i+1}:")
                print(f"     Key: {key}")
                print(f"     Result: {result}")
                if isinstance(result, dict) and 'car_id' in result:
                    print(f"     ✅ Has car_id: {result['car_id']}")
                elif isinstance(result, dict):
                    print(f"     Available keys: {list(result.keys())}")
                else:
                    print(f"     Result type: {type(result)}")
        else:
            print("   ❌ No batch results found!")
            print(f"   batch_results attribute exists: {hasattr(transformer.batch_ai_matcher, 'batch_results')}")
            if hasattr(transformer.batch_ai_matcher, 'batch_results'):
                print(f"   batch_results value: {transformer.batch_ai_matcher.batch_results}")

        print("\n📊 PHASE 2: REBUILDING OPTIMIZATION STATE")
        # steele_df = transformer.load_sample_data("data/processed/steele_processed_complete.csv")
        steele_df = transformer.load_sample_data("data/samples/steele_test_1000.csv")
        
        # Rebuild the optimization state
        pattern_groups, unique_patterns = transformer.analyze_steele_patterns(steele_df)
        pattern_mapping = transformer.create_pattern_mapping(steele_df, pattern_groups)
        transformer.save_pattern_mapping()

        exact_match_patterns, ai_needed_patterns = transformer.validate_patterns_against_golden(pattern_mapping)
        
        # Clear the queue since we're using existing results
        transformer.batch_ai_matcher.clear_batch_queue()
        print("🔄 Using existing batch results")
        
        print("\n🔄 PHASE 3: APPLYING OPTIMIZED RESULTS")
        validation_df = transformer.apply_results_to_all_products(steele_df, ai_needed_patterns)
        validated_count = len(validation_df[validation_df['golden_validated'] == True])
        print(f"✅ {validated_count:,}/{len(steele_df):,} products validated")
        
        # DEBUG: Check validation_df structure and car_id presence
        print(f"\n🔍 DEBUG - Validation DF Info:")
        print(f"   Columns: {list(validation_df.columns)}")
        print(f"   Shape: {validation_df.shape}")
        
        # Check for car_ids (plural) column
        if 'car_ids' in validation_df.columns:
            # Count non-empty car_ids
            non_empty_car_ids = validation_df[validation_df['car_ids'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
            print(f"   Products with car_ids: {len(non_empty_car_ids):,}")
            
            if len(non_empty_car_ids) > 0:
                # Show examples of non-empty car_ids
                sample_car_ids = non_empty_car_ids.head(3)
                for idx, row in sample_car_ids.iterrows():
                    print(f"   Sample {idx}: {row['year']} {row['make']} {row['model']} -> car_ids: {row['car_ids']}")
            else:
                print(f"   ❌ All car_ids are empty!")
                # Show a few examples of what we do have
                sample_rows = validation_df.head(3)
                for idx, row in sample_rows.iterrows():
                    print(f"   Sample {idx}: {row['year']} {row['make']} {row['model']} -> car_ids: {row['car_ids']}")
        elif 'car_id' in validation_df.columns:
            car_id_count = validation_df['car_id'].notna().sum()
            print(f"   Products with car_id: {car_id_count:,}")
            # Show a few examples
            with_car_id = validation_df[validation_df['car_id'].notna()].head(3)
            print(f"   Sample car_ids: {list(with_car_id['car_id'].values)}")
        else:
            print(f"   ❌ No car_id or car_ids column found!")
            
        # Check if we have batch results with car_ids
        if hasattr(transformer.batch_ai_matcher, 'batch_results') and transformer.batch_ai_matcher.batch_results:
            print(f"\n🔍 DEBUG - Batch Results Info:")
            batch_results = transformer.batch_ai_matcher.batch_results
            print(f"   Batch results count: {len(batch_results)}")
            # Show sample batch results
            sample_results = list(batch_results.items())[:3]
            for i, (key, result) in enumerate(sample_results):
                print(f"   Sample {i+1}: {key} -> {result}")

        print("\n🔄 PHASE 4: COMPLETING TRANSFORMATION")
        
        # DEBUG: Check what we're passing to transform_to_standard_format
        print(f"\n🔍 DEBUG - Before transform_to_standard_format:")
        print(f"   steele_df shape: {steele_df.shape}")
        print(f"   validation_df shape: {validation_df.shape}")
        
        # Check a few rows that should have car_ids
        if 'car_ids' in validation_df.columns:
            validated_rows = validation_df[validation_df['golden_validated'] == True]
            print(f"   Validated rows: {len(validated_rows)}")
            if len(validated_rows) > 0:
                sample_validated = validated_rows.head(2)
                for idx, row in sample_validated.iterrows():
                    print(f"   Validated sample {idx}: {row['year']} {row['make']} {row['model']} -> car_ids: {row['car_ids']}")
        
        standard_products = transformer.transform_to_standard_format(steele_df, validation_df)
        
        # DEBUG: Check standard_products after transformation
        print(f"\n🔍 DEBUG - Standard Products Info:")
        print(f"   Type: {type(standard_products)}")
        
        if hasattr(standard_products, 'columns'):
            # It's a DataFrame
            print(f"   Columns: {list(standard_products.columns)}")
            print(f"   Shape: {standard_products.shape}")
            if 'car_id' in standard_products.columns:
                car_id_count = standard_products['car_id'].notna().sum()
                print(f"   Products with car_id: {car_id_count:,}")
                sample_car_ids = standard_products[standard_products['car_id'].notna()].head(3)
                print(f"   Sample car_ids: {list(sample_car_ids['car_id'].values)}")
            else:
                print(f"   ❌ car_id column NOT FOUND in standard_products!")
        elif isinstance(standard_products, list):
            # It's a list
            print(f"   Length: {len(standard_products)}")
            if len(standard_products) > 0:
                print(f"   Sample item type: {type(standard_products[0])}")
                
                # Check if any ProductData objects have car_ids
                products_with_car_ids = [p for p in standard_products if hasattr(p, 'car_ids') and len(p.car_ids) > 0]
                print(f"   Products with car_ids: {len(products_with_car_ids)}")
                
                if len(products_with_car_ids) > 0:
                    # Show examples of products WITH car_ids
                    for i, product in enumerate(products_with_car_ids[:3]):
                        print(f"   ✅ Product {i+1} with car_ids: {product.year_min}-{product.year_max} {product.make} {product.model} -> {product.car_ids}")
                else:
                    # Show examples of products WITHOUT car_ids (to see what we're getting)
                    print(f"   ❌ No products have car_ids!")
                    for i, product in enumerate(standard_products[:3]):
                        print(f"   Sample {i+1}: {product.year_min}-{product.year_max} {product.make} {product.model} -> car_ids: {product.car_ids}")
                        if hasattr(product, 'golden_validated'):
                            print(f"     golden_validated: {product.golden_validated}")
        else:
            print(f"   Unknown data structure!")
            
        enhanced_products = transformer.enhance_with_templates(standard_products)
        
        # DEBUG: Check enhanced_products
        print(f"\n🔍 DEBUG - Enhanced Products Info:")
        print(f"   Type: {type(enhanced_products)}")
        
        if hasattr(enhanced_products, 'columns'):
            print(f"   Columns: {list(enhanced_products.columns)}")
            print(f"   Shape: {enhanced_products.shape}")
            if 'car_id' in enhanced_products.columns:
                car_id_count = enhanced_products['car_id'].notna().sum()
                print(f"   Products with car_id: {car_id_count:,}")
            else:
                print(f"   ❌ car_id column NOT FOUND in enhanced_products!")
        elif isinstance(enhanced_products, list):
            print(f"   Length: {len(enhanced_products)}")
            if len(enhanced_products) > 0 and isinstance(enhanced_products[0], dict):
                print(f"   Sample item keys: {list(enhanced_products[0].keys())}")
                if 'car_id' in enhanced_products[0]:
                    print(f"   ✅ Sample item has car_id: {enhanced_products[0]['car_id']}")
                else:
                    print(f"   ❌ Sample item missing car_id!")
            
        final_df = transformer.transform_to_formatted_shopify_import(enhanced_products)
        
        # DEBUG: Check final_df
        print(f"\n🔍 DEBUG - Final DF Info:")
        print(f"   Type: {type(final_df)}")
        
        if hasattr(final_df, 'columns'):
            print(f"   Columns: {list(final_df.columns)}")
            print(f"   Shape: {final_df.shape}")
            if 'car_id' in final_df.columns:
                car_id_count = final_df['car_id'].notna().sum()
                print(f"   Products with car_id: {car_id_count:,}")
                sample_car_ids = final_df[final_df['car_id'].notna()].head(3)
                print(f"   Sample car_ids: {list(sample_car_ids['car_id'].values)}")
            else:
                print(f"   ❌ car_id column NOT FOUND in final_df!")
                
            # Check for any car_id related columns in final_df
            car_related_cols = [col for col in final_df.columns if 'car' in col.lower() or 'vehicle' in col.lower()]
            if car_related_cols:
                print(f"   Car-related columns found: {car_related_cols}")
                for col in car_related_cols:
                    non_null_count = final_df[col].notna().sum()
                    print(f"     {col}: {non_null_count:,} non-null values")
        elif isinstance(final_df, list):
            print(f"   Length: {len(final_df)}")
            if len(final_df) > 0 and isinstance(final_df[0], dict):
                print(f"   Sample item keys: {list(final_df[0].keys())}")
                if 'car_id' in final_df[0]:
                    print(f"   ✅ Sample item has car_id: {final_df[0]['car_id']}")
                else:
                    print(f"   ❌ Sample item missing car_id!")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = steele_dir / "data" / "transformed" / f"steele_ultra_optimized_completed_{timestamp}.csv"
        os.makedirs(output_file.parent, exist_ok=True)

        # Handle saving based on data type
        if hasattr(final_df, 'to_csv'):
            # It's a DataFrame
            final_df.to_csv(output_file, index=False)
            print(f"\n💾 DataFrame saved to: {output_file}")
        elif isinstance(final_df, list):
            # It's a list, convert to DataFrame first
            import pandas as pd
            if len(final_df) > 0 and isinstance(final_df[0], dict):
                df_to_save = pd.DataFrame(final_df)
                df_to_save.to_csv(output_file, index=False)
                print(f"\n💾 List converted to DataFrame and saved to: {output_file}")
            else:
                print(f"\n❌ Cannot save list data - items are not dictionaries!")
                return None
            
        # Display summary
        final_count = len(final_df) if hasattr(final_df, '__len__') else 0
        print("\n" + "=" * 80)
        print("✅ ULTRA-OPTIMIZED PROCESSING COMPLETE")
        print(f"   📋 Batch ID: {batch_id}")
        print(f"   📊 Products processed: {final_count:,}")
        print(f"   📊 Deduplication: {transformer.optimization_stats['deduplication_factor']:.1f}x")
        print(f"   📊 AI calls saved: {transformer.optimization_stats['ai_calls_saved']:,}")
        print(f"   📊 Golden validated: {validated_count:,}")
        print(f"   📊 Output file: {output_file}")
        print(f"   💰 Cost savings: Massive with deduplication + 50% batch API")
        print("=" * 80)

        # Show final cost report
        transformer.batch_ai_matcher.print_cost_report()

        return str(output_file)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def pattern_retrieve_and_process(input_file=None):
    """
    Pattern-based processing using the pattern_car_id_mapping.json file.
    Bypasses AI completely and uses pre-built pattern mapping for tag assignment.
    
    Args:
        input_file: Path to input CSV file (default: data/processed/steele_processed_complete.csv)
    """
    try:
        steele_dir = Path(__file__).parent
        
        # Default input file if not specified
        if input_file is None:
            input_file = "data/processed/steele_processed_complete.csv"
        
        print("=" * 80)
        print("🚀 STEELE PATTERN-BASED PROCESSING")
        print(f"   Using pattern mapping instead of AI")
        print(f"   Input file: {input_file}")
        print("=" * 80)

        # Load pattern mapping
        pattern_mapping_path = steele_dir / "data" / "pattern_car_id_mapping.json"
        if not pattern_mapping_path.exists():
            print(f"❌ Pattern mapping file not found: {pattern_mapping_path}")
            print("   Run the AI batch process first to generate pattern mapping")
            return None
            
        print("\n📁 PHASE 1: LOADING PATTERN MAPPING")
        with open(pattern_mapping_path, 'r') as f:
            pattern_mapping = json.load(f)
        print(f"✅ Loaded {len(pattern_mapping)} pattern mappings")
        
        print("\n📊 PHASE 2: LOADING INPUT DATA")
        input_path = steele_dir / input_file
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return None
            
        df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df)} rows from {input_file}")
        
        print("\n🔍 PHASE 3: PATTERN MATCHING AND TAG ASSIGNMENT")
        # Create year_make_model keys
        def create_pattern_key(row):
            year = str(int(row['Year'])) if pd.notna(row['Year']) else "Unknown"
            make = str(row['Make']).strip() if pd.notna(row['Make']) else "Unknown"
            model = str(row['Model']).strip() if pd.notna(row['Model']) else "Unknown"
            return f"{year}_{make}_{model}"
        
        df['pattern_key'] = df.apply(create_pattern_key, axis=1)
        
        # Look up tags from pattern mapping
        def lookup_tags(pattern_key):
            return pattern_mapping.get(pattern_key, [])
        
        df['car_ids'] = df['pattern_key'].apply(lookup_tags)
        
        # Pattern matching statistics
        patterns_with_tags = df[df['car_ids'].apply(lambda x: len(x) > 0)]['pattern_key'].nunique()
        total_patterns = df['pattern_key'].nunique()
        rows_with_tags = len(df[df['car_ids'].apply(lambda x: len(x) > 0)])
        
        print(f"✅ Pattern matching results:")
        print(f"   • Patterns with tags: {patterns_with_tags}/{total_patterns}")
        print(f"   • Rows with tags: {rows_with_tags:,}/{len(df):,}")
        print(f"   • Match rate: {rows_with_tags/len(df)*100:.1f}%")
        
        print("\n🔄 PHASE 4: PRODUCT CONSOLIDATION BY SKU")
        # Group by StockCode and consolidate
        consolidated_products = {}
        
        for stock_code, group in df.groupby('StockCode'):
            first_row = group.iloc[0]
            
            # Consolidate all car_ids for this SKU
            all_car_ids = set()
            for _, row in group.iterrows():
                if isinstance(row['car_ids'], list):
                    all_car_ids.update(row['car_ids'])
            
            # Create consolidated product data matching the OptimizedBatchSteeleTransformer format
            consolidated_products[stock_code] = {
                'stock_code': str(stock_code),
                'title': str(first_row['Product Name']) if pd.notna(first_row['Product Name']) else f"Product {stock_code}",
                'description': str(first_row['Description']) if pd.notna(first_row['Description']) else "",
                'year': str(int(first_row['Year'])) if pd.notna(first_row['Year']) else "1800",
                'make': str(first_row['Make']) if pd.notna(first_row['Make']) else "NONE",
                'model': str(first_row['Model']) if pd.notna(first_row['Model']) else "NONE",
                'price': float(first_row['MAP']) if pd.notna(first_row['MAP']) else 0.0,
                'cost': float(first_row['Dealer Price']) if pd.notna(first_row['Dealer Price']) else 0.0,
                'part_number': str(first_row['PartNumber']) if pd.notna(first_row['PartNumber']) else str(stock_code),
                'car_ids': list(all_car_ids),
                'golden_validated': len(all_car_ids) > 0,
                'vehicle_count': len(group)
            }
        
        print(f"✅ Consolidated {len(df)} rows into {len(consolidated_products)} unique products")
        
        print("\n🏭 PHASE 5: SHOPIFY FORMAT TRANSFORMATION")
        # Initialize transformer for final formatting
        transformer = OptimizedBatchSteeleTransformer(use_ai=False)
        
        # Convert consolidated products to the format expected by the transformer
        from utils.optimized_batch_steele_transformer import ProductData
        
        standard_products = []
        for stock_code, product_data in consolidated_products.items():
            # Create ProductData object
            product = ProductData(
                title=product_data['title'],
                year_min=product_data['year'],
                year_max=product_data['year'],
                make=product_data['make'],
                model=product_data['model'],
                mpn=product_data['part_number'],
                cost=product_data['cost'],
                price=product_data['price'],
                body_html=product_data['description'],
                car_ids=product_data['car_ids'],
                golden_validated=product_data['golden_validated'],
                fitment_source="pattern_mapping",
                processing_method="pattern_based"
            )
            standard_products.append(product)
        
        print(f"✅ Created {len(standard_products)} ProductData objects")
        
        # Apply template enhancements
        enhanced_products = transformer.enhance_with_templates(standard_products)
        print(f"✅ Enhanced products with templates")
        
        # Convert to final Shopify format
        final_df = transformer.transform_to_formatted_shopify_import(enhanced_products)
        print(f"✅ Generated Shopify format with {len(final_df)} products")
        
        print("\n💾 PHASE 6: SAVING RESULTS")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Extract input filename for the output name
        input_filename = Path(input_file).stem
        output_filename = f"pattern_based_{input_filename}_{timestamp}.csv"
        results_path = steele_dir / "data" / "results" / output_filename
        
        # Ensure results directory exists
        os.makedirs(results_path.parent, exist_ok=True)
        
        final_df.to_csv(results_path, index=False)
        
        print(f"\n✅ PATTERN-BASED PROCESSING COMPLETE!")
        print(f"📁 Results saved to: {results_path}")
        print(f"📊 Total products: {len(final_df)}")
        
        # Show statistics
        tagged_products = len([p for p in standard_products if len(p.car_ids) > 0])
        print(f"🏷️  Products with tags: {tagged_products}/{len(standard_products)}")
        
        total_tags = sum(len(p.car_ids) for p in standard_products)
        avg_tags = total_tags / len(standard_products) if standard_products else 0
        print(f"📈 Average tags per product: {avg_tags:.1f}")
        
        return results_path, final_df
        
    except Exception as e:
        print(f"❌ Error during pattern-based processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--submit-only":
            # Submit optimized batch only and save batch ID
            batch_id = main(submit_only=True)
            if batch_id and batch_id != "no_batch_needed":
                print(f"\n🎉 ULTRA-OPTIMIZED BATCH SUBMITTED: {batch_id}")
                print("📝 Save this batch ID to retrieve results later!")
        
        elif sys.argv[1] == "--retrieve" and len(sys.argv) > 2:
            # Retrieve results from completed optimized batch
            batch_id = sys.argv[2]
            result_file = retrieve_optimized_results(batch_id)
            if result_file:
                print(f"\n🎉 ULTRA-OPTIMIZED PROCESSING COMPLETED: {result_file}")
        
        elif sys.argv[1] == "--status" and len(sys.argv) > 2:
            # Check batch status
            from utils.optimized_batch_steele_transformer import OptimizedBatchSteeleTransformer
            batch_id = sys.argv[2]
            transformer = OptimizedBatchSteeleTransformer(use_ai=True)
            status = transformer.batch_ai_matcher.check_batch_status(batch_id)
            
            if status == "completed":
                print("✅ Ultra-optimized batch completed! Ready to retrieve results.")
                print(f"Run: python main_optimized.py --retrieve {batch_id}")
            elif status in ["validating", "in_progress"]:
                print("⏳ Ultra-optimized batch is still processing. Check back later.")
            elif status in ["failed", "expired", "cancelled"]:
                print("❌ Ultra-optimized batch failed or was cancelled.")
        
        elif sys.argv[1] == "--pattern" or sys.argv[1] == "--pattern-process":
            # Pattern-based processing (no AI required)
            input_file = None
            if len(sys.argv) > 2:
                input_file = sys.argv[2]
            
            result = pattern_retrieve_and_process(input_file)
            if result:
                results_path, final_df = result
                print(f"\n🎉 PATTERN-BASED PROCESSING COMPLETED: {results_path}")
        
        elif sys.argv[1] == "--enhanced-pattern":
            # Enhanced pattern processing with golden master + pattern mapping (VECTORIZED)
            print("=" * 80)
            print("🚀 ENHANCED PATTERN PROCESSING (ULTRA-FAST VECTORIZED)")
            print("   Two-stage matching: Golden Master + Pattern Mapping")
            print("   ⚡ Vectorized operations for maximum speed")
            print("=" * 80)
            
            input_file = "data/processed/steele_processed_complete.csv"
            if len(sys.argv) > 2:
                input_file = sys.argv[2]
            
            try:
                from enhanced_pattern_processor import EnhancedPatternProcessor
                
                print(f"📂 Input file: {input_file}")
                start_time = time.time()
                
                processor = EnhancedPatternProcessor()
                output_path, final_df = processor.process_file(input_file)
                
                total_time = time.time() - start_time
                print(f"\n🎉 ENHANCED PATTERN PROCESSING COMPLETED!")
                print(f"⚡ Total processing time: {total_time:.1f}s")
                print(f"📁 Results saved to: {output_path}")
                
                # Show performance tip
                if total_time > 60:  # If over 1 minute
                    print(f"\n💡 Performance tip: For even faster processing on large datasets,")
                    print(f"   use the standalone processor with vectorized mode:")
                    print(f"   python enhanced_pattern_processor.py {input_file} --mode vectorized")
                
            except FileNotFoundError:
                print(f"❌ Error: Input file not found: {input_file}")
                print("💡 Available sample files:")
                import os
                data_dir = Path(__file__).parent / "data"
                if data_dir.exists():
                    for subdir in ["processed", "samples"]:
                        subdir_path = data_dir / subdir
                        if subdir_path.exists():
                            print(f"   {subdir}/:")
                            for file in subdir_path.glob("*.csv"):
                                print(f"     - {file.name}")
                                
            except Exception as e:
                print(f"❌ Enhanced pattern processing failed: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("Ultra-Optimized Steele Processing Usage:")
            print("  python main_optimized.py                             # Complete ultra-optimized processing")
            print("  python main_optimized.py --submit-only               # Submit optimized batch for later")
            print("  python main_optimized.py --status <batch_id>         # Check optimized batch status")
            print("  python main_optimized.py --retrieve <batch_id>       # Retrieve optimized results")
            print("  python main_optimized.py --pattern [input_file]      # Pattern-based processing (NO AI)")
            print("  python main_optimized.py --enhanced-pattern [file]   # Enhanced pattern processing (BEST)")
            print("")
            print("Pattern Processing Options:")
            print("  --pattern:")
            print("    • Uses only pattern_car_id_mapping.json")
            print("    • Fast but may miss exact matches")
            print("  --enhanced-pattern (RECOMMENDED):")
            print("    • Stage 1: Exact matches from golden master dataset")
            print("    • Stage 2: Pattern mapping fallback")
            print("    • Higher coverage and better accuracy")
            print("    • No AI required - instant processing")
            print("")
            print("Examples:")
            print("  python main_optimized.py --enhanced-pattern")
            print("  python main_optimized.py --enhanced-pattern data/samples/steele_test_1000.csv")
            print("  python main_optimized.py --pattern data/processed/my_custom_steele_data.csv")
            print("")
            print("Benefits:")
            print("  • Pattern deduplication eliminates 2+ hour queuing time")
            print("  • 50% batch API cost savings")
            print("  • Same accuracy and quality as original")
            print("  • Handles 300k+ products efficiently")
            sys.exit(1)
    else:
        # Default: complete ultra-optimized processing
        main()