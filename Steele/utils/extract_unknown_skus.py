import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
steele_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def extract_unknown_skus_from_steele_data(input_file: str = None, output_file: str = None) -> pd.DataFrame:
    """
    Extract all rows with Unknown,Unknown make/model from Steele data for AI processing.
    This addresses the hallucination issue where AI incorrectly generates Unknown tags
    when fitment information is actually available in the description.
    
    Args:
        input_file: Path to input CSV file (default: samples/first_100_stock_codes_sample.csv)
        output_file: Path to output CSV file (default: samples/unknown_skus_sample.csv)
        
    Returns:
        DataFrame containing only Unknown SKUs with their descriptions for AI processing
    """
    
    # Set default paths
    if input_file is None:
        input_file = steele_root / "data" / "samples" / "first_100_stock_codes_sample.csv"
    if output_file is None:
        output_file = steele_root / "data" / "samples" / "unknown_skus_sample.csv"
    
    print(f"📖 Loading Steele data from: {input_file}")
    
    # Load the data
    try:
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df)} total records")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load data from {input_file}: {str(e)}")
    
    # Filter for Unknown,Unknown entries
    # Looking for rows where Make and Model are both "Unknown"
    if 'Make' not in df.columns or 'Model' not in df.columns:
        raise ValueError("Data file must contain 'Make' and 'Model' columns")
    
    unknown_filter = (df['Make'] == 'Unknown') & (df['Model'] == 'Unknown')
    unknown_df = df[unknown_filter].copy()
    
    print(f"✅ Found {len(unknown_df)} Unknown SKUs out of {len(df)} total records")
    
    if len(unknown_df) == 0:
        print("⚠️  No Unknown SKUs found in the data")
        return pd.DataFrame()
    
    # Report on the issues found
    print("\n🔍 Analysis of Unknown SKUs:")
    print(f"   - {len(unknown_df)} records have Unknown Make/Model")
    print(f"   - This represents {len(unknown_df)/len(df)*100:.1f}% of the dataset")
    
    # Check which ones have descriptions that might contain fitment info
    description_available = unknown_df['Description'].notna().sum()
    print(f"   - {description_available} Unknown SKUs have descriptions available for AI extraction")
    
    # Show examples of descriptions that contain potential fitment info
    print("\n📝 Sample descriptions containing potential fitment information:")
    for idx, row in unknown_df.head(3).iterrows():
        description = str(row['Description'])[:150]
        print(f"   SKU {row['StockCode']}: {description}...")
    
    # Save the filtered data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    unknown_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved Unknown SKUs to: {output_file}")
    
    return unknown_df

def analyze_fitment_potential_in_descriptions(unknown_df: pd.DataFrame) -> None:
    """
    Analyze the descriptions to identify which ones contain potential fitment information
    that could be extracted using AI, similar to the REM approach.
    """
    
    print("\n🔍 Analyzing fitment extraction potential:")
    
    # Common patterns that indicate fitment information
    fitment_indicators = [
        'compatible with',
        'fits',
        'for use with',
        'models',
        'year',
        '19\d{2}',  # Years like 1912, 1915
        '20\d{2}',  # Years like 2005, 2010
        'sterns-knight',
        'independent',
        'ford',
        'chevrolet',
        'chrysler',
        'dodge',
        'plymouth'
    ]
    
    fitment_potential = []
    
    for idx, row in unknown_df.iterrows():
        description = str(row['Description']).lower()
        matches = []
        
        for indicator in fitment_indicators:
            if indicator in description:
                matches.append(indicator)
        
        if matches:
            fitment_potential.append({
                'StockCode': row['StockCode'],
                'ProductName': row['Product Name'],
                'Description': row['Description'],
                'FitmentIndicators': matches,
                'PotentialExtractable': len(matches) >= 2  # At least 2 indicators
            })
    
    extractable_count = sum(1 for item in fitment_potential if item['PotentialExtractable'])
    
    print(f"   - {len(fitment_potential)} SKUs have potential fitment indicators")
    print(f"   - {extractable_count} SKUs have high extraction potential (2+ indicators)")
    print(f"   - {extractable_count/len(unknown_df)*100:.1f}% extraction success rate expected")
    
    # Show examples
    print(f"\n📋 Examples with high extraction potential:")
    high_potential = [item for item in fitment_potential if item['PotentialExtractable']][:3]
    
    for item in high_potential:
        print(f"   SKU {item['StockCode']}:")
        print(f"     Product: {item['ProductName']}")
        print(f"     Indicators: {', '.join(item['FitmentIndicators'])}")
        print(f"     Description: {item['Description'][:100]}...")
        print()

def main():
    """Main function to extract Unknown SKUs and analyze fitment potential"""
    
    print("🚀 Steele Unknown SKU Extraction Tool")
    print("=" * 50)
    print("This tool extracts SKUs with Unknown make/model for AI-based fitment extraction.")
    print("Similar to the REM model approach for incomplete fitment data.\n")
    
    try:
        # Extract Unknown SKUs
        unknown_df = extract_unknown_skus_from_steele_data()
        
        if len(unknown_df) > 0:
            # Analyze fitment extraction potential
            analyze_fitment_potential_in_descriptions(unknown_df)
            
            print("\n✅ Next Steps:")
            print("1. Use the generated unknown_skus_sample.csv for AI fitment extraction")
            print("2. Apply REM-style AI extraction to parse descriptions for year/make/model")
            print("3. Validate AI extractions against Golden Master dataset")
            print("4. Replace Unknown tags with accurate vehicle fitment data")
            
        else:
            print("No Unknown SKUs found to process.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 