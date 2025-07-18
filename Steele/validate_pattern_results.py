#!/usr/bin/env python3
"""
Validation script to show pattern processor results in a readable format
"""

import pandas as pd
from pathlib import Path

def main():
    # Find the most recent pattern tagged file
    results_dir = Path("data/results")
    pattern_files = list(results_dir.glob("pattern_tagged_shopify_*.csv"))
    
    if not pattern_files:
        print("No pattern tagged files found!")
        return
        
    latest_file = max(pattern_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    # Load the results
    df = pd.read_csv(latest_file)
    
    print(f"\n=== Pattern Processing Results Summary ===")
    print(f"Total products: {len(df)}")
    print(f"Products with tags: {len(df[df['Tags'] != ''])}")
    print(f"Products without tags: {len(df[df['Tags'] == ''])}")
    print(f"Percentage with tags: {len(df[df['Tags'] != '']) / len(df) * 100:.1f}%")
    
    print(f"\n=== Sample Products with Tags ===")
    tagged_products = df[df['Tags'] != ''].head(5)
    
    for idx, row in tagged_products.iterrows():
        print(f"\nProduct: {row['Title']}")
        print(f"SKU: {row['Variant SKU']}")
        print(f"Price: ${row['Variant Price']}")
        print(f"Tags: {row['Tags'][:100]}{'...' if len(row['Tags']) > 100 else ''}")
        tag_count = len([t.strip() for t in row['Tags'].split(',') if t.strip()])
        print(f"Tag count: {tag_count}")
    
    print(f"\n=== Tag Statistics ===")
    # Calculate tag statistics
    all_tags = []
    for tags_str in df[df['Tags'] != '']['Tags']:
        if pd.notna(tags_str) and tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    unique_tags = set(all_tags)
    print(f"Total tag instances: {len(all_tags)}")
    print(f"Unique tags: {len(unique_tags)}")
    print(f"Average tags per product: {len(all_tags) / len(df[df['Tags'] != '']):.1f}")
    
    # Show most common tags
    from collections import Counter
    tag_counts = Counter(all_tags)
    print(f"\n=== Most Common Tags (Top 10) ===")
    for tag, count in tag_counts.most_common(10):
        print(f"{tag}: {count} products")

if __name__ == "__main__":
    main()