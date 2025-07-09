#!/usr/bin/env python3
import pandas as pd
import os

# Load optimized file
df = pd.read_csv('data/results/optimized_shopify_format_20250705_161905.csv')

# Create sample
sample = df.head(10)
sample.to_csv('data/results/OPTIMIZED_SAMPLE_10_products.csv', index=False)

print(f"✅ Optimized sample created!")
print(f"   📁 File: data/results/OPTIMIZED_SAMPLE_10_products.csv")
print(f"   📊 Size: {os.path.getsize('data/results/OPTIMIZED_SAMPLE_10_products.csv') / 1024:.1f} KB")
print(f"   📊 Products: {len(sample)}")

print("\n🔍 Sample preview:")
for i in range(3):
    title = sample.iloc[i]['Title']
    tags = str(sample.iloc[i]['Tags'])[:50]
    sku = sample.iloc[i]['Variant SKU']
    price = sample.iloc[i]['Variant Price']
    print(f"   {i+1}. {title} - SKU: {sku} - ${price}")
    print(f"      Tags: {tags}...") 