import pandas as pd
from transform import transform_data

golden_df = pd.read_csv('data/samples/golden_sample.csv')
vendor_df = pd.read_csv('data/samples/REM_sample.csv')

# Transform data
result_df = transform_data(golden_df, vendor_df)

print(result_df)