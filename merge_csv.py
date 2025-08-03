import pandas as pd

# Load data
retail_df = pd.read_csv("csv_files/synthetic_online_retail_data.csv")
category_df = pd.read_csv("csv_files/fashion_category_data.csv")

# Clean types
retail_df['product_id'] = retail_df['product_id'].astype(str).str.strip()
category_df['product_id'] = category_df['product_id'].astype(str).str.strip()

# Merge based on product_id
merged_df = pd.merge(
    retail_df,
    category_df,
    on='product_id',
    how='left',
    suffixes=('', '_cat')
)

# Save output
merged_df.to_csv("combined_order_category_data.csv", index=False)

# Print merge stats
print(f" Merged rows: {len(merged_df)}")
print(f" Missing category info: {merged_df['category_id'].isna().sum()}")
