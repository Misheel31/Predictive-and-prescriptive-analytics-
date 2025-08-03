import pandas as pd

df = pd.read_csv("csv_files/combined_order_category_data.csv")

# Summary of missing values
print(df.isna().sum())

df_clean = df.dropna(subset=['product_id', 'category_id', 'price'])

df['price'] = df['price'].fillna(df['price'].median())
df['category_name'] = df['category_name'].fillna(df['category_name'].mode()[0])
print("Missing values after cleaning:\n", df.isna().sum())
