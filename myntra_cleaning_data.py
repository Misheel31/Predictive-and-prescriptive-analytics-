import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('myntra_products_catalog.csv')

df['category_name'] = 'Fashion'
fashion_df = df.copy()

# Fill missing Gender
fashion_df['Gender'] = fashion_df['Gender'].fillna('Unknown')

# Fill missing PrimaryColor
fashion_df['PrimaryColor'] = fashion_df['PrimaryColor'].fillna('Unknown')

# Simulate order_date
fashion_df['order_date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, size=len(fashion_df)), unit='D')
fashion_df['quantity'] = 1
fashion_df['price'] = fashion_df['Price (INR)']
fashion_df['total_price'] = fashion_df['price'] * fashion_df['quantity']

# Final check
print("\nCleaned Data Summary:")
print(fashion_df.isnull().sum())
