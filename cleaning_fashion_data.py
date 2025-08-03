import pandas as pd

df = pd.read_csv('csv_files/synthetic_online_retail_data.csv')

fashion_df = df[df['category_name']== 'Fashion'].copy()

# Check missing values
print("\nMissing Values Summary:")
print(fashion_df.isnull().sum())

# Fix missing values safely using .loc
fashion_df.loc[:, 'review_score'] = fashion_df['review_score'].fillna(fashion_df['review_score'].mean())
fashion_df.loc[:, 'gender'] = fashion_df['gender'].fillna('Unknown')

# Convert order_date to datetime format safely
fashion_df.loc[:, 'order_date'] = pd.to_datetime(fashion_df['order_date'])

# Create total_price column safely
fashion_df.loc[:, 'total_price'] = fashion_df['price'] * fashion_df['quantity']
