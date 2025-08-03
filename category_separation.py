import pandas as pd

df = pd.read_csv('csv_files/synthetic_online_retail_data.csv')

print("Full dataset preview:")
print(df.head())

fashion_df = df[df['category_name']== 'Fashion']

fashion_df= fashion_df.reset_index(drop=True)

fashion_df.to_csv('synthetic_online_retail_data.csv', index=False)

print("\nFiltered Fashion Category Summary:")
print(fashion_df['category_name'].value_counts())
print(f"\nNumber of records in Fashion category:{len(fashion_df)}")