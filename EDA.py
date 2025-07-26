import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("myntra_products_catalog.csv")

# Rename columns for easier access or use as-is with proper quotes
df.rename(columns={"Price (INR)": "Price_INR"}, inplace=True)

# Distribution of price
plt.figure(figsize=(8, 5))
sns.histplot(df['Price_INR'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price (INR)")
plt.show()

#'ProductBrand' to show top 10 brands by count
top_brands = df['ProductBrand'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_brands.values, y=top_brands.index)
plt.title("Top 10 Product Brands")
plt.xlabel("Count")
plt.ylabel("Product Brand")
plt.show()

# Price by gender boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Gender', y='Price_INR')
plt.title("Price Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Price (INR)")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df[['Price_INR', 'NumImages']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
