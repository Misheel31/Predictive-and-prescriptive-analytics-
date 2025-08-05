import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# 1. Load and preprocess your data (same as your earlier model code)
df = pd.read_csv("csv_files/myntra_products_catalogs.csv")

df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"price (inr)": "price_inr",
                   "date purchase": "purchase_date",
                   "review rating": "review_rating"}, inplace=True)

# Simulate target variable (quantity)
np.random.seed(42)
df["quantity"] = np.random.poisson(lam=2, size=len(df)) + 1
df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
df["purchase_date"].fillna(pd.Timestamp("2023-01-01"), inplace=True)
df["review_rating"] = df["review_rating"].fillna(df["review_rating"].mean())
df["gender"] = df["gender"].fillna("Unknown")
df = pd.get_dummies(df, columns=["gender"], drop_first=True)
df["payment method"] = df["payment method"].fillna("Unknown")
df = pd.get_dummies(df, columns=["payment method"], drop_first=True)
df["day_of_week"] = df["purchase_date"].dt.dayofweek
df["month"] = df["purchase_date"].dt.month
df["season"] = df["month"] % 12 // 3 + 1
df["log_quantity"] = np.log1p(df["quantity"])

# Prepare features for prediction (same feature list as model training)
features = ["review_rating", "day_of_week", "season", "price_inr"]
features += [c for c in df.columns if c.startswith("gender_") or c.startswith("payment method_")]

X = df[features]
y = df["log_quantity"]

# Import your trained RandomForestRegressor (or re-train it here for demonstration)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Predict demand (log scale) and transform back
df['predicted_log_quantity'] = model.predict(X)
df['PredictedDemand'] = np.expm1(df['predicted_log_quantity'])

# Now for price optimization: use first 10 products for simplicity
opt_df = df[['productid', 'productname', 'price_inr', 'PredictedDemand']].dropna().head(10).copy()

# Add CostPrice as 60% of price_inr (assumption)
opt_df['CostPrice'] = opt_df['price_inr'] * 0.6
# Assume Inventory = 80 for all for example
opt_df['Inventory'] = 80

# Create LP problem
model_lp = LpProblem(name="price-optimization", sense=LpMaximize)

# Decision variables: price for each product (bounded)
price_vars = {
    row.productid: LpVariable(name=f"price_{row.productid}",
                             lowBound=row.CostPrice + 10,
                             upBound=row.price_inr * 1.5)
    for _, row in opt_df.iterrows()
}

# Objective: Maximize profit = (price - cost) * min(predicted demand, inventory)
profit_terms = []
for _, row in opt_df.iterrows():
    pid = row.productid
    cost = row.CostPrice
    demand_limit = min(row.PredictedDemand, row.Inventory)
    profit_terms.append((price_vars[pid] - cost) * demand_limit)

model_lp += lpSum(profit_terms)

# Solve LP
model_lp.solve()

# Output results
print("\n Optimal Pricing Suggestions:\n")
for _, row in opt_df.iterrows():
    pid = row.productid
    product = (row.productname[:35] + "...") if len(row.productname) > 35 else row.productname
    old_price = row.price_inr
    new_price = price_vars[pid].value()
    print(f"{product}\n  Old Price: ₹{old_price:.2f} → New Price: ₹{new_price:.2f}\n")

# Save results
opt_df['OptimizedPrice'] = opt_df['productid'].map(lambda pid: round(price_vars[pid].value(), 2))
opt_df.to_csv("optimized_prices.csv", index=False)
print("\nSaved optimized prices to 'optimized_prices.csv'")
