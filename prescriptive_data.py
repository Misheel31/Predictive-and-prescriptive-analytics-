import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

df = pd.read_csv("csv_files/myntra_products_catalogs.csv")

# Cleaning data and using only the first 10 rows 
df = df[['ProductID', 'ProductName', 'Price (INR)']].dropna().head(10)

# Step 2: Add assumed values
df['CostPrice'] = df['Price (INR)'] * 0.6
df['PredictedDemand'] = 100  
df['Inventory'] = 80

#Creating LP problem
model = LpProblem(name="price-optimization", sense=LpMaximize)

# Decision variables: new selling price
price_vars = {
    row.ProductID: LpVariable(name=f"price_{row.ProductID}", lowBound=row.CostPrice + 10, upBound=row['Price (INR)'] * 1.5)
    for _, row in df.iterrows()
}

# Objective: Maximize profit = (price - cost) * min(demand, inventory)
profit_terms = []
for _, row in df.iterrows():
    pid = row.ProductID
    cost = row.CostPrice
    demand_limit = min(row.PredictedDemand, row.Inventory)
    
    profit = (price_vars[pid] - cost) * demand_limit
    profit_terms.append(profit)

model += lpSum(profit_terms)

# Solve the problem
model.solve()

# Output results
print("\n Optimal Pricing Suggestions:\n")
for _, row in df.iterrows():
    pid = row.ProductID
    product = row.ProductName[:35] + "..."
    orig_price = row['Price (INR)']
    new_price = price_vars[pid].value()
    print(f"{product}\n  Old Price: ₹{orig_price} → New Price: ₹{new_price:.2f}\n")


df['OptimizedPrice'] = df['ProductID'].map(lambda pid: round(price_vars[pid].value(), 2))
df.to_csv("optimized_prices.csv", index=False)
print("\n Saved optimized prices to 'optimized_prices.csv'")
