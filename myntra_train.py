import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load Myntra product data
df = pd.read_csv("myntra_products_catalog.csv")

# Renaming price column
df = df.rename(columns={"Price (INR)": "price"})

# Simulate order_date, quantity, and review_score
np.random.seed(42)
df['order_date'] = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(0, 365, size=len(df)), unit="D")
df['quantity'] = np.random.poisson(lam=2, size=len(df)) + 1  # ensure quantity >= 1
df['review_score'] = np.round(np.random.uniform(2.5, 5.0, size=len(df)), 1)

# Create derived features
df['day_of_week'] = df['order_date'].dt.dayofweek
df['season'] = df['order_date'].dt.month % 12 // 3 + 1

# Encode gender
df['Gender'] = df['Gender'].fillna("Unknown")
df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

df['log_quantity'] = np.log1p(df['quantity'])

# Plot quantity distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['quantity'], bins=30, kde=True)
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr_features = ['quantity', 'review_score', 'day_of_week', 'season', 'price']
corr = df[corr_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Select features
features = ['review_score', 'day_of_week', 'season', 'price']
features += [col for col in df.columns if col.startswith('Gender_')]

X = df[features]
y = df['log_quantity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

print("\n--- Initial Model Evaluation ---")
print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"Root Mean Squared Error: {mean_squared_error(y_true, y_pred, squared=False):.4f}")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model evaluation
print("\n--- Best Parameters from GridSearchCV ---")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best_log = best_model.predict(X_test)
y_pred_best = np.expm1(y_pred_best_log)

print("\n--- Tuned Model Evaluation ---")
print(f"Tuned R2 Score: {r2_score(y_true, y_pred_best):.4f}")
print(f"Tuned Mean Absolute Error: {mean_absolute_error(y_true, y_pred_best):.4f}")
print(f"Tuned RMSE: {mean_squared_error(y_true, y_pred_best, squared=False):.4f}")

# Feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances (Tuned Model)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.tight_layout()
plt.show()

print("\n Predictive demand forecasting completed successfully.")
