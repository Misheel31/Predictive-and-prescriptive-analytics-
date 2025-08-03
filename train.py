import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

fashion_df = pd.read_csv('csv_files/combined_order_category_data.csv')

fashion_df['order_date_cat'] = pd.to_datetime(fashion_df['order_date_cat'], errors='coerce')

# Extract day of week and season from 'order_date_cat'
fashion_df['day_of_week'] = fashion_df['order_date_cat'].dt.dayofweek
fashion_df['season'] = fashion_df['order_date_cat'].dt.month % 12 // 3 + 1

# One-hot encoding categorical variables (original categorical columns)
fashion_df = pd.get_dummies(
    fashion_df,
    columns=['gender', 'category_name', 'payment_method', 'city'],
    drop_first=True
)

cat_cols = ['gender_cat', 'category_name_cat', 'payment_method_cat', 'city_cat']
for col in cat_cols:
    if col in fashion_df.columns:
        le = LabelEncoder()
        fashion_df[col] = le.fit_transform(fashion_df[col].astype(str))

fashion_df['log_quantity'] = np.log1p(fashion_df['quantity'])

# Plot quantity distribution (original quantity, not logged)
plt.figure(figsize=(6, 4))
sns.histplot(fashion_df['quantity'], bins=30, kde=True)
plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric features
features_corr = ['quantity', 'review_score', 'day_of_week', 'season', 'price', 'age']
corr = fashion_df[features_corr].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Selecting features for modeling
features = ['review_score', 'day_of_week', 'season', 'price', 'age']

# Added one-hot encoded columns for gender, category_name, payment_method, and city
prefixes = ['gender_', 'category_name_', 'payment_method_', 'city_']
for prefix in prefixes:
    features += [col for col in fashion_df.columns if col.startswith(prefix)]

# Added label encoded _cat columns to features
for col in cat_cols:
    if col in fashion_df.columns:
        features.append(col)

X = fashion_df[features]
y = fashion_df['log_quantity'] 

# Drop any non-numeric columns if any remain (should not happen)
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print("Warning: Non-numeric columns found and will be dropped:", non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train initial RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on test set (log scale)
y_pred_log = model.predict(X_test)

# Reverse log transform to get original quantity scale
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Evaluation metrics
print("\n--- Initial Model Evaluation ---")
print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"Root Mean Squared Error: {mean_squared_error(y_true, y_pred, squared=False):.4f}")

# Hyperparameter tuning with GridSearchCV
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

print("\n--- Best Parameters from GridSearchCV ---")
print(grid_search.best_params_)

# Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_best_log = best_model.predict(X_test)
y_pred_best = np.expm1(y_pred_best_log)

print("\n--- Tuned Model Evaluation ---")
print(f"Tuned R2 Score: {r2_score(y_true, y_pred_best):.4f}")
print(f"Tuned Mean Absolute Error: {mean_absolute_error(y_true, y_pred_best):.4f}")
print(f"Tuned Root Mean Squared Error: {mean_squared_error(y_true, y_pred_best, squared=False):.4f}")

# Plot feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [X.columns[i] for i in indices]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Tuned Model)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.tight_layout()
plt.show()

print("\nPredictive demand forecasting completed successfully.")
