# import streamlit as st
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Fashion Sales Dashboard", layout="wide")

# # Load data
# df = pd.read_csv("combined_order_category_data.csv", parse_dates=["order_date"])
# df['total_price'] = df['price'] * df['quantity']

# # Title
# st.title("👗 Fashion Sales Dashboard")

# # Sidebar filters
# st.sidebar.header("Filter Data")
# start_date = st.sidebar.date_input("Start Date", df["order_date"].min().date())
# end_date = st.sidebar.date_input("End Date", df["order_date"].max().date())
# city_filter = st.sidebar.multiselect("City", options=df["city"].dropna().unique(), default=df["city"].dropna().unique())
# category_filter = st.sidebar.multiselect("Category", options=df["category_name"].dropna().unique(), default=df["category_name"].dropna().unique())
# gender_filter = st.sidebar.multiselect("Gender", options=df["gender"].dropna().unique(), default=df["gender"].dropna().unique())

# # Filter data
# filtered_df = df[
#     (df["order_date"].dt.date >= start_date) &
#     (df["order_date"].dt.date <= end_date) &
#     (df["city"].isin(city_filter)) &
#     (df["category_name"].isin(category_filter)) &
#     (df["gender"].isin(gender_filter))
# ]

# # Metrics
# total_revenue = filtered_df["total_price"].sum()
# avg_order_value = filtered_df["total_price"].mean()
# total_orders = filtered_df.shape[0]

# col1, col2, col3 = st.columns(3)
# col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
# col2.metric("📦 Total Orders", total_orders)
# col3.metric("💳 Avg Order Value", f"${avg_order_value:,.2f}")

# # Raw data toggle
# if st.checkbox("🔍 Show raw data"):
#     st.write(filtered_df.head())

# # Revenue by gender
# st.subheader("Total Revenue by Gender")
# revenue_by_gender = filtered_df.groupby("gender")["total_price"].sum().reset_index()
# fig1 = px.bar(revenue_by_gender, x="gender", y="total_price", color="gender", title="Revenue by Gender")
# st.plotly_chart(fig1, use_container_width=True)

# # Avg review score by category
# st.subheader("Avg Review Score by Category")
# avg_review = filtered_df.groupby("category_name")["review_score"].mean().reset_index()
# fig2 = px.bar(avg_review, x="category_name", y="review_score", title="Avg Review Score by Category")
# st.plotly_chart(fig2, use_container_width=True)

# # Sales over time
# st.subheader("Sales Over Time")
# sales_time = filtered_df.groupby("order_date")["total_price"].sum().reset_index()
# fig3 = px.line(sales_time, x="order_date", y="total_price", title="Sales Over Time")
# st.plotly_chart(fig3, use_container_width=True)

# # Top 5 best-selling products
# st.subheader("Top 5 Best-Selling Products")
# top_products = filtered_df.groupby("product_name")["total_price"].sum().nlargest(5).reset_index()
# fig4 = px.bar(top_products, x="product_name", y="total_price", title="Top 5 Products", color="total_price")
# st.plotly_chart(fig4, use_container_width=True)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Set up Streamlit
st.set_page_config(layout="wide")
st.title("Myntra Product Catalog - Exploratory Data Analysis")

# Load dataset
df = pd.read_csv("myntra_products_catalog.csv")

# Clean and normalize column names
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"price (inr)": "price_inr"}, inplace=True)

# Sidebar filters
st.sidebar.header("Filter Options")

gender_options = df["gender"].dropna().unique()
brand_options = df["productbrand"].dropna().unique()

gender_filter = st.sidebar.multiselect("Gender", options=gender_options, default=gender_options)
brand_filter = st.sidebar.multiselect("Product Brand", options=brand_options, default=brand_options)
st.write(df.columns.tolist())

# Filtered DataFrame
filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["productbrand"].isin(brand_filter))
]

st.subheader("Filtered Data Overview")
st.write(filtered_df.head())

# ---------- Plot 1: Price Distribution ----------
st.subheader("Price Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_df["price_inr"], bins=50, kde=True, ax=ax1)
st.pyplot(fig1)

# ---------- Plot 2: Top 10 Product Brands ----------
st.subheader("Top 10 Product Brands by Count")
top_brands = filtered_df["productbrand"].value_counts().head(10)
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax2)
ax2.set_xlabel("Count")
ax2.set_ylabel("Product Brand")
st.pyplot(fig2)

# ---------- Plot 3: Price by Gender ----------
st.subheader("Price Distribution by Gender")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.boxplot(data=filtered_df, x="gender", y="price_inr", ax=ax3)
st.pyplot(fig3)

# ---------- Plot 4: Correlation Heatmap ----------
st.subheader("Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"])
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

