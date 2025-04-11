
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customer_purchases.csv", parse_dates=["PurchaseDate"])

# Feature Engineering
df['TotalPrice'] = df['Quantity'] * df['Price']
df['Month'] = df['PurchaseDate'].dt.to_period("M")
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"])

# Top Products by Revenue
top_products = df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette="coolwarm")
plt.title("Top Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("top_products.png")
plt.close()

# Monthly Revenue Trend
monthly_revenue = df.groupby("Month")["TotalPrice"].sum()
plt.figure(figsize=(8, 5))
monthly_revenue.plot(marker='o', color='blue')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_revenue.png")
plt.close()

# Gender Distribution
plt.figure(figsize=(6, 6))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=["skyblue", "lightpink"])
plt.title("Gender Distribution")
plt.ylabel('')
plt.tight_layout()
plt.savefig("gender_distribution.png")
plt.close()

# Region vs Category Heatmap
pivot = df.pivot_table(index='Region', columns='Category', values='TotalPrice', aggfunc='sum')
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Revenue by Region and Category")
plt.tight_layout()
plt.savefig("region_category_heatmap.png")
plt.close()
