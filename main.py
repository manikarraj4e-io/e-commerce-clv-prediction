import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb  # Import XGBoost

print("Loading dataset...")
df = pd.read_csv('online_retail_II.csv')

print("Cleaning data...")
df = df.dropna(subset=['Customer ID'])
df = df[~df['Invoice'].astype(str).str.startswith('C')]
df['TotalPrice'] = df['Quantity'] * df['Price']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Processing customer timelines...")
df = df.sort_values(['Customer ID', 'InvoiceDate'])
df['OrderRank'] = df.groupby('Customer ID')['InvoiceDate'].rank(method='dense', ascending=True)

first_three_df = df[df['OrderRank'] <= 3]
future_df = df[df['OrderRank'] > 3]

print("Engineering advanced features...")
features = first_three_df.groupby('Customer ID').agg(
    Total_Spent_First_3=('TotalPrice', 'sum'),
    Avg_Order_Value=('TotalPrice', 'mean'),
    Days_Between_1st_and_3rd=('InvoiceDate', lambda x: (x.max() - x.min()).days),
    Total_Items_Bought=('Quantity', 'sum'),
    Unique_Products_Bought=('StockCode', 'nunique'),  # Variety of items bought
    Max_Single_Item_Spend=('TotalPrice', 'max'),      # Did they buy an expensive single item?
    Total_Orders_Count=('Invoice', 'nunique')         # Exact distinct orders in first batch
).reset_index()

target = future_df.groupby('Customer ID')['TotalPrice'].sum().reset_index()
target.columns = ['Customer ID', 'Future_Spend']

model_df = pd.merge(features, target, on='Customer ID', how='left')
model_df['Future_Spend'] = model_df['Future_Spend'].fillna(0)

print("Training XGBoost Model...")
X = model_df[[
    'Total_Spent_First_3', 
    'Avg_Order_Value', 
    'Days_Between_1st_and_3rd', 
    'Total_Items_Bought',
    'Unique_Products_Bought',
    'Max_Single_Item_Spend',
    'Total_Orders_Count'
]]
y = model_df['Future_Spend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use XGBoost Regressor instead of Random Forest
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- XGBoost Model Performance ---")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")