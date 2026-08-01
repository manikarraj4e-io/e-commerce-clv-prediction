import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Professional CLV Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL UI STYLING (DARK SIDEBAR & CLEAN MAIN) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f8fafc !important;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & MODEL TRAINING (Cached) ---
@st.cache_data
def load_and_train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_csv_path = os.path.join(base_dir, "online_retail_II.csv")

    df = None

    # Option 1: Try reading local CSV if present
    if os.path.exists(local_csv_path):
        try:
            df = pd.read_csv(local_csv_path, encoding='latin1')
        except Exception:
            df = None

    # Option 2: Fallback to direct raw CSV dataset online
    if df is None:
        try:
            st.info("⏬ Local dataset 'online_retail_II.csv' not found. Fetching clean online dataset stream...")
            fallback_url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
            df = pd.read_csv(fallback_url, encoding='latin1')
        except Exception as e:
            st.error(f"❌ Failed to load dataset. Error details: {e}")
            st.stop()

    # Standardize column names across dataset variants
    col_rename = {
        'Customer ID': 'Customer ID',
        'CustomerID': 'Customer ID',
        'Invoice': 'Invoice',
        'InvoiceNo': 'Invoice',
        'InvoiceDate': 'InvoiceDate',
        'Quantity': 'Quantity',
        'Price': 'Price',
        'UnitPrice': 'Price',
        'StockCode': 'StockCode'
    }
    df = df.rename(columns=col_rename)

    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    df = df.sort_values(['Customer ID', 'InvoiceDate'])
    df['OrderRank'] = df.groupby('Customer ID')['InvoiceDate'].rank(method='dense', ascending=True)
    
    first_three_df = df[df['OrderRank'] <= 3]
    future_df = df[df['OrderRank'] > 3]
    
    features = first_three_df.groupby('Customer ID').agg(
        Total_Spent_First_3=('TotalPrice', 'sum'),
        Avg_Order_Value=('TotalPrice', 'mean'),
        Days_Between_1st_and_3rd=('InvoiceDate', lambda x: (x.max() - x.min()).days),
        Total_Items_Bought=('Quantity', 'sum'),
        Unique_Products_Bought=('StockCode', 'nunique'),
        Max_Single_Item_Spend=('TotalPrice', 'max'),
        Total_Orders_Count=('Invoice', 'nunique')
    ).reset_index()
    
    target = future_df.groupby('Customer ID')['TotalPrice'].sum().reset_index()
    target.columns = ['Customer ID', 'Future_Spend']
    
    model_df = pd.merge(features, target, on='Customer ID', how='left')
    model_df['Future_Spend'] = model_df['Future_Spend'].fillna(0)
    
    feature_cols = [
        'Total_Spent_First_3', 'Avg_Order_Value', 'Days_Between_1st_and_3rd', 
        'Total_Items_Bought', 'Unique_Products_Bought', 'Max_Single_Item_Spend', 'Total_Orders_Count'
    ]
    
    X = model_df[feature_cols]
    y = model_df['Future_Spend']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    return model, rmse, r2, model_df, feature_cols, eval_df

with st.spinner("Training XGBoost model & compiling professional dashboard..."):
    model, rmse, r2, model_df, feature_cols, eval_df = load_and_train_model()

# --- HEADER SECTION ---
st.title("Customer Lifetime Value Command Center")
st.markdown("Financial forecasting dashboard for strategic e-commerce marketing decisions.")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.title("📊 Global Metrics")
st.sidebar.metric(label="Model R² Score", value=f"{r2:.2f}")
st.sidebar.metric(label="RMSE", value=f"{rmse:,.0f}")

st.sidebar.markdown("---")
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select View", ["Executive Overview", "Customer Simulator"])

st.sidebar.markdown("---")
st.sidebar.info("Model trained on historical transaction logs to predict 5-year future customer spend.")

# --- VIEW 1: EXECUTIVE OVERVIEW ---
if app_mode == "Executive Overview":
    st.subheader("Enterprise Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Key Drivers of Future Value")
        st.write("Behavioral features from the first 3 purchases ranked by importance.")
        
        importance = model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        feature_map = {
            'Total_Spent_First_3': 'Total Spent (3 Orders)',
            'Avg_Order_Value': 'Avg Order Value',
            'Days_Between_1st_and_3rd': 'Days Elapsed (1st to 3rd)',
            'Total_Items_Bought': 'Total Items',
            'Unique_Products_Bought': 'Unique Products',
            'Max_Single_Item_Spend': 'Max Item Price',
            'Total_Orders_Count': 'Distinct Orders'
        }
        importance_df['Feature'] = importance_df['Feature'].map(feature_map)
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale=px.colors.sequential.Tealgrn,
                                template='plotly_white')
        fig_importance.update_layout(xaxis_title="Feature Importance Score", yaxis_title="", margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        st.markdown("### 2. Future Revenue Distribution")
        st.write("Customer spend concentration across the historical dataset (Log Scale).")
        
        fig_hist = px.histogram(model_df[model_df['Future_Spend'] > 0], x='Future_Spend', 
                                nbins=50, log_x=True, 
                                color_discrete_sequence=['#3b82f6'],
                                template='plotly_white')
        fig_hist.update_layout(xaxis_title="Future Spend ($ - Log Scale)", yaxis_title="Customer Count", margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 3. Actual vs. Predicted Performance")
        st.write("Evaluating model accuracy on test holdout data (Sampled).")
        
        sample_eval = eval_df.sample(min(400, len(eval_df)))
        fig_scatter = px.scatter(sample_eval, x='Actual', y='Predicted', opacity=0.5,
                                 template='plotly_white', trendline="ols")
        
        max_val = max(sample_eval['Actual'].max(), sample_eval['Predicted'].max())
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(dash="dash", color="red", width=2))
        fig_scatter.update_layout(xaxis_title="Actual Future Spend ($)", yaxis_title="Predicted Spend ($)", 
                                  margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.markdown("### 4. Customer Segmentation Breakdown")
        st.write("Proportion of customer base categorized by predicted value tiers.")
        
        def categorize_tier(val):
            if val > 3000: return "High-Value VIP (> $3k)"
            elif val > 500: return "Standard Buyer ($500 - $3k)"
            else: return "Low Value (< $500)"
            
        model_df['Tier'] = model_df['Future_Spend'].apply(categorize_tier)
        tier_counts = model_df['Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        
        fig_donut = px.pie(tier_counts, names='Tier', values='Count', hole=0.5,
                           color_discrete_sequence=px.colors.sequential.Sunset,
                           template='plotly_white')
        fig_donut.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_donut, use_container_width=True)

# --- VIEW 2: CUSTOMER SIMULATOR ---
elif app_mode == "Customer Simulator":
    st.subheader("Live Customer Value Calculator")
    st.markdown("Adjust parameters below to simulate a customer's first 3 purchase behaviors and predict their long-term revenue impact.")

    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        total_spent = st.number_input("Total Spent in First 3 Purchases ($)", min_value=1.0, max_value=50000.0, value=150.0)
        avg_order = st.number_input("Average Order Value ($)", min_value=1.0, max_value=10000.0, value=50.0)
        days_between = st.slider("Days Elapsed Between 1st and 3rd Purchase", min_value=0, max_value=365, value=15)
        total_items = st.number_input("Total Items Bought", min_value=1, max_value=5000, value=10)
        
    with col_input2:
        unique_products = st.number_input("Unique Products Bought (Variety)", min_value=1, max_value=1000, value=5)
        max_single_item = st.number_input("Max Single Item Spend ($)", min_value=1.0, max_value=5000.0, value=40.0)
        total_orders = st.slider("Exact Distinct Orders Count", min_value=1, max_value=3, value=3)

    st.markdown("---")
    
    if st.button("Run 5-Year CLV Prediction", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[
            total_spent, avg_order, days_between, total_items, unique_products, max_single_item, total_orders
        ]], columns=feature_cols)
        
        with st.spinner("Calculating prediction..."):
            prediction = model.predict(input_data)[0]
            predicted_value = max(0, prediction)
        
        st.success("Prediction Complete")
        st.metric(label="Estimated 5-Year Customer Value", value=f"${predicted_value:,.2f}")
        
        if predicted_value > 3000:
            st.balloons()
            st.info("🏆 **Segment: High-Value VIP** — Action: Enroll immediately in premium loyalty tier. Assign dedicated account manager.")
        elif predicted_value > 500:
            st.info("👤 **Segment: Standard Buyer** — Action: Target with standard retention campaigns and cross-sell offers.")
        else:
            st.info("🧊 **Segment: Low Value / One-Time** — Action: Automate email marketing. Do not invest heavy acquisition costs.")