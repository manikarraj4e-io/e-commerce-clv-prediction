# 🛒 Enterprise Customer Lifetime Value (CLV) Predictive Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, production-oriented machine learning framework and interactive decision engine designed to forecast a customer's **5-year monetary value** based exclusively on early-stage purchase behaviors.

---

## Executive Summary & Business Impact

In high-growth e-commerce, **Customer Acquisition Cost (CAC)** continues to rise globally. Predicting long-term **Customer Lifetime Value (CLV)** early in the user journey is critical for optimizing unit economics and capital allocation.

This framework processes transactional log data from a customer's **first 3 orders** to predict long-term revenue velocity. 

### Key Business Applications:
* **Capital Allocation:** Direct performance marketing spend toward acquisition channels that yield high-tier revenue cohorts.
* **Proactive VIP Retention:** Automatically identify high-potential customers early to trigger tailored loyalty campaigns and VIP onboarding workflows.
* **Churn Mitigation:** Detect low-velocity, low-engagement cohorts to trigger targeted automated re-engagement flows without over-discounting.

---

## System Architecture & Data Pipeline

┌───────────────────────────┐
│ Raw Transaction Receipts  │ (InvoiceDate, Price, Quantity, StockCode)
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ Timeline Segmentation     │ Rank purchases per customer ID; isolate Orders 1–3
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ Feature Engineering Layer │ Recency gaps, Basket Depth, Catalog Breadth, Basket Value
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ XGBoost Gradient Boosting │ Hyperparameter-tuned regression forecasting engine
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ Streamlit Decision Engine │ Executive dashboards & real-time customer simulator
└───────────────────────────┘


---

## Feature Engineering Matrix

Raw transactional logs are transformed into tabular vector representations capturing key dimensions of early buying behavior:

| Feature Dimension | Variable Name | Business Rationale |
| :--- | :--- | :--- |
| **Initial Capital Commitment** | `Total_Spent_First_3` | Cumulative monetary commitment across initial trial phase. |
| **Basket Economics** | `Avg_Order_Value` | Baseline willingness-to-spend per transaction. |
| **Purchasing Velocity** | `Days_Between_1st_and_3rd` | Recency/frequency signal measuring days elapsed between Order 1 and Order 3. |
| **Order Volume** | `Total_Items_Bought` | Physical unit velocity across initial purchases. |
| **Catalog Breadth** | `Unique_Products_Bought` | Variety index measuring customer exploration across product categories. |
| **Item Price Threshold** | `Max_Single_Item_Spend` | Identifies luxury/high-ticket item buyers versus budget-conscious shoppers. |
| **Order Continuity** | `Total_Orders_Count` | Quantifies distinct completed transaction batches. |

---

## Model Architecture & Performance

The predictive core utilizes an **XGBoost Regressor** tuned for non-linear feature interactions and high-dimensional tabular data.

* **Primary Model:** XGBoost Gradient Boosted Decision Trees
* **Hyperparameters:** `n_estimators=150`, `learning_rate=0.08`, `max_depth=5`
* **Validation Strategy:** Chronological/Holdout Train-Test Split (80/20)
* **Model $R^2$ Score:** `0.41`
* **Root Mean Squared Error (RMSE):** `$13,857` *(Driven by B2B wholesale transaction volume spikes in historical logs)*

---

## Dashboard Capabilities

The Streamlit decision center equips stakeholders with two functional modules:

### 1. Executive Analytics Overview
* **Key Feature Drivers:** Plotly horizontal bar visualization ranking feature importance scores derived from XGBoost gradient boosting.
* **Cohort Distribution:** Log-scaled frequency distribution detailing customer value concentration.
* **Model Diagnostic:** Scatter plot comparing ground-truth actual spend against model predicted trajectory.
* **Customer Tier Segmentation:** Donut chart breaking down customer cohorts into **High-Value VIP**, **Standard Buyer**, and **Low-Value / Single-Purchase** segments.

### 2. Interactive Scenario Simulator
Allows growth marketers and finance leaders to input customer transactional parameters and instantly generate 5-year CLV predictions alongside prescribed strategic business recommendations.

---

## Quick Start & Local Deployment

### Prerequisites
* Python 3.10+
* Git

### Installation Steps

```bash
git clone https://github.com/Emmanuelrajj4e/e-commerce-clv-prediction.git
cd e-commerce-clv-prediction
pip install -r requirements.txt
streamlit run app.py
