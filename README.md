# ML Model Repository

This repository contains machine learning models for various business use cases including sales forecasting and customer behavior prediction.

## 📁 Model Files

### **1. Ice Cream Sales Forecasting Models**

#### **ML-Job-1-child1_model.pkl**
- **Type**: Linear Regression
- **Purpose**: Predicts ice cream sales based on weather and temporal factors
- **Features**: Date, DayOfWeek, Month, Temperature, Rainfall
- **Target**: IceCreamsSold
- **Algorithm**: Linear Regression
- **Serialization Date**: 2026-01-29T18:19:47.681000
- **Scikit-learn Version**: 1.3.0

#### **ML-Job-2-child3_model.pkl**
- **Type**: Lasso Regression
- **Purpose**: Ice cream sales prediction with regularization
- **Features**: Date, DayOfWeek, Month, Temperature, Rainfall
- **Target**: IceCreamsSold
- **Algorithm**: Lasso Regression (alpha=1.0)
- **Serialization Date**: 2026-01-29T18:33:09.378000
- **Scikit-learn Version**: 1.3.0

#### **ML-Job-2-child3-1_model.pkl**
- **Type**: Lasso Regression
- **Purpose**: Updated ice cream sales prediction model
- **Features**: Date, DayOfWeek, Month, Temperature, Rainfall
- **Target**: IceCreamsSold
- **Algorithm**: Lasso Regression (alpha=1.0)
- **Serialization Date**: 2026-01-29T18:42:52.078000
- **Scikit-learn Version**: 1.3.0

### **2. Customer Spending Prediction Model**

#### **ML-Job-3-child3-1_model.pkl**
- **Type**: Lasso Regression
- **Purpose**: Predicts customer average spending based on purchase frequency
- **Features**: Name, AverageFrequency
- **Target**: AverageSpend
- **Algorithm**: Lasso Regression (alpha=1.0)
- **Serialization Date**: 2026-01-29T18:58:15.438000
- **Scikit-learn Version**: 1.3.0

## 🛠️ Model Pipeline Structure

All models use a scikit-learn Pipeline with the following preprocessing steps:

### **Preprocessing Pipeline**
1. **ColumnTransformer** for feature-specific transformations:
   - **Numerical Features**: Imputation + Scaling
     - `SimpleImputer` (median strategy)
     - `StandardScaler` (for Lasso models) or `passthrough` (for Linear Regression)
   - **Categorical Features**: Imputation + Encoding
     - `SimpleImputer` (constant strategy with "missing")
     - `OneHotEncoder` (first category drop, handle_unknown='ignore')

2. **Regressor**: Linear or Lasso regression

## 📊 Model Input Formats

### **For Ice Cream Sales Models:**
```json
{
  "input_data": {
    "columns": ["Date", "DayOfWeek", "Month", "Temperature", "Rainfall"],
    "index": [0, 1, 2],
    "data": [
      ["2025-06-15", "Sunday", "June", 75.5, 0.0],
      ["2025-06-16", "Monday", "June", 72.0, 0.1],
      ["2025-06-17", "Tuesday", "June", 78.2, 0.0]
    ]
  }
}
