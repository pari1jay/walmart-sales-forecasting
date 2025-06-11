# 🛒 Walmart Sales Forecasting

## Overview

This project aims to build machine learning models to forecast **weekly sales** across 45 Walmart stores. Using historical sales data, store-specific attributes, and external economic indicators, the project helps Walmart better understand and anticipate sales trends. The outcome is focussed on improved inventory management, pricing strategy, and marketing efforts.

---

## Team Members

- Parimala Anjanappa  
- Sree Surya Sreemanth Yedidhi  
- Vivek Chandra  
- Yagnapraseeda Atmuri  

---

## Defining  Business Problem

Walmart, a global retail giant, needs accurate weekly sales forecasts for each store to:

- Optimize inventory levels
- Improve resource/staff allocation
- Plan holiday and promotional markdown campaigns
- Adapt to external economic and seasonal variations like holiday season

---

## Datasets Used

All datasets were sourced from the Kaggle competition:  
[Walmart Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

1. **train.csv**  
   Historical weekly sales for 45 Walmart stores (2010–2012)  
   - `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`

2. **stores.csv**  
   Store metadata  
   - `Store`, `Type`, `Size`, `City`, `State`, `Zip`

3. **features.csv**  
   External and economic factors  
   - `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `MarkDown1-5`

---

## Data Preprocessing

- **Missing Values**:  
  - `MarkDown1-5`: Replaced with 0 (indicates no promotion)  
  - `CPI` & `Unemployment`: Imputed with median values  

- **Feature Engineering**:  
  - `Total_MarkDown` = Sum of `MarkDown1-5`  
  - Extracted `Year`, `Month`, and `Week` from `Date`  

- **Outlier Handling**:  
  - Removed negative and extreme `Weekly_Sales` values using Z-score and boxplot  

- **Encoding & Scaling**:  
  - One-Hot Encoding for categorical features (`Store`, `Dept`, `Type`)  
  - MinMaxScaler applied to numerical features  

- **Feature Selection**:  
  - Recursive Feature Elimination (RFE) using Random Forest Regressor  
  - Selected top features: `Temperature`, `Fuel_Price`, `Unemployment`, `CPI`, `Total_MarkDown`, `Month`, etc.

---

## Exploratory Data Analysis (EDA)

- Departments 92 and 95 had the highest average sales  
- Store 20 was the top-performing store  
- Sales were higher between 40°F–80°F and during holidays  
- Higher unemployment and fuel prices correlated with lower sales  
- CPI showed mixed influence depending on store type  

---

## Models Built - below is the brief summary of our model results

| Model                    | R² Score | RMSE   | MAE    |
|-------------------------|----------|--------|--------|
| Linear Regression        | 0.9086   | 0.0731 | 0.0407 |
| Random Forest Regressor  | 0.9120   | 0.0720 | 0.0400 |
| Decision Tree Regressor  | 0.9360   | 0.0610 | 0.0350 |
| Gradient Boosted Regressor | **0.9361** | **0.0605** | **0.0342** |
| Deep Neural Network (Keras) | 0.8451   | ~      | ~      |

 **Best Performing Model**: Gradient Boosted Tree Regressor  
 DNN had competitive performance but lower accuracy than tree-based models

---

## Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Keras (for DNN)
- PySpark (limited model support used for preprocessing in Spark)
- Kaggle Datasets

---

## Challenges Faced

- Spark limitations on Z-score functions led to alternate outlier handling (boxplot)
- PySpark had limited support for models like XGBoost or KNN
- Missing markdown data impacted certain time periods

---

## Key Takeaways

- Holiday periods and markdown promotions significantly drive sales
- External factors (fuel price, temperature, unemployment) affect customer behavior
- Gradient Boosted Trees offer high predictive accuracy and flexibility for this use case
- Deep learning models require more tuning to match or exceed traditional ML performance

---



