# Vendor Invoice Intelligence System
🔗 Live Application: https://vendor-invoice-intelligence-system.onrender.com

## Overview
* Data science-driven system designed for intelligent vendor invoice analysis
* Automates freight cost prediction and invoice risk detection
* Replaces manual validation with scalable, data-driven decision making

## Problem Statement
* Manual invoice verification is time-consuming and error-prone
* Freight estimation lacks consistency
* Risky or anomalous invoices often go undetected

## Solution Approach
* Applied supervised machine learning techniques on structured invoice data
* Designed an end-to-end pipeline from data extraction to prediction
* Ensured feature consistency between training and inference

## Freight Cost Prediction (Regression)
* Predicts freight cost using:
  * Quantity
  * Invoice Dollars
* Models implemented:
  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
* Evaluation metrics:
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * R² Score
* Best model selected based on minimum error and stability
<img width="1690" height="852" alt="Screenshot 2026-04-13 213707" src="https://github.com/user-attachments/assets/d80f95db-e5c0-47b9-a830-3092b305ab3f" />

## Invoice Risk Detection (Classification)
* Identifies potentially risky invoices
* Feature inputs include:
  * Invoice quantity & value
  * Freight cost
  * Aggregated purchase metrics
* Model used:
  * Random Forest Classifier
* Optimization:
  * Hyperparameter tuning using GridSearchCV
  * F1-score used for balanced performance
* Output:
  * Safe Invoice
  * Manual Review Required
<img width="1469" height="841" alt="Screenshot 2026-04-13 213749" src="https://github.com/user-attachments/assets/792c7b3c-20b3-425a-9403-f2a7a189ef70" />

## Data Processing & Feature Engineering
* Data extracted from SQLite database
* Feature transformations using Pandas
* Handling missing values and inconsistencies
* Derived features such as delays and aggregated metrics

## Machine Learning Concepts Applied
* Regression Analysis
* Classification Modeling
* Feature Engineering
* Model Evaluation & Validation
* Hyperparameter Tuning
* Handling real-world noisy data
* End-to-end ML pipeline design

## Tools & Technologies
* Python
* Pandas, NumPy
* Scikit-learn
* SQLite
* Joblib

## Key Highlights
* Built a complete ML pipeline from raw data to deployment-ready predictions
* Demonstrates strong understanding of applied data science
* Focus on real-world business problem solving
* Optimized model performance using proper evaluation techniques

## Use Cases
* Freight cost optimization
* Invoice anomaly detection
* Financial risk analysis
* Vendor performance monitoring

## Author
Priya
[priyaax21@gmail.com
B.Tech CSE 3rd Year
