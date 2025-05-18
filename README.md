# Global-Housing-Market-Price-Increase-Prediction

Global Housing Market Price Prediction
This project predicts whether house prices will increase in various countries based on economic and market features using a Logistic Regression model. It includes data preprocessing, exploratory data analysis (EDA), clustering (KMeans and DBSCAN), model training, and a Streamlit web app for interactive predictions.
Table of Contents

Project Overview
Features
Dataset
Installation
Usage
File Structure
Model Details
Streamlit App
Contributing
License

Project Overview
The Global Housing Market Price Prediction project analyzes housing market data from 20 countries (2015–2024) to predict price increases. It performs:

Data Cleaning: Handles missing values and outliers.
EDA: Visualizes distributions, correlations, and trends.
Clustering: Groups countries using KMeans and DBSCAN.
Classification: Trains multiple models, with Logistic Regression (SMOTE) selected as the best based on F1 score (~0.757).
Deployment: A Streamlit app allows users to input features and get predictions.

Features

Predicts house price increases using economic indicators (e.g., Rent Index, Mortgage Rate, GDP Growth).
Supports 20 countries with one-hot encoded country features.
Visualizes prediction probabilities, feature importance, and input feature distributions.
Includes KMeans and DBSCAN clustering for market segmentation.
Handles imbalanced data with SMOTE.

Dataset
The dataset (global_housing_market_extended.csv) contains yearly housing market data (2015–2024) for 20 countries, with features:

Numerical Features: House Price Index, Rent Index, Affordability Ratio, Mortgage Rate (%), Inflation Rate (%), GDP Growth (%), Population Growth (%), Urbanization Rate (%), Construction Index.
Derived Features: House Price Index_lag, Price_Increase (target).
Categorical Feature: Country.

Note: The dataset is assumed to be included in the repository. Ensure it is placed in the root directory.
Installation

Clone the Repository:
git clone https://github.com/your-username/global-housing-market-prediction.git
cd global-housing-market-prediction


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

The requirements.txt includes:
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
shap
xgboost
streamlit


Ensure Dataset:

Place global_housing_market_extended.csv in the root directory.



Usage
Training the Model

Run the training script to preprocess data, perform EDA, cluster, train models, and save the model/scaler:python train_model.py


This generates:
logistic_model.pkl: Fine-tuned Logistic Regression model.
scaler.pkl: StandardScaler for numerical features.



Running the Streamlit App

Start the Streamlit app:streamlit run streamlit_app.py


Open http://localhost:8501 in your browser.
Enter values for the 9 numerical features and select a country:
Rent Index
Affordability Ratio
Mortgage Rate (%)
Inflation Rate (%)
GDP Growth (%)
Population Growth (%)
Urbanization Rate (%)
Construction Index
House Price Index (Previous Year)
Country (dropdown)


Click Predict to view:
Prediction (Yes/No for price increase).
Probability of price increase/no increase.
Plots for probabilities, feature importance, and input features.



Example Input

Rent Index: 21.0
Affordability Ratio: 13.0
Mortgage Rate (%): 7.0
Inflation Rate (%): 10.0
GDP Growth (%): 3.0
Population Growth (%): 5.0
Urbanization Rate (%): 80.0
Construction Index: 100.0
House Price Index (Previous Year): 75.0
Country: Germany

File Structure
global-housing-market-prediction/
├── global_housing_market_extended.csv  # Dataset
├── train_model.py                     # Training script (data processing, EDA, clustering, modeling)
├── streamlit_app.py                   # Streamlit app for model deployment
├── logistic_model.pkl                 # Saved Logistic Regression model
├── scaler.pkl                         # Saved StandardScaler
├── requirements.txt                   # Dependencies
└── README.md                          # Project documentation

Model Details

Best Model: Logistic Regression (SMOTE) with test F1 score ~0.757.
Features:
Numerical: Rent Index, Affordability Ratio, Mortgage Rate (%), Inflation Rate (%), GDP Growth (%), Population Growth (%), Urbanization Rate (%), Construction Index, House Price Index_lag.
Categorical: Country (one-hot encoded, Australia as reference).


Preprocessing:
Missing values filled with country-wise means.
Outliers capped using IQR (1.0 multiplier).
Numerical features standardized with StandardScaler.
SMOTE applied to handle class imbalance.


Training:
Evaluated 9 models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, AdaBoost, XGBoost, SVM, KNN, Gradient Boosting) on original and SMOTE data.
Fine-tuned with GridSearchCV (e.g., C, penalty for Logistic Regression).


Clustering:
KMeans (3 clusters) and DBSCAN for market segmentation.



Streamlit App
The Streamlit app provides an interactive interface to:

Input 9 numerical features and select a country.
Display predictions and probabilities.
Visualize:
Bar plot of prediction probabilities.
Feature importance (Logistic Regression coefficients).
Input feature values.



Troubleshooting:

Ensure logistic_model.pkl and scaler.pkl are in the same directory as streamlit_app.py.
If feature order errors occur, verify numerical_features in streamlit_app.py matches continuous_cols in train_model.py.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.

Please ensure code follows PEP 8 and includes documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Built with Python, scikit-learn, and Streamlit. For questions, open an issue or contact [arpitsingh4297@gmail.com].
