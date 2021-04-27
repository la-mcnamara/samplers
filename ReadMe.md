# ML Sampler Details

## HeartFailure.ipynb
This notebook analyzes the Kaggle heart failure dataset here:
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

This analysis fits various machine learning models to the dataset (logistic regression, random forest, xgboost) and uses cross-validation to explore which of these maximizes model performance metrics.

## ML_Sampler.py
This script analyzes the classic iris dataset as follows: 
<li>fits basic linear regression models predicting sepal_length using sklearn and statsmodels
<li>fits logistic regression model using sklearn recursive feature elimination (RFE) to identify best subset of features
<li>fits random forest model for the multiclass problem of predicting which of 3 iris species is represented by the data

The data is sourced from: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv