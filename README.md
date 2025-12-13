# UKMSRP: Multimodal Stroke Risk Prediction

We developed a unified machine learning framework termed UKMSRP for long-term stroke risk prediction based on multimodal data from the UK Biobank. This framework integrates demographic variables, biochemical measurements, proteomic features, and other high-dimensional biomarkers to construct and evaluate predictive models for incident stroke. 
It provides a complete and modular pipeline for:

# Requirements
The main requirements are listed below:
Python ≥ 3.8
NumPy
Pandas
Scikit-learn
XGBoost
SciPy
Matplotlib
Seaborn
SHAP
Lifelines 
Joblib
# Description of the UKMSRP Source Code
xgboost.ipynb

The code implements an XGBoost-based prediction framework for stroke risk, including model training, internal cross-validation in the England cohort, and independent external validation in the Scotland and Wales cohorts.
feature_selection.ipynb

The code performs feature clustering, importance ranking, and DeLong-based feature selection using XGBoost models to identify the most informative biomarkers for stroke risk prediction.
cox_proportional_hazards_regression_models.ipynb

The code conducts large-scale univariate Cox proportional hazards regression with parallel computing and FDR correction to identify features significantly associated with time-to-event outcomes.
