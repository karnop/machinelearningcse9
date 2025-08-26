# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Optional: Install mlflow (uncomment if not installed)
# !pip install mlflow

import mlflow
import mlflow.sklearn

# ✅ DO NOT use file:// or localhost URI for local tracking in Windows
# ✅ Simply let MLflow use the default path `./mlruns`
mlflow.set_experiment("First Experiment")  # This will create the experiment if not exists

# Step 1: Create a synthetic imbalanced binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=8,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Define model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train logistic regression model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Evaluate model
y_pred = lr.predict(X_test)
report = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

print(report)

# ✅ Start MLflow logging
with mlflow.start_run():
    # Log model parameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })

    # Log model artifact
    mlflow.sklearn.log_model(lr, "Logistic_Regression_Model")

print("Run complete. Open http://127.0.0.1:5000 to view results.")
