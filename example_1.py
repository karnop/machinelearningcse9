import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 10 classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# Step 1: Generate synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
    weights=[0.85, 0.15], flip_y=0.01, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Step 2: Define models dictionary
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "NaiveBayes": GaussianNB()
}

# Step 3: Set MLflow experiment
mlflow.set_experiment("Multiple Models Experiment")

# Step 4: Train and log all models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log params (if possible)
        try:
            mlflow.log_params(model.get_params())
        except:
            pass  # Some models like Pipeline or NaiveBayes may not have .get_params()

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"]
        })

        # Log the model
        mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)

        print(f"Logged: {model_name} with accuracy = {acc:.4f}")
