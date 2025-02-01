import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # To handle SMOTE within cross-validation
from scipy.stats import randint, uniform  # For defining hyperparameter distributions

def train_model(data: pd.DataFrame):
    """
    Train an XGBoost model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target variable.

    Returns:
        xgb.XGBClassifier: Trained and tuned XGBoost model.
    """
    # Split dataset into features (X) and target (y)
    X = data.drop(columns=["close_approach"])
    y = data["close_approach"]

    # Split into training and validation sets (stratified to maintain class distribution)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define the pipeline with SMOTE and XGBoost
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),  # Oversample the minority class
        ("xgb", xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr")),
    ])

    # Define hyperparameter distributions for RandomizedSearchCV
    param_dist = {
        "xgb__max_depth": randint(3, 10),  # Range for max_depth
        "xgb__learning_rate": uniform(0.01, 0.3),  # Range for learning_rate
        "xgb__subsample": uniform(0.6, 0.4),  # Range for subsample
        "xgb__colsample_bytree": uniform(0.6, 0.4),  # Range for colsample_bytree
        "xgb__lambda": uniform(0, 2),  # Range for L2 regularization (lambda)
        "xgb__alpha": uniform(0, 2),  # Range for L1 regularization (alpha)
        "xgb__n_estimators": randint(50, 200),  # Range for n_estimators
        "xgb__scale_pos_weight": [None, (y_train == 0).sum() / (y_train == 1).sum()],  # Optional
    }

    # Perform Randomized Search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings to sample
        scoring="f1",  # Use F1-score for imbalanced datasets
        cv=5,  # 5-fold cross-validation
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
        verbose=2,
    )

    # Fit the Randomized Search on the training data
    random_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    threshold = 0.2  # Lower the threshold to increase recall
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Recalculate metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"Adjusted Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return best_model


if __name__ == "__main__":
    # Load the dataset
    # data = pd.read_csv("datasets/tle495510.csv")
    data = pd.read_csv("datasets/tle19900.csv")

    # Train the model with hyperparameter tuning
    model = train_model(data)

    # 