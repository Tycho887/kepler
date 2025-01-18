import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(data: pd.DataFrame, hyperparameters: dict):
    """
    Train an XGBoost model using the given hyperparameters and track accuracy during training.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target variable.
        hyperparameters (dict): Hyperparameters for the XGBoost model.

    Returns:
        xgb.Booster: Trained XGBoost model.
    """
    # Split dataset into features (X) and target (y)
    X = data.drop(columns=["close_approach"])
    y = data["close_approach"]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate scale_pos_weight for class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # Update hyperparameters with scale_pos_weight and eval_metric
    hyperparameters["scale_pos_weight"] = scale_pos_weight

    # Train the model with the given hyperparameters
    model = xgb.XGBClassifier(**hyperparameters)
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # Evaluate the final model accuracy on the validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # Rate of false-negatives

    false_negatives = np.sum((y_val == 1) & (y_pred == 0))
    false_positives = np.sum((y_val == 0) & (y_pred == 1))
    true_positives = np.sum((y_val == 1) & (y_pred == 1))
    true_negatives = np.sum((y_val == 0) & (y_pred == 0))

    print(f"Rate of false-negatives: {false_negatives:.4f}")
    print(f"Rate of false-positives: {false_positives:.4f}")
    print(f"Rate of true-positives: {true_positives:.4f}")
    print(f"Rate of true-negatives: {true_negatives:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    return model



if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("datasets/tle11175.csv")

    hyperparameters = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 1.0,
        "n_estimators": 100,
        "eval_metric": "recall",  # Set eval_metric here
    }

    # Train the model
    model = train_model(data, hyperparameters)
