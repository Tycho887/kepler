import pandas as pd
import ML as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna

def optimize_xgb(data: pd.DataFrame):
    """
    Perform hyperparameter optimization for XGBoost using Optuna.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target variable.

    Returns:
        dict: Best hyperparameters found during optimization.
    """
    # Split dataset into features (X) and target (y)
    X = data.drop(columns=["close_approach"])
    y = data["close_approach"]

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def objective(trial):
        """
        Objective function for Optuna optimization.

        Parameters:
            trial: Optuna trial object.

        Returns:
            float: Validation accuracy (negative for minimization).
        """
        # Define the parameter search space
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        }

        # Train the model with the suggested parameters
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=10,
            verbose=False
        )

        # Predict on the validation set
        y_pred = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)  # Run for 50 trials or 10 minutes

    # Output the best parameters
    print("Best parameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    return study.best_params

# Example usage (assuming `dataset` is already prepared):
# optimal_params = optimize_xgboost(dataset)
# print(optimal_params)
