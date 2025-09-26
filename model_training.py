import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib
import subprocess
import sys
import os
import json
import uuid

# Supported models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from pysr import PySRRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.utils.validation import check_X_y

# Helper function for PySR custom operator
def inv(x):
    return 1/x

# --- Monkey-patch for gplearn ---
# gplearn is not fully compatible with scikit-learn >= 1.0. This is a workaround.
# It adds the missing _validate_data method and n_features_in_ attribute.
def _validate_data(self, X, y, reset=True, validate_separately=False, **check_params):
    if y is None:
        raise ValueError("y must be an array or a list.")
    # In gplearn, SymbolicRegressor's y can be continuous, and SymbolicClassifier's y is discrete.
    # We'll rely on the existing check_X_y to handle this.
    X, y = check_X_y(X, y, y_numeric=True if isinstance(self, SymbolicRegressor) else False)
    self.n_features_in_ = X.shape[1] # Set the number of features
    return X, y

SymbolicRegressor._validate_data = _validate_data
SymbolicClassifier._validate_data = _validate_data
# --- End of monkey-patch ---


# Map model names to their classes, types, and hyperparameters
MODEL_MAP = {
    # Regression
    "LinearRegression": {
        "class": LinearRegression, "type": "Regression", "params": {}
    },
    "RandomForestRegressor": {
        "class": RandomForestRegressor, "type": "Regression",
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10},
            "max_depth": {"type": "int", "min": 1, "max": 50, "default": 10, "step": 1},
        }
    },
    "DecisionTreeRegressor": {
        "class": DecisionTreeRegressor, "type": "Regression",
        "params": {
            "max_depth": {"type": "int", "min": 1, "max": 50, "default": 5, "step": 1},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1},
        }
    },
    "GradientBoostingRegressor": {
        "class": GradientBoostingRegressor, "type": "Regression",
        "params": {
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
        }
    },
    "LGBMRegressor": {
        "class": LGBMRegressor, "type": "Regression",
        "params": {
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
            "num_leaves": {"type": "int", "min": 10, "max": 100, "default": 31, "step": 1},
        }
    },
    "SymbolicRegressor(gplearn)": {
        "class": SymbolicRegressor, "type": "Regression",
        "params": {
            "population_size": {"type": "int", "min": 100, "max": 2000, "default": 1000, "step": 100},
            "generations": {"type": "int", "min": 10, "max": 100, "default": 20, "step": 5},
        }
    },
    "SymbolicRegressor(PySR)": {
        "class": PySRRegressor, "type": "Regression",
        "params": {
            "niterations": {"type": "int", "min": 5, "max": 100, "default": 40, "step": 5},
            "maxsize": {"type": "int", "min": 10, "max": 50, "default": 20, "step": 1},
        }
    },
    # Classification
    "LogisticRegression": {
        "class": LogisticRegression, "type": "Classification", "params": {}
    },
    "RandomForestClassifier": {
        "class": RandomForestClassifier, "type": "Classification",
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 10},
            "max_depth": {"type": "int", "min": 1, "max": 50, "default": 10, "step": 1},
        }
    },
    "DecisionTreeClassifier": {
        "class": DecisionTreeClassifier, "type": "Classification",
        "params": {
            "max_depth": {"type": "int", "min": 1, "max": 50, "default": 5, "step": 1},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1},
        }
    },
    "GradientBoostingClassifier": {
        "class": GradientBoostingClassifier, "type": "Classification",
        "params": {
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
        }
    },
    "LGBMClassifier": {
        "class": LGBMClassifier, "type": "Classification",
        "params": {
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100, "step": 10},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
            "num_leaves": {"type": "int", "min": 10, "max": 100, "default": 31, "step": 1},
        }
    },
    "SymbolicClassifier(gplearn)": {
        "class": SymbolicClassifier, "type": "Classification",
        "params": {
            "population_size": {"type": "int", "min": 100, "max": 2000, "default": 1000, "step": 100},
            "generations": {"type": "int", "min": 10, "max": 100, "default": 20, "step": 5},
        }
    },
    "SymbolicClassifier(PySR)": {
        "class": PySRRegressor, "type": "Classification",
        "params": {
            "niterations": {"type": "int", "min": 5, "max": 100, "default": 40, "step": 5},
            "maxsize": {"type": "int", "min": 10, "max": 50, "default": 20, "step": 1},
        }
    },
}

def train_and_evaluate(df: pd.DataFrame, target_column: str, model_name: str, problem_type: str, params: dict):
    """
    Trains a model with specified hyperparameters and evaluates it.

    Args:
        df: The input DataFrame.
        target_column: The name of the target variable column.
        model_name: The name of the model to train.
        problem_type: 'Regression' or 'Classification'.
        params: A dictionary of hyperparameters for the model.

    Returns:
        A tuple containing:
        - score (float): The evaluation score.
        - model: The trained model object.
        - X_test: The test features.
        - y_test: The test target variable.
        - predictions: The model's predictions on the test set.
        - error (str): An error message, if any.
    """
    if model_name not in MODEL_MAP:
        return None, None, None, None, None, f"Error: Model '{model_name}' is not supported."

    if MODEL_MAP[model_name]["type"] != problem_type:
        return None, None, None, None, None, f"Error: Model '{model_name}' is a {MODEL_MAP[model_name]['type']} model, but the problem is {problem_type}."

    try:
        # 1. Prepare data
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Simple preprocessing: one-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train model
        model_class = MODEL_MAP[model_name]["class"]

        # For PySR, run as an external process to avoid deadlocks with Streamlit
        if "PySR" in model_name:
            # Create unique temp file paths
            run_id = uuid.uuid4()
            input_path = f"temp_pysr_input_{run_id}.csv"
            output_path = f"temp_pysr_model_{run_id}.joblib"

            try:
                # Save training data to a temporary CSV
                train_df = pd.DataFrame(X_train, columns=X.columns)
                train_df['target'] = y_train
                train_df.to_csv(input_path, index=False)

                # Execute the external script
                command = [
                    sys.executable,
                    "run_pysr.py",
                    input_path,
                    output_path,
                    json.dumps(params),
                    problem_type.lower()
                ]
                # Using capture_output=True to get stdout/stderr
                result = subprocess.run(command, check=True, capture_output=True, text=True)

                # Load the trained model from the file
                model = joblib.load(output_path)

            except subprocess.CalledProcessError as e:
                # Provide detailed error information if the subprocess fails
                error_message = f"PySR subprocess failed.\nExit Code: {e.returncode}\n"
                error_message += f"Stdout: {e.stdout}\n"
                error_message += f"Stderr: {e.stderr}"
                return None, None, None, None, None, error_message
            finally:
                # Clean up temporary files
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
        else:
            model = model_class(**params)
            model.fit(X_train, y_train)

        # 4. Evaluate model
        predictions = model.predict(X_test)

        if problem_type == "Regression":
            score = r2_score(y_test, predictions)
        else: # Classification
            score = accuracy_score(y_test, predictions)

        return score, model, X_test, y_test, predictions, None

    except Exception as e:
        return None, None, None, None, None, f"An error occurred during training: {e}"

def find_best_params(df: pd.DataFrame, target_column: str, model_name: str, problem_type: str):
    """
    Finds the best hyperparameters for a model using Bayesian Optimization.

    Args:
        df: The input DataFrame.
        target_column: The name of the target variable column.
        model_name: The name of the model to train.
        problem_type: 'Regression' or 'Classification'.

    Returns:
        A tuple containing:
        - score (float): The evaluation score of the best model on the test set.
        - model: The best trained model object.
        - X_test: The test features.
        - y_test: The test target variable.
        - predictions: The best model's predictions on the test set.
        - error (str): An error message, if any.
    """
    if model_name not in MODEL_MAP or not MODEL_MAP[model_name]["params"]:
        return None, None, None, None, None, f"Error: Model '{model_name}' is not configured for hyperparameter search."

    try:
        # 1. Prepare data
        y = df[target_column]
        X = df.drop(columns=[target_column])
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Define search space
        search_spaces = {}
        for param, details in MODEL_MAP[model_name]["params"].items():
            if details["type"] == "int":
                search_spaces[param] = Integer(details["min"], details["max"])
            elif details["type"] == "float":
                search_spaces[param] = Real(details["min"], details["max"], "log-uniform")

        # 3. Configure and run Bayesian Search
        model_class = MODEL_MAP[model_name]["class"]
        opt = BayesSearchCV(
            estimator=model_class(),
            search_spaces=search_spaces,
            n_iter=20,  # Number of iterations
            cv=3,       # Number of cross-validation folds
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        opt.fit(X_train, y_train)

        # 4. Evaluate the best found model on the hold-out test set
        best_model = opt.best_estimator_
        predictions = best_model.predict(X_test)

        if problem_type == "Regression":
            score = r2_score(y_test, predictions)
        else: # Classification
            score = accuracy_score(y_test, predictions)

        return score, best_model, X_test, y_test, predictions, None

    except Exception as e:
        return None, None, None, None, None, f"An error occurred during Bayesian optimization: {e}"
