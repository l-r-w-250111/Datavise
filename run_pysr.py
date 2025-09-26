import pandas as pd
import joblib
import sys
import json
from pysr import PySRRegressor

# Helper function for custom operator
def inv(x):
    return 1/x

def run_pysr(input_path, output_path, pysr_params_json, task_type='regression'):
    """
    Loads data, runs PySR, and saves the trained model.
    """
    print(f"PySR subprocess started. Loading data from {input_path}...")

    # Load data
    data = pd.read_csv(input_path)
    X = data.drop(columns=['target'])
    y = data['target']

    print(f"Data loaded. Initializing PySR for {task_type}...")

    # Load UI parameters
    pysr_params = json.loads(pysr_params_json)

    # Define stable default parameters
    stable_params = {
        "procs": 0,
        "progress": False, # Keep False to be safe with stdout capturing
        "binary_operators": ["+", "*"],
        "unary_operators": ["cos", "exp", "sin", "inv(x) = 1/x"],
        "extra_sympy_mappings": {"inv": inv},
        "temp_equation_file": True,
    }

    # Set default loss based on task type
    if task_type == 'classification':
        stable_params["elementwise_loss"] = "SigmoidLoss()"
    else: # Regression
        stable_params["elementwise_loss"] = "loss(prediction, target) = (prediction - target)^2"


    # Merge UI params into the defaults
    final_params = {**stable_params, **pysr_params}

    model = PySRRegressor(**final_params)

    print("Fitting PySR model...")
    model.fit(X, y)
    print("Model fitting complete.")

    # Save the trained model
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_pysr.py <input_path> <output_path> <pysr_params_json> <task_type>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pysr_params_json = sys.argv[3]
    task_type = sys.argv[4]

    run_pysr(input_path, output_path, pysr_params_json, task_type)