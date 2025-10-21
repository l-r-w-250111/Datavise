# AI-Powered Prediction Simulator

This is an interactive web application built with Streamlit that allows users to upload their own datasets, get model recommendations from a local AI, and run machine learning simulations for both regression and classification tasks. The application also supports time series forecasting with models like LSTM and Prophet.

## Features

- **Interactive UI:** A user-friendly web interface powered by Streamlit.
- **Multiple Prediction Modes:** Supports `Tabular Prediction` (for regression/classification), `Time Series Forecasting`, and a `PINN Solver`.
- **AI-Powered Model Suggestions:** Integrates with a local Large Language Model (LLM) via Ollama to suggest appropriate models for tabular data or to generate differential equations from natural language for the PINN solver.
- **Flexible Problem Types:** Users can manually select whether their tabular task is `Regression` or `Classification`.
- **Automated Preprocessing:** Automatically handles categorical features and label encoding for classification tasks.
- **Hyperparameter Tuning:**
  - **Manual Tuning:** Adjust model hyperparameters directly in the UI.
  - **Bayesian Optimization:** Automatically find the best parameters for your models.
- **Model Evaluation & Comparison:**
  - View performance metrics (R² for regression, Accuracy for classification).
  - Compare multiple models with clear charts and tables.
- **In-Depth Analysis:**
  - **Feature Importance:** Visualize the most influential features for a given model.
  - **SHAP Analysis:** Explain model predictions and understand the rationale behind them.
  - **Symbolic Models:** For symbolic regressors like `gplearn` and `PySR`, the discovered mathematical formula is displayed.
- **Downloadable Models:** Download trained tabular models (`.joblib`) or PINN models (`.keras`) for offline use.

## How to Use the PINN Solver

The PINN (Physics-Informed Neural Network) solver allows you to find solutions to user-defined ordinary differential equations (ODEs).

1.  **Select the Mode:** Choose "PINN Solver" from the sidebar.
2.  **Define the Equation:**
    -   **Manually:** Enter the differential equation in its residual form (i.e., the part that equals zero) in the text box.
        -   Use `y` for the function `y(x)`.
        -   Use `dy/dx` for the first derivative and `d2y/dx2` for the second derivative.
        -   Example: For `dy/dx = -y`, the residual is `dy/dx + y`.
    -   **AI Suggestion:** Alternatively, describe your physical problem in the text area under "Or, get a suggestion from the AI" and click "Suggest Equation". The AI will generate the equation for you.
3.  **Upload Condition Data:**
    -   Create a CSV file with two columns: `x` and `y`.
    -   These points represent the known initial or boundary conditions for your problem. For example, for an initial condition `y(0) = 1`, your CSV would contain `0,1`.
    -   Upload this file using the file uploader.
4.  **Train the Model:**
    -   Adjust hyperparameters like epochs, layers, and neurons as needed.
    -   Click "Train PINN Model". The application will train the neural network to satisfy both the ODE and your provided condition data.
5.  **Analyze and Save:**
    -   Once training is complete, you will see a plot of the solution.
    -   You can download the trained model using the "Download Trained Model" button. You can later re-upload this `.keras` file in the "Load Existing Model" section to make further predictions without retraining.

## Setup and Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running for AI model suggestions.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure Ollama is running:**
    Make sure the Ollama service is active and has a model available (e.g., `llama3`). You can pull a model with:
    ```bash
    ollama pull llama3
    ```

## How to Run the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    > **Note:** The first time you run the application, it may take a minute or two to load. This is because large libraries like TensorFlow and PySR need to be initialized. Subsequent loads will be much faster.

2.  **Open your web browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Configuration

The application can be configured by editing the `settings.json` file. This allows you to change the endpoint for the Ollama API and specify which large language model to use for generating suggestions.

-   `ollama_api_url`: The full URL for the Ollama API's generation endpoint.
-   `llm_model`: The name of the Ollama model to use (e.g., `llama3`, `gemma3:12b`).

**Example `settings.json`:**
```json
{
  "ollama_api_url": "http://localhost:11434/api/generate",
  "llm_model": "gemma3:12b"
}
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

For information on the licenses of the third-party libraries used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.
