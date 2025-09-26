# AI-Powered Prediction Simulator

This is an interactive web application built with Streamlit that allows users to upload their own datasets, get model recommendations from a local AI, and run machine learning simulations for both regression and classification tasks. The application also supports time series forecasting with models like LSTM and Prophet.

## Features

- **Interactive UI:** A user-friendly web interface powered by Streamlit.
- **Multiple Prediction Modes:** Supports both `Tabular Prediction` (for regression/classification) and `Time Series Forecasting`.
- **AI-Powered Model Suggestions:** Integrates with a local Large Language Model (LLM) via Ollama to suggest appropriate models based on your data.
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
- **Downloadable Models:** Download the trained model (`.joblib` file) for offline use.

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
    streamlit run src/app.py
    ```

2.  **Open your web browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Configuration

The application can be configured by editing the `settings.json` file. This allows you to change the endpoint for the Ollama API and specify which large language model to use for generating suggestions.

-   `ollama_api_url`: The full URL for the Ollama API's generation endpoint.
-   `llm_model`: The name of the Ollama model to use (e.g., `llama3`, `gemma:7b`).

**Example `settings.json`:**
```json
{
  "ollama_api_url": "http://localhost:11434/api/generate",
  "llm_model": "gemma:7b"
}
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

For information on the licenses of the third-party libraries used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.
