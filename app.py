import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
from llm_integration import create_prompt, get_llm_suggestions, create_pinn_prompt
from model_training import train_and_evaluate, find_best_params, MODEL_MAP
from ts_model_training import train_and_predict_lstm, train_and_predict_prophet, predict_future_lstm
from pinn_solver import PINNSolver
from prophet.plot import plot_plotly
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix

# Helper function for SHAP plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide")
    st.title("🧠 AI-Powered Prediction Simulator")

    st.write("""
    Upload data, get AI model suggestions, run simulations, compare results, and download the best model.
    """)

    # --- Initialize session state ---
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = ""
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = {}
    if 'prophet_model' not in st.session_state:
        st.session_state.prophet_model = None
    if 'prophet_regressors' not in st.session_state:
        st.session_state.prophet_regressors = []
    if 'lstm_model' not in st.session_state:
        st.session_state.lstm_model = None
    if 'lstm_scaler' not in st.session_state:
        st.session_state.lstm_scaler = None
    if 'lstm_sequence_length' not in st.session_state:
        st.session_state.lstm_sequence_length = 0
    if 'lstm_results_df' not in st.session_state:
        st.session_state.lstm_results_df = None
    if 'pinn_solver' not in st.session_state:
        st.session_state.pinn_solver = None
    if 'pinn_history' not in st.session_state:
        st.session_state.pinn_history = None
    if 'pinn_condition_data' not in st.session_state:
        st.session_state.pinn_condition_data = None


    # --- Mode Selection ---
    prediction_mode = st.sidebar.selectbox(
        "Select Prediction Mode",
        ["Tabular Prediction", "Time Series Forecasting", "PINN Solver"]
    )
    st.sidebar.write("---")

    if prediction_mode == "Tabular Prediction":
        run_tabular_prediction()
    elif prediction_mode == "Time Series Forecasting":
        run_time_series_forecasting()
    elif prediction_mode == "PINN Solver":
        run_pinn_solver()

def run_tabular_prediction():
    st.header("Tabular Prediction")
    uploaded_file = st.file_uploader("1. Upload your CSV data", type="csv", key="tabular_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            st.subheader("2. Select the column to predict")
            columns = df.columns.tolist()
            if not columns:
                st.warning("The uploaded CSV file has no columns.")
                return

            target_column = st.selectbox("Which column to predict?", options=columns, key='target_column_selector')
            st.info(f"You selected to predict: **{target_column}**")
            st.subheader("Data Preview"); st.dataframe(df.head())

            # --- Step 3: Get Model Suggestions ---
            st.subheader("3. Get Model Suggestions from AI")

            # Allow user to select the problem type
            problem_type = st.selectbox(
                "Select the problem type",
                ["Regression", "Classification"]
            )
            st.write(f"**Selected Problem Type:** {problem_type}")

            user_comment = st.text_area(
                "Enter any comments about the data characteristics (optional)",
                placeholder="e.g., The data has daily and yearly seasonality due to weather conditions."
            )

            if st.button("Suggest Models"):
                with st.spinner("Asking AI... (ensure Ollama is running)"):
                    available_models_for_prompt = [
                        name for name, details in MODEL_MAP.items() if details["type"] == problem_type
                    ]
                    prompt = create_prompt(
                        column_names=df.columns.drop(target_column).tolist(),
                        target_column=target_column,
                        problem_type=problem_type,
                        available_models=available_models_for_prompt,
                        user_comment=user_comment
                    )
                    st.session_state.suggestions = get_llm_suggestions(prompt)
                    st.session_state.results = []
                    st.session_state.trained_models = {}
                    st.session_state.prediction_data = {}

            if st.session_state.suggestions:
                st.text_area("Raw AI Suggestions", st.session_state.suggestions, height=100)

                st.subheader("4. Run Simulation")
                all_available_models = [
                    name for name, details in MODEL_MAP.items() if details["type"] == problem_type
                ]
                llm_suggested_models = [name.strip() for name in st.session_state.suggestions.split(',')]

                validated_suggestions = [name for name in llm_suggested_models if name in all_available_models]

                if not all_available_models:
                    st.warning("No models are configured for this problem type.")
                else:
                    selected_models = st.multiselect(
                        "Choose models to run (AI suggestions are pre-selected)",
                        options=all_available_models,
                        default=validated_suggestions if validated_suggestions else all_available_models[:1],
                        key=f"multiselect_{st.session_state.suggestions}"
                    )

                    hyperparams = {}
                    with st.expander("Advanced Settings"):
                        # Disable Bayesian Optimization for incompatible models
                        is_symbolic_selected = any('Symbolic' in m for m in selected_models)
                        if is_symbolic_selected:
                            st.warning("Bayesian Optimization is not compatible with Symbolic models (gplearn, PySR) and has been disabled for this selection.")

                        use_bayesian_opt = st.checkbox(
                            "Find best parameters with Bayesian Optimization",
                            key="bayesian_opt_checkbox",
                            help="If checked, the app will automatically search for the best hyperparameters. This may take longer.",
                            disabled=is_symbolic_selected
                        )
                        st.write("---")
                        st.write("Manual Hyperparameter Tuning (disabled if Bayesian Optimization is active):")

                        for model_name in selected_models:
                            model_info = MODEL_MAP.get(model_name, {})
                            if model_info["params"]:
                                st.write(f"**Parameters for {model_name}**")
                                hyperparams[model_name] = {}
                                for param_name, param_details in model_info["params"].items():
                                    is_disabled = use_bayesian_opt
                                    if param_details["type"] == "int":
                                        hyperparams[model_name][param_name] = st.slider(
                                            f"{param_name} ({model_name})",
                                            min_value=param_details["min"], max_value=param_details["max"],
                                            value=param_details["default"], step=param_details["step"],
                                            key=f"{model_name}_{param_name}", disabled=is_disabled
                                        )
                                    elif param_details["type"] == "float":
                                        hyperparams[model_name][param_name] = st.slider(
                                            f"{param_name} ({model_name})",
                                            min_value=param_details["min"], max_value=param_details["max"],
                                            value=param_details["default"], step=param_details["step"],
                                            format="%.2f", key=f"{model_name}_{param_name}", disabled=is_disabled
                                        )

                    if st.button(f"Run Simulation for {len(selected_models)} models"):
                        st.session_state.results = []
                        st.session_state.trained_models = {}
                        st.session_state.prediction_data = {}
                        progress_bar = st.progress(0)

                        for i, model_name in enumerate(selected_models):
                            if use_bayesian_opt:
                                with st.spinner(f"Running Bayesian Optimization for {model_name}... This will take a moment."):
                                    score, model, X_test, y_test, predictions, error = find_best_params(
                                        df, target_column, model_name, problem_type
                                    )
                            else:
                                spinner_message = f"Training {model_name} with specified parameters..."
                                if "PySR" in model_name:
                                    spinner_message = "Running PySR in a separate process. This may take several minutes. Please wait for completion..."

                                with st.spinner(spinner_message):
                                    model_params = hyperparams.get(model_name, {})
                                    score, model, X_test, y_test, predictions, error = train_and_evaluate(
                                        df, target_column, model_name, problem_type, model_params
                                    )

                            if error:
                                st.error(f"Error processing {model_name}: {error}")
                            else:
                                model_display_name = f"{model_name} (Bayesian Opt.)" if use_bayesian_opt else model_name
                                st.session_state.results.append({"Model": model_display_name, "Score": score})
                                st.session_state.trained_models[model_display_name] = model
                                st.session_state.prediction_data[model_display_name] = {
                                    "X_test": X_test, "y_test": y_test, "predictions": predictions
                                }
                            progress_bar.progress((i + 1) / len(selected_models))

            if st.session_state.results:
                st.subheader("5. Compare Results & Perform Predictions")
                results_df = pd.DataFrame(st.session_state.results)
                metric_label = "R² Score" if problem_type == "Regression" else "Accuracy"
                results_df = results_df.rename(columns={"Score": metric_label}).sort_values(by=metric_label, ascending=False)

                st.write("### Performance Comparison")
                st.dataframe(results_df)

                st.write("### Performance Chart")
                chart_df = results_df.set_index("Model")
                st.bar_chart(chart_df)

                st.write("### Analyze & Download Model")
                selected_model_name = st.selectbox("Select model to analyze", options=list(st.session_state.trained_models.keys()))

                if selected_model_name:
                    st.write(f"#### Prediction Results for {selected_model_name} on Test Data")
                    pred_data = st.session_state.prediction_data[selected_model_name]
                    result_preview_df = pred_data['X_test'].copy()
                    result_preview_df['Actual'] = pred_data['y_test']
                    result_preview_df['Predicted'] = pred_data['predictions']
                    st.dataframe(result_preview_df.head())

                    # --- Display Equation for Symbolic Models ---
                    model_to_save = st.session_state.trained_models[selected_model_name]
                    if "Symbolic" in selected_model_name:
                        st.write("#### Discovered Formula")
                        if 'gplearn' in selected_model_name and hasattr(model_to_save, '_program'):
                            st.latex(str(model_to_save._program))
                        elif 'PySR' in selected_model_name and hasattr(model_to_save, 'equations_'):
                             # Display the best equation from the PySR model
                            best_equation = model_to_save.get_best()
                            st.latex(best_equation['equation'])
                        else:
                            st.warning("Could not retrieve formula from this symbolic model.")

                    if problem_type == "Regression":
                        fig, ax = plt.subplots()
                        ax.scatter(pred_data['y_test'], pred_data['predictions'], alpha=0.5)
                        ax.plot([pred_data['y_test'].min(), pred_data['y_test'].max()], [pred_data['y_test'].min(), pred_data['y_test'].max()], 'r--', lw=2)
                        ax.set_xlabel("Actual Values")
                        ax.set_ylabel("Predicted Values")
                        ax.set_title(f"Actual vs. Predicted Values for {selected_model_name}")
                        st.pyplot(fig)

                    elif problem_type == "Classification":
                        st.write("#### Classification Report")
                        report = classification_report(pred_data['y_test'], pred_data['predictions'])
                        st.text_area("Report", value=report, height=250)

                        st.write("#### Confusion Matrix")
                        cm = confusion_matrix(pred_data['y_test'], pred_data['predictions'])
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)

                    model_to_save = st.session_state.trained_models[selected_model_name]

                    # --- Feature Importance ---
                    if hasattr(model_to_save, 'feature_importances_'):
                        st.write("#### Feature Importances")
                        feature_imp = pd.DataFrame(
                            sorted(zip(model_to_save.feature_importances_, pred_data['X_test'].columns)),
                            columns=['Value','Feature']
                        )
                        feature_imp_sorted = feature_imp.sort_values(by="Value", ascending=False).head(20)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="Value", y="Feature", data=feature_imp_sorted, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)

                    # --- SHAP Analysis Section ---
                    try:
                        st.write("---")
                        st.subheader("Analyze Prediction Rationale (SHAP)")

                        # Allow user to select a data point from the test set to explain
                        selected_idx = st.selectbox(
                            "Select a data point from the test set to explain:",
                            pred_data['X_test'].index
                        )

                        if selected_idx is not None:
                            with st.spinner("Calculating SHAP values..."):
                                # 1. Create a SHAP explainer
                                # Use TreeExplainer for tree-based models for performance, otherwise KernelExplainer
                                if any(model_type in selected_model_name for model_type in ['RandomForest', 'GradientBoosting', 'LGBM']):
                                    explainer = shap.TreeExplainer(model_to_save, pred_data['X_test'])
                                else:
                                    if hasattr(model_to_save, 'predict_proba'):
                                        explainer = shap.KernelExplainer(model_to_save.predict_proba, pred_data['X_test'])
                                    else:
                                        explainer = shap.KernelExplainer(model_to_save.predict, pred_data['X_test'])

                                # 2. Get the specific data point to explain
                                X_explain = pred_data['X_test'].loc[[selected_idx]]

                                # 3. Calculate SHAP values for the selected data point
                                shap_values = explainer.shap_values(X_explain)

                                # 4. Display the force plot
                                st.write(f"**SHAP Force Plot for data point {selected_idx}**")
                                if isinstance(shap_values, list): # For classification
                                    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X_explain))
                                else: # For regression
                                    st_shap(shap.force_plot(explainer.expected_value, shap_values, X_explain))
                    except Exception as e:
                        st.warning(
                            "Could not generate SHAP analysis for this model. "
                            "This may be expected for certain model types (like Symbolic Regressors). "
                            "Skipping SHAP analysis."
                        )
                    # --- End SHAP Section ---

                    model_buffer = BytesIO()
                    joblib.dump(model_to_save, model_buffer)
                    model_buffer.seek(0)
                    st.download_button(
                        label=f"Download {selected_model_name} Model",
                        data=model_buffer,
                        file_name=f"{selected_model_name.lower().replace(' ', '_')}.joblib",
                        mime="application/octet-stream"
                    )

                    st.write("---")
                    st.write(f"#### Predict with {selected_model_name}")
                    with st.form(key="prediction_form"):
                        st.write("Enter new data for prediction:")

                        feature_columns = pred_data['X_test'].columns
                        input_data = {}

                        cols = st.columns(4)
                        for i, feature in enumerate(feature_columns):
                            with cols[i % 4]:
                                input_data[feature] = st.number_input(
                                    label=feature,
                                    key=f"input_{feature}",
                                    value=0.0,
                                    format="%.4f"
                                )

                        submit_button = st.form_submit_button(label="Predict")

                        if submit_button:
                            try:
                                input_df = pd.DataFrame([input_data])
                                input_df = input_df[feature_columns]

                                prediction = model_to_save.predict(input_df)
                                prediction_proba = None
                                if problem_type == "Classification" and hasattr(model_to_save, "predict_proba"):
                                    prediction_proba = model_to_save.predict_proba(input_df)

                                st.success(f"**Prediction:** `{prediction[0]}`")
                                if prediction_proba is not None:
                                    st.write("**Prediction Probabilities:**")
                                    st.dataframe(pd.DataFrame(prediction_proba, columns=model_to_save.classes_))
                            except Exception as form_e:
                                st.error(f"An error occurred during prediction: {form_e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def run_time_series_forecasting():
    st.header("Time Series Forecasting")
    st.write("This mode uses models like LSTM and Prophet to forecast future values based on past data.")

    uploaded_file = st.file_uploader("1. Upload your time series CSV data", type="csv", key="ts_uploader")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("2. Configure Time Series Data")
            all_columns = df.columns.tolist()
            date_col = st.selectbox("Select the Date/Time column", options=all_columns)
            value_col = st.selectbox("Select the Value column to predict", options=all_columns)

            try:
                df[date_col] = pd.to_datetime(df[date_col])
                st.success(f"Column '{date_col}' successfully converted to datetime.")
            except Exception as e:
                st.error(f"Could not convert '{date_col}' to datetime. Error: {e}")
                return

            st.subheader("3. Configure Model")
            ts_model_name = st.selectbox("Choose a forecasting model", options=["LSTM", "Prophet"])

            if ts_model_name == "LSTM":
                with st.expander("LSTM Hyperparameters"):
                    sequence_length = st.slider("Sequence Length (Window Size)", 5, 100, 30, 1, key="lstm_seq")
                    lstm_units = st.slider("LSTM Units", 10, 200, 50, 10, key="lstm_units")
                    epochs = st.slider("Epochs", 1, 100, 20, 1, key="lstm_epochs")
                    batch_size = st.slider("Batch Size", 1, 64, 32, 1, key="lstm_batch")

            elif ts_model_name == "Prophet":
                with st.expander("Prophet Parameters"):
                    periods_to_forecast = st.number_input("Periods to Forecast into the Future", min_value=1, max_value=365*5, value=30, key="prophet_periods")

                    # --- Multivariate Forecasting UI ---
                    st.write("---")
                    st.write("**Optional: Add External Regressors for Multivariate Forecasting**")

                    # Exclude date and value columns from regressor options
                    available_regressors = [col for col in all_columns if col not in [date_col, value_col]]

                    selected_regressors = st.multiselect(
                        "Select external regressor columns (X)",
                        options=available_regressors,
                        help="Select other variables that might influence the value you are trying to predict."
                    )

                    future_regressor_values = {}
                    if selected_regressors:
                        st.write("Define future values for the selected regressors for the forecast period:")
                        for regressor in selected_regressors:
                            # Use a default value, e.g., the mean of the column, or 0
                            default_value = df[regressor].mean() if pd.api.types.is_numeric_dtype(df[regressor]) else 0.0
                            future_regressor_values[regressor] = st.number_input(
                                f"Future value for {regressor}",
                                key=f"future_{regressor}",
                                value=default_value
                            )
                    # --- End of Multivariate UI ---

            # --- Calculation Block ---
            if st.button("Run Forecast", key="run_ts_forecast"):
                # Reset state before new run
                st.session_state['lstm_results_df'] = None
                st.session_state.prophet_model = None
                st.session_state.prophet_regressors = []

                if ts_model_name == "LSTM":
                    with st.spinner(f"Training LSTM model for {epochs} epochs..."):
                        results_df, lstm_model, lstm_scaler, error = train_and_predict_lstm(
                            df, date_col, value_col, sequence_length, lstm_units, epochs, batch_size
                        )
                    if error:
                        st.error(error)
                    else:
                        st.session_state['lstm_model'] = lstm_model
                        st.session_state['lstm_scaler'] = lstm_scaler
                        st.session_state['lstm_sequence_length'] = sequence_length
                        st.session_state['lstm_results_df'] = results_df
                        st.success("LSTM Forecast complete!")

                elif ts_model_name == "Prophet":
                    with st.spinner("Training Prophet model and making forecast..."):
                        prophet_model, forecast_df, error = train_and_predict_prophet(
                            df, date_col, value_col, periods_to_forecast,
                            regressors=selected_regressors,
                            future_regressor_values=future_regressor_values
                        )
                    if error:
                        st.error(error)
                    else:
                        st.session_state.prophet_model = prophet_model
                        st.session_state.prophet_regressors = selected_regressors
                        # Also save the forecast df to state to avoid re-plotting issues
                        st.session_state.prophet_forecast_df = forecast_df
                        st.success("Prophet Forecast complete!")

            # --- Display Block (runs independently of the button press) ---
            if ts_model_name == "LSTM" and st.session_state.get('lstm_results_df') is not None:
                st.subheader("Forecast vs. Actual Data")
                results_df = st.session_state['lstm_results_df']
                plot_df = results_df.rename(columns={value_col: "Actual"}).set_index(df[date_col][len(df)-len(results_df):])
                st.line_chart(plot_df)
                st.dataframe(plot_df)

                # --- Predict Next Value UI ---
                if st.session_state.lstm_model:
                    st.write("---")
                    st.subheader("Predict Next Value(s)")

                    # Single Value Prediction
                    st.write(f"**Predict a single next value based on the last sequence of the dataset.**")
                    with st.form(key="lstm_predict_form"):
                        last_sequence_str = ", ".join(map(str, df[value_col].iloc[-st.session_state.lstm_sequence_length:].values))
                        st.text_area(
                            "Last sequence from data (for reference)",
                            value=last_sequence_str,
                            disabled=True, height=100
                        )
                        lstm_predict_submit = st.form_submit_button("Predict Single Next Value")

                        if lstm_predict_submit:
                            try:
                                input_values = np.array([float(v.strip()) for v in last_sequence_str.split(',')])
                                scaled_input = st.session_state.lstm_scaler.transform(input_values.reshape(-1, 1))
                                reshaped_input = scaled_input.reshape((1, st.session_state.lstm_sequence_length, 1))
                                prediction_scaled = st.session_state.lstm_model.predict(reshaped_input)
                                prediction = st.session_state.lstm_scaler.inverse_transform(prediction_scaled)
                                st.success(f"**Predicted Next Value:** `{prediction[0][0]:,.6f}`") # Increased precision
                            except Exception as e:
                                st.error(f"An error occurred during prediction: {e}")
                    
                    st.write("---")
                    # Multi-step ahead prediction
                    st.write(f"**Forecast multiple steps into the future using a sliding window.**")
                    with st.form(key="lstm_multi_predict_form"):
                        n_steps = st.number_input("How many steps to predict ahead?", min_value=1, max_value=100, value=10)
                        multi_predict_submit = st.form_submit_button("Run Multi-Step Forecast")

                        if multi_predict_submit:
                            with st.spinner("Running multi-step forecast..."):
                                try:
                                    # Get the last sequence from the original data
                                    last_sequence = df[value_col].iloc[-st.session_state.lstm_sequence_length:].values
                                    
                                    # Get future predictions
                                    future_preds = predict_future_lstm(
                                        st.session_state.lstm_model,
                                        st.session_state.lstm_scaler,
                                        last_sequence,
                                        st.session_state.lstm_sequence_length,
                                        n_steps
                                    )

                                    st.success("Multi-step forecast complete!")
                                    
                                    # Create a dataframe for the results
                                    future_df = pd.DataFrame(future_preds, columns=["Forecast"])
                                    future_df.index = future_df.index + 1 # Start index from 1
                                    
                                    st.write("#### Forecasted Values")
                                    st.dataframe(future_df.style.format("{:.6f}"))

                                    # Plot the results
                                    st.write("#### Forecast Plot")
                                    fig, ax = plt.subplots()
                                    ax.plot(future_df.index, future_df['Forecast'], marker='o', linestyle='-', label='Future Forecast')
                                    ax.set_xlabel("Step Ahead")
                                    ax.set_ylabel("Predicted Value")
                                    ax.set_title(f"{n_steps}-Step Ahead Forecast")
                                    ax.legend()
                                    ax.grid(True)
                                    st.pyplot(fig)

                                except Exception as e:
                                    st.error(f"An error occurred during multi-step prediction: {e}")

            if ts_model_name == "Prophet" and st.session_state.get('prophet_model') is not None:
                prophet_model = st.session_state.prophet_model
                forecast_df = st.session_state.prophet_forecast_df

                st.subheader("Forecast Plot")
                plot_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
                plot_df = plot_df.join(df.rename(columns={date_col: 'ds', value_col: 'Actual'}).set_index('ds'))
                plot_df = plot_df[['Actual', 'yhat', 'yhat_lower', 'yhat_upper']]
                plot_df.rename(columns={'yhat': 'Forecast', 'yhat_lower': 'Lower Confidence', 'yhat_upper': 'Upper Confidence'}, inplace=True)
                st.line_chart(plot_df)

                st.subheader("Forecast Components")
                fig2 = prophet_model.plot_components(forecast_df)
                st.pyplot(fig2)

                st.subheader("Forecast Data")
                st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

                if st.session_state.prophet_regressors:
                    st.write("---")
                    st.subheader("Predict with Custom Values")
                    st.write("Enter values for the regressors to get a single point forecast.")

                    with st.form(key="what_if_form"):
                        what_if_regressors = {}
                        for regressor in st.session_state.prophet_regressors:
                            what_if_regressors[regressor] = st.number_input(f"Value for {regressor}", key=f"what_if_{regressor}")

                        what_if_submit = st.form_submit_button("Run Single Prediction")

                        if what_if_submit:
                            with st.spinner("Predicting..."):
                                model = st.session_state.prophet_model
                                future_df = model.make_future_dataframe(periods=1)
                                for regressor, value in what_if_regressors.items():
                                    future_df[regressor] = value
                                forecast = model.predict(future_df)
                                predicted_value = forecast.iloc[-1]['yhat']
                                st.success(f"**Predicted Value (yhat):** `{predicted_value:,.2f}`")

        except Exception as e:
            st.error(f"An error occurred while processing the time series data: {e}")

def run_pinn_solver():
    """
    UI for the generalized Physics-Informed Neural Network (PINN) solver.
    """
    st.header("⚡ General Purpose PINN Solver")
    st.write("Define a differential equation, provide boundary/initial conditions, and train a PINN to find the solution.")

    # --- 1. Load Existing Model (Optional) ---
    with st.expander("1. Load Existing Model (Optional)"):
        uploaded_model_file = st.file_uploader("Upload a trained PINN model (.keras)", type="keras")
        if uploaded_model_file:
            with st.spinner("Loading model..."):
                # Save the uploaded file temporarily to load it from a path
                with open("temp_model.keras", "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
                
                solver = PINNSolver.load_model("temp_model.keras")
                if solver:
                    st.session_state.pinn_solver = solver
                    st.success("Model loaded successfully!")
                    # Clean up the temporary file
                    os.remove("temp_model.keras")
                else:
                    st.error("Failed to load the model.")

    # --- 2. Define Problem and Train ---
    st.subheader("2. Define Problem and Train New Model")
    
    col1, col2 = st.columns(2)
    with col1:
        # ODE Input
        st.write("**Define the Differential Equation**")

        # Session state to hold the ODE string
        if 'pinn_ode_string' not in st.session_state:
            st.session_state.pinn_ode_string = "dy/dx + y"

        # LLM Suggestion Box
        with st.expander("Or, get a suggestion from the AI"):
            user_description = st.text_area(
                "Describe the physical problem or phenomenon",
                placeholder="e.g., A simple harmonic oscillator where acceleration is proportional to position, with a constant of 4."
            )
            if st.button("Suggest Equation"):
                if user_description:
                    with st.spinner("Asking AI for an equation..."):
                        prompt = create_pinn_prompt(user_description)
                        suggestion = get_llm_suggestions(prompt)
                        if "Error:" not in suggestion:
                            st.session_state.pinn_ode_string = suggestion
                            st.success("Suggestion received!")
                        else:
                            st.error(suggestion)
                else:
                    st.warning("Please enter a description first.")
        
        ode_string = st.text_input(
            "Enter the ODE residual",
            key='pinn_ode_string', # Bind the text input to the session state
            help="Define the equation in its residual form (equal to zero). Use 'y' for the function and 'x' for the variable. Use 'dy/dx' for the first derivative and 'd2y/dx2' for the second."
        )
        st.write("---")
        # Condition Data Upload
        uploaded_conditions_file = st.file_uploader(
            "Upload Initial/Boundary Conditions (CSV)",
            type="csv",
            help="CSV must have 'x' and 'y' columns."
        )
        if uploaded_conditions_file:
            st.session_state.pinn_condition_data = pd.read_csv(uploaded_conditions_file)
            st.write("Data Preview:")
            st.dataframe(st.session_state.pinn_condition_data.head())

    with col2:
        # Hyperparameters
        st.write("Training Hyperparameters")
        epochs = st.slider("Epochs", 100, 20000, 5000, 100, key="pinn_epochs")
        domain_points = st.slider("Collocation Points", 50, 1000, 200, 10, key="pinn_domain_points")
        layers = st.slider("Hidden Layers", 1, 10, 3, 1, key="pinn_layers")
        neurons = st.slider("Neurons per Layer", 8, 128, 32, 4, key="pinn_neurons")

    if st.button("Train PINN Model", key="train_pinn"):
        if ode_string and st.session_state.pinn_condition_data is not None:
            with st.spinner("Training PINN... See console for progress."):
                try:
                    # Initialize a new solver
                    solver = PINNSolver(num_hidden_layers=layers, num_neurons_per_layer=neurons)
                    
                    # Create domain points based on the condition data's range
                    x_min = st.session_state.pinn_condition_data['x'].min()
                    x_max = st.session_state.pinn_condition_data['x'].max()
                    domain_x = np.linspace(x_min, x_max, domain_points).reshape(-1, 1)

                    # Train the model
                    solver.train(ode_string, domain_x, st.session_state.pinn_condition_data, epochs)
                    
                    # Store results in session state
                    st.session_state.pinn_solver = solver
                    st.session_state.pinn_history = solver.history
                    st.success("PINN training completed!")

                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
        else:
            st.warning("Please provide both an ODE string and condition data to train a model.")

    # --- 3. Analyze Results and Predict ---
    if st.session_state.pinn_solver:
        st.subheader("3. Analyze Results and Predict")
        solver = st.session_state.pinn_solver

        # Display loss history if available from the last training run
        if st.session_state.pinn_history:
            st.write("#### Training Loss History")
            history_df = pd.DataFrame(st.session_state.pinn_history, columns=["Total Loss", "Data Loss", "Physics Loss"])
            st.line_chart(history_df)

        # Plot solution
        st.write("#### Solution Plot")
        condition_data = st.session_state.pinn_condition_data
        if condition_data is not None:
            x_min, x_max = condition_data['x'].min(), condition_data['x'].max()
            x_plot = np.linspace(x_min, x_max, 400).reshape(-1, 1)
            y_pred = solver.predict(x_plot).numpy()

            fig, ax = plt.subplots()
            ax.plot(x_plot, y_pred, label="PINN Solution", color="red", zorder=2)
            ax.scatter(condition_data['x'], condition_data['y'], label="Condition Data", color="blue", zorder=3)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("PINN Solution vs. Condition Data")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Download model
        model_buffer = BytesIO()
        try:
            # Temporarily save to a file, read it into buffer, then delete.
            solver.save_model("temp_model_download.keras")
            with open("temp_model_download.keras", "rb") as f:
                model_buffer.write(f.read())
            os.remove("temp_model_download.keras")
            model_buffer.seek(0)
            
            st.download_button(
                label="Download Trained Model",
                data=model_buffer,
                file_name="pinn_model.keras",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Could not prepare model for download: {e}")

        # Prediction form
        st.write("---")
        st.write("#### Predict on New Data")
        with st.form("pinn_predict_form"):
            x_input = st.number_input("Enter a value for x to predict y:", value=0.0, format="%.4f")
            predict_button = st.form_submit_button("Predict")
            if predict_button:
                try:
                    prediction = solver.predict(np.array([[x_input]]))
                    st.success(f"**Predicted y({x_input}) =** `{prediction.numpy()[0][0]:.6f}`")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
