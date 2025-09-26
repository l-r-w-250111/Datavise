import requests
import json
import os

# --- Configuration Loading ---
DEFAULT_CONFIG = {
    "ollama_api_url": "http://localhost:11434/api/generate",
    "llm_model": "gemma3:12b"
}

def load_config():
    """Loads configuration from settings.json, with fallbacks to defaults."""
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", "r") as f:
                config = json.load(f)
                # Ensure all keys have a value, falling back to default if missing
                config = {**DEFAULT_CONFIG, **config}
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load or parse settings.json: {e}. Using default settings.")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

# Load config once when the module is imported
CONFIG = load_config()
OLLAMA_API_URL = CONFIG["ollama_api_url"]
LLM_MODEL = CONFIG["llm_model"]
# --- End Configuration Loading ---


def create_prompt(column_names: list, target_column: str, problem_type: str, available_models: list, user_comment: str = ""):
    """
    Generates a prompt for Ollama to suggest ML models from a provided list.
    """

    user_comment_section = ""
    if user_comment:
        user_comment_section = f"""
**User Comments about the data:**
{user_comment}
"""

    prompt = f"""
You are an expert data scientist. Your task is to recommend a few suitable machine learning models from a given list, considering the user's comments.

**Problem Description:**
- Data Columns: {', '.join(column_names)}
- Target Variable: {target_column}
- Problem Type: {problem_type}
{user_comment_section}
**Available Models:**
{', '.join(available_models)}

Based on the problem description and any user comments, select 3 to 5 of the most appropriate models from the "Available Models" list above.
Your answer MUST be a simple comma-separated list of the model names you recommend, and nothing else.

For example: RandomForestRegressor, GradientBoostingRegressor, SymbolicRegressor

Your recommended models:
"""
    return prompt

def get_llm_suggestions(prompt: str):
    """
    Calls the Ollama API to get model suggestions.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        A string containing the model suggestions, or an error message.
    """
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,  # Set to False to get the full response at once
            "options": {
                "temperature": 0.0,
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes

        # The full response is a single JSON object when stream=False
        response_data = response.json()

        # Extract the content of the response
        full_response = response_data.get("response", "").strip()

        return full_response

    except requests.exceptions.ConnectionError:
        return f"Error: Connection to Ollama server at '{OLLAMA_API_URL}' failed. Is Ollama running?"
    except requests.exceptions.RequestException as e:
        return f"Error: An error occurred while communicating with Ollama: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to decode the response from Ollama."
