# full_smolagents_quickstart.py

# --- Installation Note ---
# Before running this code, ensure you have installed smolagents with the necessary extras:
# pip install smolagents[toolkit,litellm,transformers]
# This command includes default tools (like web search), LiteLLM for OpenAI/Anthropic,
# and Transformers for local models.

# --- Import necessary modules ---
from smolagents import CodeAgent, InferenceClientModel
from smolagents import DuckDuckGoSearchTool
from smolagents import LiteLLMModel
from smolagents import TransformersModel

import os

# --- Configuration (Optional but Recommended) ---
# Set up API keys as environment variables if you plan to use OpenAI/Anthropic models
# For OpenAI:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# For Anthropic:
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
# Replace "YOUR_OPENAI_API_KEY" and "YOUR_ANTHROPIC_API_KEY" with your actual keys.
# It's best practice to load these from environment variables or a secure configuration.

print("--- Starting Smolagents Quickstart Examples ---")
print("Make sure you have installed smolagents with: pip install smolagents[toolkit,litellm,transformers]\n")

# --- Example 1: Create Your First Agent (No Tools) ---
print("--- Example 1: Agent without tools (CodeAgent) ---")
try:
    # Initialize a model using Hugging Face Inference API (default model if not specified)
    # This will use a default model, often a smaller, general-purpose one.
    print("Initializing InferenceClientModel (default Hugging Face model)...")
    model_no_tools = InferenceClientModel()

    # Create an agent with no tools
    print("Creating CodeAgent with no tools...")
    agent_no_tools = CodeAgent(tools=[], model=model_no_tools)

    # Run the agent with a task that can be solved by internal reasoning (Python code)
    task_no_tools = "Calculate the sum of numbers from 1 to 10"
    print(f"Running agent with task: '{task_no_tools}'")
    result_no_tools = agent_no_tools.run(task_no_tools)
    print(f"Result from agent (no tools): {result_no_tools}")
    print("-" * 50)

except Exception as e:
    print(f"Error in Example 1: {e}")
    print("This might be due to issues with the default Hugging Face Inference API or network.\n")


# --- Example 2: Adding Tools (DuckDuckGo Search) ---
print("\n--- Example 2: Agent with DuckDuckGoSearchTool ---")
try:
    # Initialize a model
    print("Initializing InferenceClientModel for tool-using agent...")
    model_with_tools = InferenceClientModel()

    # Create an agent with the DuckDuckGoSearchTool
    print("Creating CodeAgent with DuckDuckGoSearchTool...")
    agent_with_tools = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model_with_tools,
    )

    # Now the agent can search the web!
    task_with_tools = "What is the current weather in Paris?"
    print(f"Running agent with task: '{task_with_tools}'")
    result_with_tools = agent_with_tools.run(task_with_tools)
    print(f"Result from agent (with DuckDuckGoSearchTool): {result_with_tools}")
    print("Note: Weather information can change quickly. This result is based on the search at the time of execution.")
    print("-" * 50)

except Exception as e:
    print(f"Error in Example 2: {e}")
    print("Ensure you have network access for DuckDuckGoSearchTool.\n")


# --- Example 3: Using Different Models ---
print("\n--- Example 3: Demonstrating different model integrations ---")

# --- 3.1 Using a specific model from Hugging Face Inference API ---
print("\n--- 3.1: Using a specific model from Hugging Face Inference API ---")
try:
    # You might need to have access to this model or it might be behind a paywall/rate limit.
    # Replace with a publicly available model if 'meta-llama/Llama-2-70b-chat-hf' causes issues.
    hf_model_id = "meta-llama/Llama-2-70b-chat-hf" # Or try "mistralai/Mixtral-8x7B-Instruct-v0.1" if Llama-2-70b is too restrictive
    print(f"Initializing InferenceClientModel with model_id='{hf_model_id}'...")
    model_hf_specific = InferenceClientModel(model_id=hf_model_id)

    agent_hf_specific = CodeAgent(tools=[], model=model_hf_specific)
    task_hf_specific = "Explain the concept of quantum entanglement in simple terms."
    print(f"Running agent with task: '{task_hf_specific}' using {hf_model_id}")
    result_hf_specific = agent_hf_specific.run(task_hf_specific)
    print(f"Result (Hugging Face specific model): {result_hf_specific[:200]}...") # Print first 200 chars
    print("-" * 50)

except Exception as e:
    print(f"Error in Example 3.1 (Hugging Face specific model): {e}")
    print("This often means the specified model is not accessible via the Inference API or requires an API token (HF_TOKEN).\n")


# --- 3.2 Using OpenAI/Anthropic via LiteLLM ---
print("\n--- 3.2: Using OpenAI/Anthropic via LiteLLM (Requires API Key) ---")
# Make sure you have `pip install smolagents[litellm]` and `OPENAI_API_KEY` set in environment.
try:
    if "OPENAI_API_KEY" in os.environ:
        print("OPENAI_API_KEY found. Initializing LiteLLMModel for OpenAI...")
        model_litellm_openai = LiteLLMModel(model_id="gpt-3.5-turbo") # You can also try "gpt-3.5-turbo"

        agent_litellm_openai = CodeAgent(tools=[], model=model_litellm_openai)
        task_litellm_openai = "Write a short poem about a rainy day in Mumbai."
        print(f"Running agent with task: '{task_litellm_openai}' using GPT-4 via LiteLLM")
        result_litellm_openai = agent_litellm_openai.run(task_litellm_openai)
        print(f"Result (LiteLLM OpenAI): {result_litellm_openai}")
    else:
        print("Skipping LiteLLM OpenAI example: OPENAI_API_KEY environment variable not found.")
        print("To run this, please set your OPENAI_API_KEY: export OPENAI_API_KEY='YOUR_KEY_HERE'\n")
    print("-" * 50)

except Exception as e:
    print(f"Error in Example 3.2 (LiteLLM OpenAI): {e}")
    print("Common issues: Invalid API key, rate limits, or network problems. Ensure `OPENAI_API_KEY` is correct.\n")


# --- 3.3 Using local models via Transformers ---
print("\n--- 3.3: Using local models via Transformers ---")
# Requires `pip install smolagents[transformers]` and significant local resources (RAM, GPU recommended).
try:
    # This will attempt to load the model locally. It can be very resource-intensive.
    # For a quicker test, consider smaller models or comment this out if you lack resources.
    local_model_id = "meta-llama/Llama-2-7b-chat-hf" # Requires significant RAM (e.g., 8-16GB+)
    # For a smaller, faster test, you might try a very small model, but they often lack quality.
    # e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0" or "HuggingFaceH4/zephyr-7b-beta" (still large)
    print(f"Initializing TransformersModel with model_id='{local_model_id}' for local execution...")
    print(f"Warning: This requires significant local resources ({local_model_id} needs a lot of RAM).")
    print("If it hangs or fails due to memory, you might need to adjust or skip this section.")
    model_transformers = TransformersModel(model_id=local_model_id)

    agent_transformers = CodeAgent(tools=[], model=model_transformers)
    task_transformers = "What is the capital of France?"
    print(f"Running agent with task: '{task_transformers}' using local model '{local_model_id}'")
    result_transformers = agent_transformers.run(task_transformers)
    print(f"Result (Transformers local model): {result_transformers}")
    print("-" * 50)

except Exception as e:
    print(f"Error in Example 3.3 (Transformers local model): {e}")
    print("Common issues: Insufficient RAM/VRAM, model not found locally, or compatibility problems.\n")
    print("If you encounter 'CUDA out of memory', try running on CPU or with a smaller model, or skip this.\n")


print("\n--- Smolagents Quickstart Examples Finished ---")