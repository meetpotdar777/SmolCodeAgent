Smolagents 🤖

This repository contains a Python script (full_smolagents_quickstart.py) demonstrating the core functionalities of smolagents, an open-source Python library designed for building and running AI agents with minimal code.

🚀 What is Smolagents?
smolagents simplifies the process of creating AI agents, offering first-class support for CodeAgent (agents that write and execute code), common tool-calling, and seamless integration with various Large Language Models (LLMs) and tools. It's built for simplicity, flexibility, and extensibility.

✨ Key Features Demonstrated
Basic Agent Creation: Running an agent for simple computational tasks. ➕

Tool Integration: Using DuckDuckGoSearchTool to enable web search capabilities. 🌐

Model Agnostic: Examples showing integration with:

Hugging Face Inference API (default and specific models). 🤗

OpenAI models via LiteLLM (requires API key). 🔑

Local models via Transformers (requires significant resources and potentially Hugging Face login/access). 💻

📋 Prerequisites
Before running the script, ensure you have Python 3.8+ installed. ✅

🛠️ Installation
Clone this repository (or copy the full_smolagents_quickstart.py file):

# If you have a Git repository setup
git clone <your-repo-url>
cd <your-repo-name>

(Or simply create full_smolagents_quickstart.py and paste the code into it.)

Install smolagents with all necessary extras:
Open your terminal or command prompt and run:

pip install smolagents[toolkit,litellm,transformers]

This command installs:

smolagents core library. 🧠

toolkit for default tools like DuckDuckGoSearchTool. 🔧

litellm for integrations with various LLM APIs (e.g., OpenAI). 🔗

transformers for loading and running models locally. ⚡

🔑 API Key and Hugging Face Login Setup
Many of the examples rely on external AI models and APIs. You'll need to configure your environment for them. 🔒

1. Hugging Face Login (for InferenceClientModel and TransformersModel)
Some models on Hugging Face (including the default ones used by InferenceClientModel and most Llama models) are gated or require authentication.

Get a Hugging Face Token:

Go to huggingface.co/settings/tokens. 🔗

Click "New token", give it a name (e.g., smolagents-token), and select the "read" role. 📝

Copy the token immediately. 📋

Log in via CLI:
Open your terminal and run:

huggingface-cli login

Paste your copied Hugging Face token when prompted. 🔐

2. OpenAI API Key (for LiteLLMModel with OpenAI)
If you plan to run Example 3.2 (using OpenAI models like GPT-4 or GPT-3.5-turbo), you need an OpenAI API key. 💡

Get an OpenAI API Key:

Go to platform.openai.com/account/api-keys. 🔑

Click "Create new secret key". ✨

Copy the key. 📋

Set as Environment Variable:
Open your terminal (the same one you'll use to run the Python script) and set the environment variable. 💻

Windows (Command Prompt):

set OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY_HERE"

Windows (PowerShell):

$env:OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY_HERE"

Linux/macOS:

export OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY_HERE"

Important: Replace sk-proj-YOUR_OPENAI_API_KEY_HERE with your actual OpenAI API key. Ensure there are no extra spaces or characters. ❗

💻 The Quickstart Code
Below is the full_smolagents_quickstart.py script. 📄

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
    # We'll explicitly set a more reliable public model to avoid "Response ended prematurely" issues.
    print("Initializing InferenceClientModel with 'HuggingFaceH4/zephyr-7b-beta'...")
    model_no_tools = InferenceClientModel(model_id="HuggingFaceH4/zephyr-7b-beta")

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
    print("This might be due to issues with the Hugging Face Inference API or network.\n")


# --- Example 2: Adding Tools (DuckDuckGo Search) ---
print("\n--- Example 2: Agent with DuckDuckGoSearchTool ---")
try:
    # Initialize a model, explicitly setting a more reliable public model.
    print("Initializing InferenceClientModel for tool-using agent with 'HuggingFaceH4/zephyr-7b-beta'...")
    model_with_tools = InferenceClientModel(model_id="HuggingFaceH4/zephyr-7b-beta")

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
    print("Ensure you have network access for DuckDuckGoSearchTool and that the model is accessible.\n")


# --- Example 3: Demonstrating different model integrations ---
print("\n--- Example 3: Demonstrating different model integrations ---")

# --- 3.1 Using a specific model from Hugging Face Inference API ---
print("\n--- 3.1: Using a specific model from Hugging Face Inference API ---")
try:
    # We'll use a publicly available model here.
    # For gated models like Llama-2-70b-chat-hf, you must have granted access and logged in via `huggingface-cli login`.
    hf_model_id = "mistralai/Mistral-7B-Instruct-v0.2" # A popular, often unrestricted alternative
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
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"): # Basic check for key format
        print("OPENAI_API_KEY found. Initializing LiteLLMModel for OpenAI...")
        # Using gpt-3.5-turbo as it's more widely accessible than gpt-4
        model_litellm_openai = LiteLLMModel(model_id="gpt-3.5-turbo")

        agent_litellm_openai = CodeAgent(tools=[], model=model_litellm_openai)
        task_litellm_openai = "Write a short poem about a rainy day in Mumbai."
        print(f"Running agent with task: '{task_litellm_openai}' using GPT-3.5-turbo via LiteLLM")
        result_litellm_openai = agent_litellm_openai.run(task_litellm_openai)
        print(f"Result (LiteLLM OpenAI): {result_litellm_openai}")
    else:
        print("Skipping LiteLLM OpenAI example: OPENAI_API_KEY environment variable not found or invalid format.")
        print("To run this, please set your OPENAI_API_KEY (starting with 'sk-proj-' or 'sk-'): export OPENAI_API_KEY='YOUR_KEY_HERE'\n")
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
    # We'll use a widely accessible model here. For Llama models, you must have gated access granted.
    local_model_id = "HuggingFaceH4/zephyr-7b-beta" # A popular, non-gated alternative
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
    print("Common issues: Insufficient RAM/VRAM, model not found locally, or compatibility problems.")
    print("If you encounter 'CUDA out of memory', try running on CPU or with a smaller model, or skip this.\n")


print("\n--- Smolagents Quickstart Examples Finished ---")


🏃 Running the Script
Save the code: Save the code block above as full_smolagents_quickstart.py in your chosen directory. 💾

Open your terminal or command prompt in that directory. 🖥️

Ensure API keys are set and you are logged into Hugging Face as per the "API Key and Hugging Face Login Setup" section. ⚙️

Execute the script:

python full_smolagents_quickstart.py

🐛 Troubleshooting Common Issues
If you encounter errors, here are the most frequent ones and their solutions: 🛠️

You must provide an api_key to work with featherless-ai API or log in with huggingface-cli login.

Cause: The default or chosen Hugging Face model requires authentication. 🤔

Solution: Ensure you've run huggingface-cli login successfully with a valid Hugging Face token. Alternatively, change model_id to a truly public model like HuggingFaceH4/zephyr-7b-beta. ✅

AuthenticationError: OpenAIException - Incorrect API key provided:

Cause: Your OPENAI_API_KEY is invalid, expired, or doesn't have access to the specified model (e.g., gpt-4). 🛑

Solution:

Go to platform.openai.com/account/api-keys and generate a brand new secret key. 🆕

Ensure you have active billing/credits on your OpenAI account. 💲

Set the OPENAI_API_KEY environment variable correctly in your terminal before running the script. ➡️

Consider using gpt-3.5-turbo for testing, as it's generally more accessible. 👍

Access to model ... is restricted and you are not in the authorized list. (for Llama models via InferenceClientModel or TransformersModel)

Cause: Llama models (and some others) are "gated" on Hugging Face. You need explicit access. 🚫

Solution:

Go to the model's page on Hugging Face (e.g., huggingface.co/meta-llama/Llama-2-7b-chat-hf). 🌐

Read and accept the license terms to request access. Access is usually granted quickly. 📄

Ensure you are logged in via huggingface-cli login. 🔐

Alternatively (recommended for quicker testing): Change the model_id in your script to an unrestricted model like HuggingFaceH4/zephyr-7b-beta for both InferenceClientModel and TransformersModel examples. 🔄

Response ended prematurely (for InferenceClientModel using default/large models)

Cause: Often a network timeout, server overload on Hugging Face's free inference API, or the model taking too long to generate a response. ⏳

Solution: As implemented in the updated code, switch to smaller, more stable, and widely available public models like HuggingFaceH4/zephyr-7b-beta or mistralai/Mistral-7B-Instruct-v0.2 for InferenceClientModel. ✅

CUDA out of memory / script hangs (for TransformersModel)

Cause: Running large language models locally is very resource-intensive, especially on systems without a dedicated GPU or insufficient RAM. 🧠💥

Solution:

Use a smaller model: Choose models with fewer parameters (e.g., 1B or 3B, like TinyLlama/TinyLlama-1.1B-Chat-v1.0), though their quality will be lower. 🤏

Run on CPU (slower): If you have enough RAM but no GPU, PyTorch will often fall back to CPU, which is much slower. 🐌

Skip this example: If your hardware limitations are too severe, you might need to comment out or skip Example 3.3. ⏭️

By following these steps, you should be able to get the smolagents quickstart examples running effectively on your system. 🎉
