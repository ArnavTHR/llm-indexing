LLM Tabular Data Extraction Benchmark
A lightweight benchmarking project to evaluate how prompt complexity impacts the latency and accuracy of Large Language Models extracting specific rows from tabular data. It compares a cloud-based API (gemini-3.5-flash) against a locally hosted open-weight model (unsloth/Llama-3.2-1B-Instruct).


Key Features::

---Latency Tracking: Measures Time-To-First-Token (TTFT) and Generation Time using streaming responses.

---Prompt Evaluation: Tests Short, Medium, Long, and Descriptive prompt formats to see how verbosity affects processing speed.

---Dual-Environment Setup: Includes scripts for both cloud API routing (Google GenAI) and local GPU execution (Hugging Face Transformers/PyTorch).

---Automated Visualization: Generates stacked bar charts comparing indexing vs. generation times.


Prerequisites::

Google Colab (recommended for easy execution).

A Gemini API Key (stored in Colab Secrets as GEMINI_API_KEY).

T4 GPU enabled in Colab (for the local Llama benchmark).


---->>Quick Start Guide::

1. Run the Gemini API Benchmark
This script creates a 200-row synthetic dataset and queries the Gemini API to extract a specific row.

Open a new Google Colab notebook.

Add your GEMINI_API_KEY to Colab Secrets.

Install the SDK: !pip install google-genai pandas

Paste and run the Gemini Python script (File 1) to view the streaming outputs and TTFT metrics.

2. Run the Local Llama Benchmark
This script loads a 1B parameter Llama model into GPU memory, runs extraction tasks, and plots the latency.

Open a Colab notebook and set the Runtime to T4 GPU.

Install dependencies: !pip install -q transformers accelerate pandas matplotlib

Paste and run the PyTorch benchmark script (File 2).

In a new cell, run the visualization script using the generated results_df variable:

Python
import pandas as pd
import matplotlib.pyplot as plt

plot_df = results_df.rename(columns={
    "Scenario": "Prompt Type",
    "Processing Time (s)": "Indexing Time (s)",
    "Generation Time (s)": "Generating Time (s)"
}).set_index('Prompt Type')

plot_df[['Indexing Time (s)', 'Generating Time (s)']].plot.bar(stacked=True, figsize=(8, 6))
plt.title('LLM Processing Time: Indexing vs. Generating')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
