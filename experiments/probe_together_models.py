#!/usr/bin/env python3
"""
Probe Together AI to find which models are actually serverless (no dedicated endpoint needed).
Usage: .venv/bin/python experiments/probe_together_models.py
"""

import openai

API_KEY = "tgp_v1_ltHhNSf3rT-bfZKM6wJpoDIZxEK9KfByeRuuHNnu2GA"

CANDIDATES = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen3-235B-A22B-fp8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-8B",
    "moonshotai/Kimi-K2-Thinking",
    "google/gemma-2-9b-it",
    "google/gemma-3-27b-it",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
]

client = openai.OpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")

available, unavailable = [], []

for model in CANDIDATES:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"  OK  {model}")
        available.append(model)
    except openai.BadRequestError as e:
        if "non-serverless" in str(e):
            print(f"  --  {model}  (dedicated only)")
            unavailable.append(model)
        else:
            print(f"  ??  {model}  ({e})")
    except Exception as e:
        print(f"  ERR {model}  ({type(e).__name__}: {e})")

print(f"\n=== SERVERLESS ({len(available)}) ===")
for m in available:
    print(f"  {m}")

print(f"\n=== DEDICATED ONLY ({len(unavailable)}) ===")
for m in unavailable:
    print(f"  {m}")
