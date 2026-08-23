import time
import psutil
import os
import json
import faiss
from llama_cpp import Llama

from embedder import load_embedder

process = psutil.Process(os.getpid())

def get_ram_mb():
    return process.memory_info().rss / (1024 * 1024)

print("=== ADTC Benchmark: Offline Shona AI Coding Tutor ===\n")

ram_start = get_ram_mb()
print(f"[RAM] Before loading anything: {ram_start:.1f} MB")

# Load embedder from model/all-MiniLM-L6-v2 (offline)
embedder = load_embedder()
ram_after_embedder = get_ram_mb()
print(f"[RAM] After loading embedder: {ram_after_embedder:.1f} MB")

# Load FAISS index
index = faiss.read_index("data/syllabus.index")
with open("data/syllabus_map.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)
ram_after_index = get_ram_mb()
print(f"[RAM] After loading FAISS index: {ram_after_index:.1f} MB")

# Load LLM
llm_load_start = time.time()
llm = Llama(
    model_path="./model/gemma-2-2b-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    verbose=False
)
llm_load_time = time.time() - llm_load_start
ram_after_llm = get_ram_mb()
print(f"[RAM] After loading LLM: {ram_after_llm:.1f} MB")
print(f"[TIME] LLM load time: {llm_load_time:.2f} seconds\n")

# Run a test generation and measure tokens/sec
test_prompt = "<start_of_turn>user\nExplain what a for loop is in Python, briefly.<end_of_turn>\n<start_of_turn>model\n"

print("Running test generation...")
gen_start = time.time()
output = llm(test_prompt, max_tokens=200, stop=["<|end|>"], echo=False)
gen_time = time.time() - gen_start

generated_text = output["choices"][0]["text"]
token_count = output["usage"]["completion_tokens"]
tokens_per_sec = token_count / gen_time if gen_time > 0 else 0

ram_peak = get_ram_mb()

print(f"\n[GENERATION] Tokens generated: {token_count}")
print(f"[GENERATION] Time taken: {gen_time:.2f} seconds")
print(f"[GENERATION] Speed: {tokens_per_sec:.2f} tokens/sec")
print(f"\n[RAM] Peak RSS during run: {ram_peak:.1f} MB ({ram_peak/1024:.2f} GB)")
print(f"[BUDGET] 7 GB limit = 7168 MB — {'WITHIN BUDGET' if ram_peak < 7168 else 'OVER BUDGET - CRITICAL'}")

print("\n--- Sample output ---")
print(generated_text[:300])
