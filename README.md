# Offline Shona AI Coding Tutor

An offline AI coding tutor for the **Africa Deep Tech Challenge 2026 — Laptop LLM Challenge**. Teaches Python and CS fundamentals in English and Shona, running fully on-device with no internet required.

## Features
- Fully offline inference (Phi-3.5-mini, quantized, via llama.cpp)
- Bilingual: English and Shona, for both explanations and practice questions
- RAG-grounded answers from a curated 21-topic CS syllabus
- Two modes: ask a question, or generate practice questions on a topic
- Runs within a 7GB RAM budget, CPU-only (measured peak: ~4.75GB)
- Honest fallback for out-of-scope questions instead of guessing

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/tmachingur-code/adtc-shona-coding-tutor.git
cd adtc-shona-coding-tutor

# 2. Create virtual environment
python3 -m venv adtc-tutor-env
source adtc-tutor-env/bin/activate

# 3. Install build tools
sudo apt update && sudo apt install build-essential cmake -y

# 4. Install dependencies
pip install llama-cpp-python sentence-transformers faiss-cpu numpy huggingface_hub psutil

# 5. Download the model
mkdir -p models
hf download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir ./models

# 6. Build the RAG index
python3 build_index.py

# 7. Run the tutor
python3 rag_tutor.py
```

## Usage

You'll be asked to choose a mode and a language, then either ask a question or name a topic.

```
Choose mode - (1) Ask a question, (2) Get practice questions: 1
Language (english/shona): shona
Ask a coding question: Chii chinonzi for loop

--- Tutor's Answer ---
Musoro: For Loops
Tsanangudzo: For loop inodzokorora chikamu chekodhi kamwe nekamwe...
```

## How It Works

Your question is embedded and matched against a curated syllabus using FAISS. English answers/practice questions are generated live by the model using the matched context; Shona answers/practice questions return curated, human-verified content directly (for accuracy — see REPORT.md for why). Questions outside the syllabus scope return a clear fallback message instead of a guess.

**Note on Shona input:** questions phrased fully in English, or in Shona with an embedded English technical term (e.g. "Chii chinonzi for loop"), match reliably. Fully Shona questions with no English term do not currently match reliably — see REPORT.md Section 5 for details.

## Syllabus Coverage

Python Basics · Control Flow · Functions · Data Structures · Algorithms & Reasoning (sorting, searching, Big-O) · Debugging Fundamentals — 21 topics total.

## Checking Performance

```bash
python3 benchmark.py
```
Reports peak RAM usage and generation speed (tokens/sec).

See `REPORT.md` for full design rationale, constraints, benchmarks, and known limitations.

