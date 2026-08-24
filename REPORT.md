# Technical Report — Offline Shona AI Coding Tutor

**Team ID:** 
**Domain:** coding_assistants
**Model:** gemma-2-2b-it-Q4_K_M

**Runtime:** local `llama.cpp` inference with a local `all-MiniLM-L6-v2` embedder and FAISS index

---

## Problem

Access to coding education across much of Africa is constrained by unreliable or expensive internet connectivity, the cost of cloud-based AI tutoring subscriptions, and the fact that most instructional material — including AI tutoring tools — is delivered exclusively in English, disadvantaging students who are more comfortable learning technical concepts in a local language.

This is a first-hand problem for our team: we draw on lived experience from Gokwe North, Zimbabwe, a rural community where consistent connectivity and subscription-based tools are not a given, and where Shona is the primary language of everyday communication.

**Target user:** students and self-learners in under-connected areas who want to learn foundational Python and computer science, in English or Shona, without needing a stable internet connection or a paid subscription.

**Why local matters here:** cloud-based tutors are unusable without reliable data access, and even where connectivity exists, the recurring cost is a real barrier for students. Running entirely on-device removes both blockers — the tutor works the same whether a student has connectivity that day or not.

## Design Decisions

**Base model:** We started with Phi-3.5-mini-instruct (Microsoft, 3.8B), then switched to **Gemma-2-2b-it (Google, 2.6B)** after empirical comparison (see Benchmarks).

**Quantization:** Q4_K_M — chosen for a balance of output quality and memory footprint, standard practice for CPU-only local inference via `llama.cpp`.

**Alternatives considered:**
- **Phi-3.5-mini-instruct (Q4_K_M)** — our original choice. Benchmarked at ~6.8 tokens/sec and 4.75GB peak RAM. Working, but slower and heavier than necessary given our RAM headroom, so we tested a smaller model.
- **Gemma-2-2b-it (Q4_K_M)** — selected as final model. ~42% faster (9.67 tokens/sec self-measured) and ~31% less RAM (3.29GB self-measured) than Phi-3.5-mini, with no observed quality regression across our test questions.
- **Model-generated Shona output** — tested and rejected. Both candidate models produced incoherent Shona when asked to generate it freely (Shona is a significantly under-resourced language in current LLM training data). We use curated, human-verified Shona content instead of model generation for all Shona-language output (explanations and practice questions), retrieved via the same RAG pipeline used for English. This is a deliberate accuracy tradeoff, not an oversight.

## Constraints

- **Target hardware:** 8 GB RAM, integrated GPU (no discrete GPU), Ubuntu 26.04 LTS in the recorded participant run — matches the ADTC Standard Laptop profile.
- **No GPU acceleration** — pure CPU inference via `llama.cpp` throughout. We explicitly removed a default GPU-enabled PyTorch install in favor of the CPU-only build to avoid unnecessary bloat, since our target hardware has no CUDA-capable GPU.
- **Connectivity constraint:** the application must function with zero internet dependency once the model is downloaded — this shaped our choice of RAG (retrieval over a local knowledge base) instead of any live API calls or cloud-based translation for Shona content.
- **Data availability constraint:** no existing Shona-language CS education dataset was available, so our syllabus (21 topics, English + Shona explanations, examples, and practice questions) was authored directly by the team rather than sourced.
- **Language input constraint (discovered during testing):** the sentence-embedding model used for retrieval (`all-MiniLM-L6-v2`) does not reliably match fully Shona-language questions to our syllabus (measured embedding similarity as low as 0.15 for a genuine topic match). It does reliably match *code-switched* questions — Shona sentence structure with an embedded English technical term (e.g. "Chii chinonzi for loop"). We scoped and documented this rather than overclaiming full bilingual input support.
- **Runtime context and CPU settings:** the application uses a 2,048-token llama.cpp context and eight CPU threads. The bundled Gemma model advertises an 8,192-token model context, but the smaller application context keeps resource use appropriate for the target laptop.

## Benchmarks

**Self-reported development benchmarks (own `benchmark.py`, WSL2 Ubuntu, 12 CPU cores, 7.6GB RAM available):**

| Metric | Phi-3.5-mini (3.8B) | Gemma-2-2b-it (2.6B) — final |
|---|---|---|
| Machine | WSL2 Ubuntu (12-core, 7.6GB RAM) | WSL2 Ubuntu (12-core, 7.6GB RAM) |
| RAM at peak | 4.75 GB | 3.29 GB |
| Time to first token (model load) | ~2–8 sec | ~2 sec |
| Generation speed | ~6.7–6.9 t/s | 9.67 t/s |
| Thermal throttling | None observed | None observed |

**Official adtc-profiler results (participant mode, `measured_on: participant_laptop`):**

| Metric | Value |
|---|---|
| Machine | 13th Gen Intel Core i5-1335U, 7.6GB RAM, no GPU, Ubuntu 26.04 LTS |
| Peak RSS | 2.75 GB |
| Generation speed | 9.01 t/s (official participant run; local benchmark observed 9.67 t/s) |
| First-token latency | 17,946 ms |
| Steady-state RSS | 2.63 GB |
| CPU utilization (p99) | 50.7–50.8% |
| Thermal throttling | None (`throttled: false`) |
| Parameter count | 2,614,341,888 (confirmed match against declared 2.6B estimate) |

These official figures are measured by the ADTC profiler on our development machine in participant mode; final scores are measured by the ADTC profiler on the standard evaluation machine during Gate 2.

## Additional Notes

**Practice question generation:** beyond direct Q&A, the tutor supports a separate practice mode. English questions are generated live by the model, using retrieved syllabus context when the topic is in scope and general model knowledge otherwise. In-scope Shona questions are drawn from the curated set. For an out-of-scope Shona topic, the tutor explains the limitation in Shona and supplies generated English questions.

**Retrieval accuracy safeguard:** we use a FAISS L2 distance threshold of `1.6` so that clearly out-of-scope questions do not get presented as grounded syllabus answers. The threshold is applied consistently to both answer and practice-question retrieval. Exact-match, paraphrased, and off-topic distance examples should be treated as development observations rather than universal guarantees because the values depend on the embedding model and index contents.

**Known limitations:** question input is reliable in English and in code-switched Shona (with an embedded English technical term), but not in fully Shona phrasing without an English term present; in-scope Shona output is curated rather than freely generated; out-of-scope Shona responses are bilingual fallback responses rather than Shona model generations; the syllabus is scoped to 21 core CS topics; and the application requires the model files and generated FAISS index to be present locally.
