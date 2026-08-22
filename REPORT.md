# REPORT.md — Offline Shona AI Coding Tutor

## Africa Deep Tech Challenge 2026 — The Laptop LLM Challenge

**Track:** Math & Scientific Reasoning (Coding Assistants)

**Repository:**https:// github.com/tmachingur-code/adtc-shona-coding-tutor

---

## 1. Problem Definition

Access to coding education across much of Africa is constrained by three linked barriers: unreliable or expensive internet connectivity, the cost of cloud-based AI tutoring subscriptions, and the fact that most instructional material — including AI tutoring tools — is delivered exclusively in English, disadvantaging students who are more comfortable learning technical concepts in a local language.

This is a first-hand, not hypothetical, problem: our team draws on lived experience from Gokwe North, Zimbabwe, a rural community where consistent connectivity and subscription-based tools are not a given, and where Shona is the primary language of everyday communication.

**Our solution** is an offline, on-device coding tutor that:
- Runs entirely locally on modest, widely-available laptop hardware, with no internet dependency and no API costs
- Teaches foundational Python programming and computer science reasoning across six modules
- Provides explanations and practice questions in both English and Shona
- Generates fresh practice questions on demand to support active learning, not just Q&A

## 2. Constraints

The application is designed and tested against the ADTC Standard Laptop profile:

| Constraint | Target |
|---|---|
| RAM ceiling | 7 GB (hard limit — disqualification if exceeded) |
| CPU | Intel i5 10th–12th gen / AMD Ryzen 5 3000–5000, no discrete GPU |
| OS | Ubuntu 22.04 LTS |
| Storage | 256 GB SSD |

All development and testing was carried out in a WSL2 Ubuntu environment to closely mirror the target OS, using CPU-only inference throughout (no CUDA/GPU dependencies were installed). We note in Section 8 a caveat regarding WSL2 vs. bare-metal performance.

## 3. System Architecture

The system combines three components into a single reasoning pipeline:

1. **Local LLM (generation):** Gemma-2-2b-it (Google), quantized to Q4_K_M GGUF format by bartowski, run via `llama-cpp-python` (CPU-only, no GPU offload). No fine-tuning or retraining was performed. Selected after empirical comparison against Phi-3.5-mini-instruct (Microsoft) — see Section 7 for benchmark comparison.
2. **Retrieval layer (RAG):** a curated syllabus of 21 CS concept entries is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`, ~80MB) and indexed with FAISS (`IndexFlatL2`) for fast local similarity search
3. **Bilingual response layer:** English responses (both explanations and practice questions) are generated live by the LLM using retrieved context; Shona responses (both explanations and practice questions) are returned directly from curated, human-verified content (see Design Decision 4.2)

**Flow:** student question → embedded and matched against the syllabus index → if a confident match is found (distance below a tuned threshold), relevant context is retrieved → English mode passes this context to the LLM for a generated answer; Shona mode returns the pre-written Shona content directly → if no confident match is found, a graceful fallback message (in the requested language) is returned instead of a guess.

**Two interaction modes:**
- **Q&A mode:** student asks a specific question, receives a grounded explanation
- **Practice mode:** student names a topic, receives short practice questions to test their understanding (English: freshly generated; Shona: curated)

## 4. Key Design Decisions

### 4.1 RAG over fine-tuning
We deliberately chose retrieval-augmented generation instead of fine-tuning. Fine-tuning would require substantial training data, GPU compute, and time incompatible with both our hardware constraints and development timeline. RAG lets an unmodified, general-purpose small model behave like a scoped CS tutor by supplying relevant reference material at answer time — directly matching the challenge brief's systems mandate of "RAG over local corpora."

### 4.2 Curated Shona content instead of model-generated Shona
During testing, we found that asking Phi-3.5-mini to freely generate Shona text produced incoherent, unreliable output — consistent with published research showing Shona remains a significantly under-resourced language even among models built for African-language support. Rather than presenting unreliable generated text, we made a deliberate engineering decision: **all** Shona content (explanations and practice questions) is retrieved directly from a human-written, verified knowledge base rather than generated live. This guarantees linguistic accuracy for every Shona response, at the cost of Shona output being fixed rather than freely generated — a tradeoff we consider correct for an education tool, where incorrect explanations carry real cost to a learner.

### 4.3 Retention of English technical terminology within Shona content
Programming terms (e.g., "for loop," "function," "list") are retained in English within our Shona explanations and practice questions. This reflects common technical Shona usage, where such terms are widely used as-is even in fluent Shona technical discussion, rather than being an incomplete localisation.

### 4.4 Distance-threshold fallback for out-of-scope questions
Initial testing revealed that returning the closest syllabus match regardless of relevance risked confidently answering questions our syllabus does not cover. We introduced a similarity-distance threshold (tuned empirically to 1.4 using FAISS L2 distance) so that questions falling outside our scoped syllabus return a clear, honest fallback message — in the student's requested language — rather than a mismatched answer.

We validated this threshold against several cases:
- Exact-phrasing on-topic question: distance ≈ 0.42 (correctly matched)
- Paraphrased on-topic question: distance ≈ 1.10 (correctly matched)
- Clearly off-topic question: distance ≈ 1.94 (correctly triggered fallback)

### 4.5 Scoped syllabus rather than open-domain tutoring
The knowledge base is intentionally narrow: 21 concept entries across six modules (Python Basics, Control Flow, Functions, Data Structures, Algorithms & Reasoning, Debugging Fundamentals). This scope was chosen to maximize retrieval accuracy and keep the RAG index lightweight, rather than attempting broad CS coverage that would dilute both accuracy and hardware efficiency.

### 4.6 Practice question generation
Beyond answering direct questions, the tutor can generate topic-specific practice questions on demand — English questions are generated live by the LLM using retrieved syllabus context (keeping them grounded rather than hallucinated); Shona practice questions are drawn from a curated set of two per topic, for the same accuracy reasons described in 4.2.

## 5. Confirmed Capabilities and Limitations Around Shona Input

We tested student **question input** in Shona directly (as opposed to Shona *output*, covered above) and found a precise, characterizable pattern:

- **Pure Shona questions with no English technical term** (e.g., *"Ndinodzokorora sei kodhi kakawanda"* — "how do I repeat code many times") do not reliably match the syllabus and correctly trigger the fallback response. We measured embedding similarity between a Shona phrase and its English topic equivalent at only 0.154 — confirming this is a genuine limitation of the multilingual embedding model, not a bug.
- **Code-switched questions** — Shona sentence structure with an embedded English technical term (e.g., *"Chii chinonzi for loop"* — "what is called a for loop") — **do** match correctly, since the embedding model recognizes the English keyword.

We consider this an honest and realistic characterization rather than a shortfall: code-switching between Shona and English technical vocabulary is how many Zimbabwean students authentically discuss programming, so this behavior has genuine practical value even though it falls short of full Shona-language question understanding.

**Scope statement:** question input is reliable in English, and in Shona questions that include the relevant English technical term; response output is available in full English or full Shona depending on student preference.

## 6. Tools & Technology Stack

| Component | Tool |
|---|---|
| LLM inference | llama-cpp-python + Gemma-2-2b-it (Q4_K_M GGUF) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (IndexFlatL2), CPU-only |
| RAM monitoring | psutil |
| Language | Python 3.14 |
| Development environment | WSL2 Ubuntu (mirroring Ubuntu 22.04 LTS target) |

## 7. Benchmarks

Measured using our own `benchmark.py` script (included in the repository), run in a WSL2 Ubuntu environment with 12 CPU cores and 7.6GB RAM available to the VM.

### Model comparison

We empirically benchmarked two candidate models before selecting a final one, rather than choosing arbitrarily:

| Metric | Phi-3.5-mini-instruct (3.8B) | Gemma-2-2b-it (2B) — **selected** |
|---|---|---|
| Peak RAM usage | 4.75 GB | **3.29 GB** |
| LLM load time | ~2–8 sec | ~2 sec |
| Generation speed | ~6.7–6.9 tokens/sec | **9.67 tokens/sec** |
| Answer quality (manual review) | Good | Good, no regression observed |

Gemma-2-2b-it was selected as the final model: it uses ~31% less RAM and generates ~42% faster than Phi-3.5-mini, with no observed quality regression across our test questions (English Q&A, Shona Q&A, and practice-question generation in both languages).

### Final model results

| Metric | Result | Budget / Target | Status |
|---|---|---|---|
| Peak RAM usage | 3.29 GB | 7 GB ceiling | ✅ Within budget (~47% used) |
| Generation speed | 9.67 tokens/sec | Relative to fastest submission | Documented |
| Thermal throttling | Not observed in testing | No penalty | ✅ |

**Note on speed:** we tested thread counts of 4 and 8 on Phi-3.5-mini (machine has 12 physical cores available); the difference was negligible, indicating the bottleneck was compute-bound rather than thread-limited. We confirmed this was not a WSL2 resource-constraint artifact (`free -h` and `nproc` confirmed full CPU/RAM access to the VM). We acknowledge that WSL2 virtualization may introduce overhead not present on bare-metal Ubuntu 22.04, the actual Gate 2 audit environment, and results there may differ.

## 8. African Language (Alpha) Bonus

Shona-language functionality is a core, load-bearing feature of the tutor — not a cosmetic translation layer:
- All 21 syllabus entries include verified Shona explanations, code examples, and common-mistake notes
- All 21 topics include curated Shona practice questions, generated through the same retrieval pipeline used for English
- The system correctly distinguishes between reliable and unreliable Shona input scenarios (Section 5) rather than overclaiming capability

This directly targets the challenge's African Alpha Bonus and Best Localisation Award criteria.

## 9. Known Limitations

- Question input in pure Shona (without an embedded English technical term) is not currently supported and correctly falls back rather than guessing.
- Shona responses (explanations and practice questions) are curated and fixed rather than dynamically generated, by deliberate design (Section 4.2).
- The syllabus is intentionally scoped to 21 core CS concepts; questions outside this scope correctly trigger a fallback rather than receiving substantive tutoring.
- Generation speed (~6.8 tokens/sec) is on the slower end for CPU-only inference of a 3.8B model; this is documented rather than hidden, since Speed is a scored, relative metric.

## 10. Team

Tsungirirai Machingura - SOftware Engineerig Student at African Leadership University , Rwanda

Malvin. T Machingura - Software Engineering Student at Bindura Unviersity
