# REPORT.md â€” Offline Shona AI Coding Tutor

## Africa Deep Tech Challenge 2026 â€” The Laptop LLM Challenge

**Track:** Math & Scientific Reasoning (Coding Assistants)
**Repository:** github.com/tmachingur-code/adtc-shona-coding-tutor

---

## 1. Problem Definition

Access to coding education across much of Africa is constrained by three linked barriers: unreliable or expensive internet connectivity, the cost of cloud-based AI tutoring subscriptions, and the fact that most instructional material â€” including AI tutoring tools â€” is delivered exclusively in English, disadvantaging students who are more comfortable learning technical concepts in a local language.

This is a first-hand, not hypothetical, problem: our team draws on lived experience from Gokwe North, Zimbabwe, a rural community where consistent connectivity and subscription-based tools are not a given, and where Shona is the primary language of everyday communication.

**Our solution** is an offline, on-device coding tutor that:
- Runs entirely locally on modest, widely-available laptop hardware, with no internet dependency and no API costs
- Teaches foundational Python programming and computer science reasoning (variables, control flow, functions, data structures, algorithms, and debugging)
- Provides explanations in both English and Shona, making CS education accessible in a familiar language

## 2. Constraints

The application is designed and tested against the ADTC Standard Laptop profile:

| Constraint | Target |
|---|---|
| RAM ceiling | 7 GB (hard limit â€” disqualification if exceeded) |
| CPU | Intel i5 10thâ€“12th gen / AMD Ryzen 5 3000â€“5000, no discrete GPU |
| OS | Ubuntu 22.04 LTS |
| Storage | 256 GB SSD |

All development and testing was carried out in a WSL2 Ubuntu environment to closely mirror the target OS, using CPU-only inference throughout (no CUDA/GPU dependencies were installed).

## 3. System Architecture

The system combines three components into a single reasoning pipeline:

1. **Local LLM (generation):** Phi-3.5-mini-instruct, quantized to Q4_K_M GGUF format, run via `llama-cpp-python` (CPU-only, no GPU offload)
2. **Retrieval layer (RAG):** a curated syllabus of 20 CS concept entries is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`, ~80MB) and indexed with FAISS (`IndexFlatL2`) for fast local similarity search
3. **Bilingual response layer:** English responses are generated live by the LLM using retrieved context; Shona responses are returned directly from curated, human-written explanations (see Design Decision 3.2 below)

**Flow:** student question â†’ embedded and matched against the syllabus index â†’ if a confident match is found, relevant context is retrieved â†’ English mode passes this context to the LLM for a generated answer; Shona mode returns the pre-written Shona explanation directly â†’ if no confident match is found, a graceful fallback message (in the requested language) is returned instead of a guess.

## 4. Key Design Decisions

### 4.1 RAG over fine-tuning
We deliberately chose retrieval-augmented generation instead of fine-tuning. Fine-tuning would require substantial training data, GPU compute, and time that are incompatible with both our hardware constraints and development timeline. RAG lets an unmodified, general-purpose small model (Phi-3.5-mini) behave like a scoped CS tutor by supplying relevant reference material at answer time â€” directly matching the challenge brief's systems mandate.

### 4.2 Curated Shona explanations instead of model-generated Shona
During testing, we found that asking Phi-3.5-mini to freely generate Shona text produced incoherent, unreliable output â€” the base model has limited Shona-language training data. Rather than presenting unreliable generated text, we made a deliberate engineering decision: Shona responses are retrieved directly from a human-written, verified knowledge base rather than generated live. This guarantees linguistic accuracy for every Shona response, at the cost of Shona output being fixed rather than freely generated. We consider this the more honest and higher-quality approach for an education tool, where incorrect explanations carry real cost to a learner.

### 4.3 Retention of English technical terminology within Shona explanations
Programming terms (e.g., "for loop," "function," "list") are retained in English within our Shona explanations. This reflects common technical Shona usage, where such terms are widely used as-is even in fluent Shona technical discussion, rather than being an incomplete localisation.

### 4.4 Distance-threshold fallback for out-of-scope questions
Initial testing revealed that returning the closest syllabus match regardless of relevance risked confidently answering questions our syllabus does not cover. We introduced a similarity-distance threshold (tuned empirically to 1.4 using FAISS L2 distance) so that questions falling outside our scoped syllabus return a clear, honest fallback message â€” in the student's requested language â€” rather than a mismatched answer.

We validated this threshold against three cases:
- Exact-phrasing on-topic question: distance â‰ˆ 0.42 (correctly matched)
- Paraphrased on-topic question: distance â‰ˆ 1.10 (correctly matched)
- Clearly off-topic question: distance â‰ˆ 1.94 (correctly triggered fallback)

### 4.5 Scoped syllabus rather than open-domain tutoring
The knowledge base is intentionally narrow: 20 concept entries across six modules (Python Basics, Control Flow, Functions, Data Structures, Algorithms & Reasoning, Debugging Fundamentals). This scope was chosen to maximize retrieval accuracy and keep the RAG index lightweight, rather than attempting broad CS coverage that would dilute both accuracy and hardware efficiency.

## 5. Known Limitations

- **Question input is currently English-only.** The embedding model was not validated for Shona-language question matching, so students must currently phrase questions in English even when requesting a Shona-language response. Extending robust bilingual question input is identified as future work.
- **Shona responses are fixed, not generative.** This is an intentional accuracy/reliability tradeoff (see 4.2), meaning Shona answers cannot dynamically adapt phrasing the way English answers can.
- **Syllabus scope is currently fixed at 20 entries.** Questions outside this scope correctly trigger a fallback rather than a wrong answer, but do not yet receive substantive tutoring.

## 6. Tools & Technology Stack

| Component | Tool |
|---|---|
| LLM inference | llama-cpp-python + Phi-3.5-mini-instruct (Q4_K_M GGUF) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (IndexFlatL2), CPU-only |
| Language | Python 3.14 |
| Development environment | WSL2 Ubuntu (mirroring Ubuntu 22.04 LTS target) |

## 7. Benchmarks

*Benchmarking against the ADTC local profiler (peak RAM, tokens/sec, thermal behavior) is in progress. This section will be completed with measured figures prior to Gate 1 submission.*

| Metric | Status |
|---|---|
| Peak RAM usage | Pending measurement |
| Tokens/sec (generation speed) | Pending measurement |
| Thermal behavior | Pending measurement |

## 8. African Language (Alpha) Bonus

Shona-language functionality is a core, load-bearing feature of the tutor â€” not a cosmetic translation layer. All 20 syllabus entries include verified Shona explanations, code examples, and common-mistake notes, retrievable through the same RAG pipeline used for English. This directly targets the challenge's African Alpha Bonus and Best Localisation Award criteria.

## 9. Team

Tsungirirai Machingura
Malvin T Machingura
