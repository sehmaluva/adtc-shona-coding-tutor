import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/syllabus.index")

with open("data/syllabus_map.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)

llm = Llama(
    model_path="./model/gemma-2-2b-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

SHONA_APOLOGY_PREFIX = """Ndine urombo, handikwanisi kutsanangura nechiShona nekuti
handisati ndadzidziswa nezvechinhu ichi muchiShona. Asi ndinogona kukupa
tsanangudzo muChirungu (English):
"""

DISTANCE_THRESHOLD = 1.6


def retrieve(question, top_k=1):
    """Find the closest syllabus entry and return it with its distance score."""
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return syllabus[indices[0][0]], distances[0][0]


def answer_question(question, language="english"):
    """
    Retrieve context, then generate/return a grounded answer.
    - In-scope (confident match): grounded on curated syllabus content.
    - Out-of-scope, English: the model answers using its own general
      coding knowledge (not syllabus-grounded), so the tutor can still
      respond sensibly to questions outside the curated 21 topics.
    - Out-of-scope, Shona: returns a curated fallback message rather
      than attempting ungrounded Shona generation, which we've confirmed
      is unreliable (see REPORT.md).
    """
    entry, distance = retrieve(question)
    in_scope = distance <= DISTANCE_THRESHOLD

    if language == "shona":
        if in_scope:
            return f"""Musoro: {entry['topic']}

Tsanangudzo: {entry['shona_explanation']}

Muenzaniso wekodhi:
{entry['example_code']}

Zvinowanzokanganisa: {entry['common_mistakes']}"""
        # Out-of-scope Shona: apologize in Shona, then give a real
        # English answer from the model's general knowledge, rather
        # than refusing outright.
        prompt = f"""<start_of_turn>user
You are a helpful coding and computer science tutor for beginners.
Answer the student's question clearly, simply, and accurately.

Student question: {question}
<end_of_turn>
<start_of_turn>model
"""
        output = llm(prompt, max_tokens=300, stop=["<|end|>"], echo=False)
        english_answer = output["choices"][0]["text"].strip()
        return SHONA_APOLOGY_PREFIX + "\n" + english_answer

    # English
    if in_scope:
        context = f"""Topic: {entry['topic']}
Explanation: {entry['english_explanation']}
Example code:
{entry['example_code']}
Common mistakes: {entry['common_mistakes']}"""

        prompt = f"""<start_of_turn>user
You are a helpful coding tutor. Use the following reference material to answer the student's question clearly and simply.

Reference:
{context}

Student question: {question}
<end_of_turn>
<start_of_turn>model
"""
    else:
        # Out-of-scope: no curated reference material available.
        # Let the model answer from its own general coding knowledge.
        prompt = f"""<start_of_turn>user
You are a helpful coding and computer science tutor for beginners.
Answer the student's question clearly, simply, and accurately.

Student question: {question}
<end_of_turn>
<start_of_turn>model
"""

    output = llm(prompt, max_tokens=300, stop=["<|end|>"], echo=False)
    return output["choices"][0]["text"].strip()


def generate_practice_questions(topic_query, language="english", num_questions=3):
    """
    Return practice questions for a topic.
    - English, in-scope: generated live using retrieved syllabus context.
    - English, out-of-scope: generated live using the model's general
      knowledge of the requested topic.
    - Shona: returned directly from curated, human-verified questions
      when in-scope; curated fallback otherwise (the base model does
      not reliably generate Shona - see REPORT.md).
    """
    entry, distance = retrieve(topic_query)
    in_scope = distance <= DISTANCE_THRESHOLD

    if language == "shona":
        if in_scope:
            questions = entry.get("practice_questions_shona", [])
            if questions:
                header = f"Mibvunzo yekudzidzira: {entry['topic']}\n"
                numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                return header + "\n" + numbered
        # Out-of-scope (or no curated Shona questions available):
        # apologize in Shona, then generate English practice questions.
        prompt = f"""<start_of_turn>user
Generate {num_questions} short beginner-level practice questions about: {topic_query}
Number them 1, 2, 3. Do not include answers, only the questions.
<end_of_turn>
<start_of_turn>model
"""
        output = llm(prompt, max_tokens=250, stop=["<|end|>"], echo=False)
        english_questions = output["choices"][0]["text"].strip()
        return SHONA_APOLOGY_PREFIX + "\n" + english_questions

    # English
    if in_scope:
        context = f"""Topic: {entry['topic']}
Explanation: {entry['english_explanation']}
Example code:
{entry['example_code']}"""
        prompt = f"""<start_of_turn>user
Based on the following CS topic, generate {num_questions} short practice questions
a beginner student could answer to test their understanding. Number them 1, 2, 3.
Do not include answers, only the questions.

{context}
<end_of_turn>
<start_of_turn>model
"""
    else:
        prompt = f"""<start_of_turn>user
Generate {num_questions} short beginner-level practice questions about: {topic_query}
Number them 1, 2, 3. Do not include answers, only the questions.
<end_of_turn>
<start_of_turn>model
"""

    output = llm(prompt, max_tokens=250, stop=["<|end|>"], echo=False)
    return output["choices"][0]["text"].strip()


def ask_mode():
    """Prompt until a valid mode (1 or 2) is entered."""
    while True:
        raw = input("Choose mode - (1) Ask a question, (2) Get practice questions: ").strip()
        if raw in ("1", "2"):
            return raw
        print("Please enter 1 or 2.")


def ask_language():
    """Prompt until a valid language is entered. Accepts shorthand."""
    english_aliases = {"english", "en", "e", "1"}
    shona_aliases = {"shona", "sn", "s", "0"}
    while True:
        raw = input("Language (english/shona, or 1=english, 0=shona): ").strip().lower()
        if raw in english_aliases:
            return "english"
        if raw in shona_aliases:
            return "shona"
        print("Please enter 'english', 'shona', 1, or 0.")


# Suppress harmless llama-cpp-python deallocator error on interpreter exit
sys.unraisablehook = lambda *args: None


if __name__ == "__main__":
    mode = ask_mode()
    lang = ask_language()

    if mode == "2":
        topic_prompt = "Mibvunzo yekudzidzira pamusoro pei? " if lang == "shona" else "Which topic do you want practice questions on? "
        topic = input(topic_prompt)
        print("\n--- Practice Questions ---")
        print(generate_practice_questions(topic, language=lang))
    else:
        question_prompt = "Bvunza mubvunzo wekodhi: " if lang == "shona" else "Ask a coding question: "
        question = input(question_prompt)
        print("\n--- Tutor's Answer ---")
        print(answer_question(question, language=lang))
