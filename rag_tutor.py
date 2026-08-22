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

SHONA_FALLBACK = """Ndine urombo, handisati ndadzidziswa nezvechinhu ichi.
Ndiri kugona kubatsira nezve: variable, if/else, for loop, while loop,
function, list, dictionary, string, sorting, searching, uye kuongorora
zvikanganiso (debugging). Bvunza mumwe mubvunzo une chekuita nezvinhu izvi."""

ENGLISH_FALLBACK = """I don't have information on that topic in my current knowledge base.
I can help with: variables, if/else, loops, functions, lists, dictionaries,
strings, sorting, searching, and debugging common errors.
Try asking about one of these!"""

DISTANCE_THRESHOLD = 1.4


def retrieve(question, top_k=1):
    """Find the closest syllabus entry and return it with its distance score."""
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return syllabus[indices[0][0]], distances[0][0]


def answer_question(question, language="english"):
    """Retrieve context, then generate/return a grounded answer."""
    entry, distance = retrieve(question)

    if distance > DISTANCE_THRESHOLD:
        return SHONA_FALLBACK if language == "shona" else ENGLISH_FALLBACK

    if language == "shona":
        return f"""Musoro: {entry['topic']}

Tsanangudzo: {entry['shona_explanation']}

Muenzaniso wekodhi:
{entry['example_code']}

Zvinowanzokanganisa: {entry['common_mistakes']}"""

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

    output = llm(prompt, max_tokens=300, stop=["<|end|>"], echo=False)
    return output["choices"][0]["text"].strip()


def generate_practice_questions(topic_query, language="english", num_questions=3):
    """
    Return practice questions for a topic.
    - English: generated live by the model using retrieved context.
    - Shona: returned directly from curated, human-verified questions
      (the base model does not reliably generate Shona - see REPORT.md).
    """
    entry, distance = retrieve(topic_query)

    if distance > DISTANCE_THRESHOLD:
        return SHONA_FALLBACK if language == "shona" else ENGLISH_FALLBACK

    if language == "shona":
        questions = entry.get("practice_questions_shona", [])
        if not questions:
            return SHONA_FALLBACK
        header = f"Mibvunzo yekudzidzira: {entry['topic']}\n"
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        return header + "\n" + numbered

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

    output = llm(prompt, max_tokens=250, stop=["<|end|>"], echo=False)
    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    mode = input("Choose mode - (1) Ask a question, (2) Get practice questions: ").strip()
    lang = input("Language (english/shona): ").strip().lower()

    if mode == "2":
        topic = input("Which topic do you want practice questions on? ")
        print("\n--- Practice Questions ---")
        print(generate_practice_questions(topic, language=lang))
    else:
        question = input("Ask a coding question: ")
        print("\n--- Tutor's Answer ---")
        print(answer_question(question, language=lang))
