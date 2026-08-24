"""
Tests genuinely off-syllabus coding questions to check for false-positive
matches (questions that should trigger the out-of-scope path, but instead
confidently match an unrelated syllabus topic).

Updated for the 58-topic syllabus: topics the syllabus now legitimately
covers are excluded from this list, since matching them correctly is
expected behavior, not a false positive.
"""
import json
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/syllabus.index")

with open("data/syllabus_map.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)

DISTANCE_THRESHOLD = 1.6

# Genuinely off-syllabus coding questions - topics NOT covered even by
# the expanded 58-topic syllabus.
off_syllabus_questions = [
    "how do I use decorators?",
    "what is a generator function?",
    "how do I use multithreading in Python?",
    "what is a context manager?",
    "how do I connect to a database?",
    "what is dependency injection?",
    "how do I use async/await?",
    "what is a metaclass?",
    "how do I publish a package to PyPI?",
    "what is duck typing?",
]

def retrieve(question, top_k=1):
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return syllabus[indices[0][0]], distances[0][0]

print(f"{'Question':<45} {'Matched To':<35} {'Dist':<8} {'Correct?'}")
print("-" * 100)

false_positives = 0
for q in off_syllabus_questions:
    entry, dist = retrieve(q)
    in_scope = dist <= DISTANCE_THRESHOLD
    status = "❌ FALSE POSITIVE" if in_scope else "✅ correctly out-of-scope"
    if in_scope:
        false_positives += 1
    print(f"{q:<45} {entry['topic']:<35} {dist:<8.3f} {status}")

print("-" * 100)
print(f"False positives: {false_positives}/{len(off_syllabus_questions)}")
