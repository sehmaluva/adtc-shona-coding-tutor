"""
Tests genuinely off-syllabus questions phrased as Shona code-switched
questions, to calibrate a safe Shona-specific threshold.
"""
import json
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/syllabus.index")

with open("data/syllabus_map.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)

off_syllabus_shona_questions = [
    "Chii chinonzi Python class?",
    "Ndinoshandisa sei try and except?",
    "Chii chinonzi list comprehension?",
    "Recursion inoshanda sei?",
    "Chii chinonzi object-oriented programming?",
    "Ndinoverenga sei file muPython?",
    "Chii chinonzi lambda function?",
    "Ndinoshandisa sei decorators?",
    "Chii chinonzi inheritance mu OOP?",
    "Ndinoisa sei Python package?",
]

def retrieve(question, top_k=1):
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return syllabus[indices[0][0]], distances[0][0]

print(f"{'Question':<45} {'Matched':<35} {'Dist'}")
print("-" * 90)
for q in off_syllabus_shona_questions:
    entry, dist = retrieve(q)
    print(f"{q:<45} {entry['topic']:<35} {dist:.3f}")
