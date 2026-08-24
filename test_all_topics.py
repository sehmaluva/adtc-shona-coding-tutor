"""
Tests retrieval for every syllabus entry using two phrasings per topic:
1. A natural English question about the topic
2. A code-switched Shona question ("Chii chinonzi <topic>?")

Reports the matched topic and distance for each, flagging any that
don't correctly match themselves or that exceed the fallback threshold.
"""
import json
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/syllabus.index")

with open("data/syllabus_map.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)

DISTANCE_THRESHOLD = 1.6

def retrieve(question, top_k=1):
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return syllabus[indices[0][0]], distances[0][0]

print(f"{'Topic':<40} {'Test Question':<45} {'Matched':<40} {'Dist':<8} {'OK?'}")
print("-" * 145)

fail_count = 0
for entry in syllabus:
    topic = entry["topic"]

    # Test 1: natural English question
    q_en = f"What is {topic}?"
    matched_en, dist_en = retrieve(q_en)
    ok_en = "✅" if matched_en["id"] == entry["id"] and dist_en <= DISTANCE_THRESHOLD else "❌"
    if ok_en == "❌":
        fail_count += 1
    print(f"{topic:<40} {q_en:<45} {matched_en['topic']:<40} {dist_en:<8.3f} {ok_en}")

    # Test 2: code-switched Shona question
    q_sn = f"Chii chinonzi {topic}?"
    matched_sn, dist_sn = retrieve(q_sn)
    ok_sn = "✅" if matched_sn["id"] == entry["id"] and dist_sn <= DISTANCE_THRESHOLD else "❌"
    if ok_sn == "❌":
        fail_count += 1
    print(f"{'':<40} {q_sn:<45} {matched_sn['topic']:<40} {dist_sn:<8.3f} {ok_sn}")
    print()

print("-" * 145)
print(f"Total tests: {len(syllabus) * 2} | Failures: {fail_count}")
