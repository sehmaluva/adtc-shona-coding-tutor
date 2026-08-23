import json
import numpy as np
import faiss

from embedder import load_embedder

# Load syllabus
with open("data/syllabus.json", "r", encoding="utf-8") as f:
    syllabus = json.load(f)

# Load small embedding model from model/all-MiniLM-L6-v2 (offline)
embedder = load_embedder()

# Combine topic + english_explanation for embedding (this is what gets searched)
texts = [f"{entry['topic']}. {entry['english_explanation']}" for entry in syllabus]

# Generate embeddings
embeddings = embedder.encode(texts, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and syllabus mapping
faiss.write_index(index, "data/syllabus.index")

with open("data/syllabus_map.json", "w", encoding="utf-8") as f:
    json.dump(syllabus, f, ensure_ascii=False, indent=2)

print(f"Index built successfully with {len(syllabus)} entries.")
