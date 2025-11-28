import json
import os
import uuid
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from raptor_pipeline import recursive_embed_cluster_summarize
from typing import List

# === Setup ChromaDB ===
PERSIST_DIR = "embeddings"
COLLECTION_NAME = "raptor_index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Remove old collection if exists
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    persist_directory=PERSIST_DIR
)
vectorstore._collection.delete(where={})  # Clear all docs

# === Load your document chunks ===
with open("chunks.json", "r") as f:
    chunks = json.load(f)

raw_texts = [chunk["text"] for chunk in chunks]
raw_metadata = [chunk["metadata"] for chunk in chunks]

# === Step 1: Embed and store raw chunks ===
print(f"Storing {len(raw_texts)} raw chunks...")
chunk_metadatas = [
    {
        "id": str(uuid.uuid4()),
        "type": "chunk",
        "level": 0,
        **md
    }
    for md in raw_metadata
]
vectorstore.add_texts(raw_texts, metadatas=chunk_metadatas)

# === Step 2: Run Raptor Tree and embed summaries ===
print("Running Raptor Tree...")
results = recursive_embed_cluster_summarize(raw_texts, level=1, n_levels=4)

for level, (_, df_summary) in results.items():
    summaries = df_summary["summaries"].tolist()
    summary_metadatas = [
        {
            "id": str(uuid.uuid4()),
            "type": "summary",
            "level": level,
            "cluster": int(df_summary["cluster"].iloc[i])
        }
        for i in range(len(summaries))
    ]
    print(f"Storing {len(summaries)} summaries from level {level}...")
    vectorstore.add_texts(summaries, metadatas=summary_metadatas)

# === Step 3: Store root summary as special entry ===
final_level = max(results.keys())
root_summary = "\n\n".join(results[final_level][1]["summaries"])
root_metadata = {
    "id": str(uuid.uuid4()),
    "type": "summary",
    "level": final_level,
    "cluster": 0,
    "is_root": True
}
vectorstore.add_texts([root_summary], metadatas=[root_metadata])

vectorstore.persist()
print("✅ All chunks and summaries have been embedded and stored in ChromaDB.")






