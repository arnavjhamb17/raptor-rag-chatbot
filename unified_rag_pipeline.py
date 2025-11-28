import os
import json
import uuid
from advanced_parsing import AdvancedPDFParser
from raptor_pipeline import recursive_embed_cluster_summarize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import pandas as pd
import shutil
from pdf_objects import chunk_docx
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# === CONFIG ===
DATA_DIR = "data"
CHUNKS_JSON = "chunks.json"
PERSIST_DIR = "embeddings"
COLLECTION_NAME = "raptor_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_LEVELS = 4

# === 1. Chunk PDFs and DOCX using pdf_objects ===
def chunk_files():
    all_chunks = []
    parser = AdvancedPDFParser(chunk_size=800, overlap=100)
    files_processed = 0
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                print(f"[Chunking] Processing PDF: {file_path}")
                try:
                    chunks = parser.extract_semantic_chunks(file_path)
                    all_chunks.extend(chunks)
                    files_processed += 1
                except Exception as e:
                    print(f"[Chunking] Error processing {file}: {e}")
            elif file.lower().endswith(".docx"):
                print(f"[Chunking] Processing DOCX: {file_path}")
                try:
                    docx_chunks = chunk_docx(file_path)
                    all_chunks.extend(docx_chunks)
                    files_processed += 1
                except Exception as e:
                    print(f"[Chunking] Error processing {file}: {e}")
            
    print(f"[Chunking] Final chunk count: {len(all_chunks)} (from {files_processed} files)")
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    return all_chunks

# === 2. Build Raptor Tree and Summaries ===
def build_raptor_tree(chunks):
    raw_texts = [chunk["text"] for chunk in chunks]
    print(f"[Raptor] Building Raptor tree with {len(raw_texts)} leaf chunks...")
    results = recursive_embed_cluster_summarize(raw_texts, level=1, n_levels=N_LEVELS)
    print(f"[Raptor] Built Raptor tree up to level {N_LEVELS}.")
    return results

# === 3. Store Chunks and Summaries in Vector Database ===
def store_in_vectordb(chunks, raptor_results):
    # Clean slate: delete embeddings directory if it exists
    if os.path.exists(PERSIST_DIR):
        print(f"[VectorDB] Removing existing {PERSIST_DIR}/ directory for a clean slate...")
        try:
            # Make sure the directory is writable before removing
            os.chmod(PERSIST_DIR, 0o755)
            shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            print(f"[VectorDB] Warning during cleanup: {e}")
    
    # Create directory with proper permissions
    os.makedirs(PERSIST_DIR, exist_ok=True)
    os.chmod(PERSIST_DIR, 0o755)
    
    print(f"[VectorDB] Created {PERSIST_DIR}/ directory with proper permissions...")

    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Use FAISS instead of ChromaDB to avoid permission issues
    print(f"[VectorDB] Using FAISS for vector storage...")
    use_faiss = True

    # Prepare all texts and metadata for storage
    print(f"[VectorDB] Preparing {len(chunks)} raw chunks for storage...")
    chunk_metadatas = [
        {
            "id": chunk.get("id", str(uuid.uuid4())),
            "type": "chunk",
            "level": 0,
            **chunk.get("metadata", {})
        }
        for chunk in chunks
    ]
    
    all_texts = [chunk["text"] for chunk in chunks]
    all_metadatas = chunk_metadatas[:]
    
    # Collect summaries for each level
    for level, (_, df_summary) in raptor_results.items():
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
        all_texts.extend(summaries)
        all_metadatas.extend(summary_metadatas)
        print(f"[VectorDB] Added {len(summaries)} summaries from level {level}")

    # Add root summary
    final_level = max(raptor_results.keys())
    root_summary = "\n\n".join(raptor_results[final_level][1]["summaries"])
    root_metadata = {
        "id": str(uuid.uuid4()),
        "type": "summary",
        "level": final_level,
        "cluster": 0,
        "is_root": True
    }
    all_texts.append(root_summary)
    all_metadatas.append(root_metadata)
    print("[VectorDB] Added root summary")
    
    print(f"[VectorDB] Total texts to store: {len(all_texts)}")
    
    try:
        print(f"[VectorDB] Creating FAISS vectorstore with {len(all_texts)} texts...")
        vectorstore = FAISS.from_texts(all_texts, embedding_fn, metadatas=all_metadatas)
        vectorstore.save_local(PERSIST_DIR)
        print(f"[VectorDB] ✅ FAISS successfully stored all {len(all_texts)} texts")
        print(f"[VectorDB] ✅ Vector store saved to {PERSIST_DIR}/")
    except Exception as e:
        print(f"[VectorDB] ❌ Error storing texts: {e}")
        raise

# === MAIN ===
if __name__ == "__main__":
    print("\n=== Unified RAG Pipeline: Chunking → Raptor Tree → Embedding → VectorDB ===\n")
    chunks = chunk_files()
    if not chunks:
        print("[Error] No chunks were generated. Please check your data/ directory contains valid PDF/DOCX files.")
        exit(1)
    raptor_results = build_raptor_tree(chunks)
    if raptor_results:
        store_in_vectordb(chunks, raptor_results)
    print("\n=== Pipeline Complete ===\n")
