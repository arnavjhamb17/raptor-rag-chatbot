# raptor_tree_pipeline.py

import json
import pandas as pd
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# === Required: your custom embedding and clustering modules ===
from raptor_setup import perform_clustering  # Correct import
from model import summarize_cluster, embed   # Import embed from model

# === Setup Chroma DB with HuggingFace ===
PERSIST_DIR = "embeddings"
COLLECTION_NAME = "raptor_index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=embedding_model_name)
chroma = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedding_fn, persist_directory=PERSIST_DIR)
# If you want a clean slate, manually delete the 'embeddings/' directory before running this script.

def store_to_chroma(texts: List[str], metadatas: List[dict]):
    chroma.add_texts(texts, metadatas=metadatas)

# === Step 1: Embed + Cluster ===
def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    embeddings = embed(texts)
    clusters = perform_clustering(embeddings, dim=10, threshold=0.1)
    df = pd.DataFrame({
        "text": texts,
        "embd": list(embeddings),
        "cluster": clusters
    })
    return df

# === Step 2: Summarize clusters ===
def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_clusters = embed_cluster_texts(texts)
    expanded = []
    for _, row in df_clusters.iterrows():
        for c in row["cluster"]:
            expanded.append({"text": row["text"], "embd": row["embd"], "cluster": c})
    expanded_df = pd.DataFrame(expanded)
    all_clusters = expanded_df["cluster"].unique()

    summaries = []
    for c in all_clusters:
        cluster_df = expanded_df[expanded_df["cluster"] == c]
        cluster_text = "\n".join(cluster_df["text"].tolist())
        summary = summarize_cluster(cluster_text)
        summaries.append(summary)

    df_summary = pd.DataFrame({
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    })

    # Store summaries into Chroma
    store_to_chroma(
        texts=summaries,
        metadatas=[
            {
                "id": str(uuid.uuid4()),
                "type": "summary",
                "level": level,
                "cluster": row["cluster"]
            }
            for _, row in df_summary.iterrows()
        ]
    )

    return df_clusters, df_summary

# === Step 3: Recursive Raptor Tree ===
def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = 4) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    results = {}
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    print(f"[DEBUG] Level {level}: {df_summary['cluster'].nunique()} clusters")
    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        next_texts = df_summary["summaries"].tolist()
        next_results = recursive_embed_cluster_summarize(next_texts, level + 1, n_levels)
        results.update(next_results)

    return results

# === MAIN: Run on parsed JSON chunks ===
if __name__ == "__main__":
    # Load your JSON chunk file
    with open("chunks.json", "r") as f:
        chunks = json.load(f)

    # Extract texts from chunks
    leaf_texts = [chunk["text"] for chunk in chunks if "text" in chunk]
    leaf_metadatas = [
        {
            "id": str(uuid.uuid4()),
            "type": "chunk",
            "level": 0,
            **chunk.get("metadata", {})
        }
        for chunk in chunks
    ]

    # Store raw chunks into Chroma
    store_to_chroma(
        texts=leaf_texts,
        metadatas=leaf_metadatas
    )

    # Run the full Raptor Tree pipeline
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=4)

    # Print total clusters across all levels
    total_clusters = sum(df_sum["cluster"].nunique() for _, df_sum in results.values())
    print(f"[DEBUG] Total clusters across all levels: {total_clusters}")

    # Save final root summary
    final_level = max(results.keys())
    root_summary = "\n\n".join(results[final_level][1]["summaries"])
    print("\n========= ROOT SUMMARY =========\n")
    print(root_summary)
    root_metadata = {
        "id": str(uuid.uuid4()),
        "type": "summary",
        "level": final_level,
        "cluster": 0,
        "is_root": True
    }
    store_to_chroma([root_summary], [root_metadata])

    # Optionally save all summaries and clusters
    for lvl, (df_clust, df_sum) in results.items():
        df_clust.to_csv(f"level_{lvl}_clusters.csv", index=False)
        df_sum.to_csv(f"level_{lvl}_summaries.csv", index=False)
        print(f"Level {lvl}: saved clusters and summaries.")
