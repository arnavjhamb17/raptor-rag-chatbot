import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

PERSIST_DIR = "embeddings"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load FAISS vector store
print("[BFS] Loading FAISS vector store...")
try:
    vectorstore = FAISS.load_local(PERSIST_DIR, embedding_fn, allow_dangerous_deserialization=True)
    print(f"[BFS] ✅ Successfully loaded {vectorstore.index.ntotal} vectors from FAISS")
except Exception as e:
    print(f"[BFS] ❌ Error loading FAISS: {e}")
    vectorstore = None

def retrieve_all_documents():
    if vectorstore is None:
        print("[BFS] ❌ Vector store not loaded")
        return []
    # Get a large number of documents to simulate "all" documents
    return vectorstore.similarity_search("document content", k=10000)

def cosine_score(query_emb, doc_emb):
    return cosine_similarity([query_emb], [doc_emb])[0][0]

MIN_PARAGRAPH_LENGTH = 100  # Minimum length for paragraph chunks

def bfs_search(query: str, threshold: float = 0.5, max_docs: int = 30):
    """
    Hierarchical BFS backtracking search for relevant context nodes.
    Returns a list of LangChain Document objects (summaries and/or chunks).
    """
    # Check if vectorstore is loaded
    if vectorstore is None or vectorstore.index.ntotal == 0:
        print("[BFS] ⚠️ Warning: Vector store is empty or not loaded")
        return []

    query_emb = embedding_fn.embed_query(query)
    docs = retrieve_all_documents()

    if not docs:
        print("[BFS] ⚠️ Warning: No documents found in vector store")
        return []

    # Organize nodes by level
    levels = {}
    for doc in docs:
        level = doc.metadata.get("level", -1)
        levels.setdefault(level, []).append(doc)

    if not levels:
        print("[BFS] ⚠️ Warning: No valid levels found in documents")
        return []

    max_level = max(levels.keys())

    context_nodes = []
    queue = [(max_level, doc) for doc in levels[max_level]]

    visited = set()
    while queue and len(context_nodes) < max_docs:
        next_queue = []
        for level, doc in queue:
            doc_id = (level, doc.metadata.get("cluster", doc.metadata.get("chunk_index", -1)))
            if doc_id in visited:
                continue
            visited.add(doc_id)

            doc_emb = embedding_fn.embed_query(doc.page_content)
            score = cosine_score(query_emb, doc_emb)

            if score >= threshold:
                context_nodes.append(doc)
                if level > 0:
                    children = levels.get(level - 1, [])
                    next_queue.extend([
                        (level - 1, child)
                        for child in children
                        if child.metadata.get("cluster") == doc.metadata.get("cluster")
                    ])
        queue = next_queue

    # Append raw chunks (level 0)
    chunks = [doc for doc in levels.get(0, []) if doc not in context_nodes]
    chunks_sorted = sorted(
        chunks,
        key=lambda d: cosine_score(query_emb, embedding_fn.embed_query(d.page_content)),
        reverse=True
    )
    context_nodes.extend(chunks_sorted[:max(1, max_docs - len(context_nodes))])

    # === Filter paragraph chunks by minimum length ===
    filtered_context_nodes = []
    for doc in context_nodes:
        if doc.metadata.get("chunk_type") == "paragraph":
            if len(doc.page_content) >= MIN_PARAGRAPH_LENGTH:
                filtered_context_nodes.append(doc)
        else:
            filtered_context_nodes.append(doc)
    context_nodes = filtered_context_nodes

    # === Always include the most relevant table_cluster chunk ===
    table_clusters = [doc for doc in docs if doc.metadata.get("chunk_type") == "table_cluster"]
    if table_clusters:
        best_table = max(
            table_clusters,
            key=lambda d: cosine_score(query_emb, embedding_fn.embed_query(d.page_content))
        )
        if best_table not in context_nodes:
            context_nodes.append(best_table)

    return context_nodes

if __name__ == "__main__":
    if vectorstore is None:
        print("[BFS] Cannot run test - vector store not loaded")
        exit(1)
        
    # Test query for TDS
    query = "What are the TDS provisions for revamping services?"
    
    print(f"\n{'='*80}")
    print(f"🔍 QUERY: {query}")
    print(f"{'='*80}")
    
    # Use lower threshold to get more results and higher max_docs
    results = bfs_search(query, threshold=0.2, max_docs=15)
    
    print(f"\n📊 FOUND {len(results)} RELEVANT DOCUMENTS:")
    print(f"📚 Total vectors in store: {vectorstore.index.ntotal}")
    print("-" * 80)
    
    # Group results by type and level for better organization
    summaries = []
    chunks = []
    
    for doc in results:
        doc_type = doc.metadata.get('type', 'unknown')
        if doc_type == 'summary':
            summaries.append(doc)
        else:
            chunks.append(doc)
    
    # Sort summaries by level (highest first)
    summaries.sort(key=lambda x: x.metadata.get('level', 0), reverse=True)
    
    print(f"\n🏗️  HIERARCHICAL SUMMARIES ({len(summaries)} found):")
    print("=" * 60)
    
    for i, doc in enumerate(summaries):
        metadata = doc.metadata
        level = metadata.get('level', 'N/A')
        cluster = metadata.get('cluster', 'N/A')
        is_root = metadata.get('is_root', False)
        
        print(f"\n📋 SUMMARY #{i+1} {'(ROOT)' if is_root else ''}")
        print(f"📊 Level: {level} | 🔗 Cluster: {cluster}")
        print(f"📏 Length: {len(doc.page_content)} characters")
        print(f"\n📄 FULL CONTENT:")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)
    
    print(f"\n📝 RAW CHUNKS ({len(chunks)} found):")
    print("=" * 60)
    
    for i, doc in enumerate(chunks):
        metadata = doc.metadata
        doc_id = metadata.get('id', 'N/A')
        
        print(f"\n📄 CHUNK #{i+1}")
        print(f"🆔 ID: {doc_id}")
        print(f"📏 Length: {len(doc.page_content)} characters")
        
        # Show any additional metadata
        extra_metadata = {k: v for k, v in metadata.items() if k not in ['type', 'level', 'id']}
        if extra_metadata:
            print(f"🏷️  Extra metadata: {extra_metadata}")
        
        print(f"\n📄 FULL CONTENT:")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)


    for doc in results:
        print("Text:",doc.page_content)
        print("-" * 40)
        print("Metadata:", doc.metadata)
