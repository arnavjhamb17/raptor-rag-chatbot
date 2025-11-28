# Raptor-RAG for TDS Guidance

This project delivers a Retrieval-Augmented Generation (RAG) workflow tailored to Tax Deducted at Source (TDS) use cases. It chunks and summarizes statutory PDFs/DOCX, builds a hierarchical FAISS/Chroma knowledge base via a RAPTOR tree, and exposes the knowledge through Streamlit and CLI assistants that combine OCR-driven invoice understanding with Azure OpenAI–powered TDS analysis.

## Repo Highlights

| Path | Purpose |
| --- | --- |
| `unified_rag_pipeline.py` | Orchestrates chunking, recursive Raptor summarization, FAISS persistence |
| `advanced_parsing.py` | Semantic PDF/case-law parser: section typing, overlapping windows, dedupe hashes |
| `pdf_objects.py` | Lightweight PDF/DOCX chunkers used by the pipeline (paragraph + table clustering) |
| `raptor_pipeline.py`, `raptor_setup.py` | Implements recursive embed → cluster → summarize logic and UMAP+GMM clustering |
| `vectorstore_ingest.py` | Alternate script that replays `chunks.json` into Chroma with raw+summary metadata |
| `bfs_backtrack.py` | Hierarchical BFS retriever that walks Raptor tree summaries down to leaf chunks |
| `tds_app1.py` | Streamlit UI performing OCR, invoice extraction, TDS-specific prompting, and context retrieval |
| `tds_persona_query.py` | CLI persona emitting Markdown tables with TDS sections/rates/confidence |
| `chunks.json`, `embeddings/` | Cached semantic chunks and persisted FAISS index (summaries + leaves) |

## Prerequisites

- Python 3.11 (see `.python-version`)
- Tesseract OCR and Poppler/Freetype deps required by `pytesseract`/`fitz`
- Azure OpenAI deployment (GPT-4o/GPT-4 Turbo, etc.)
- Optional GPU for faster embeddings

### Python Dependencies

```bash
pip install -r requirements.txt
pip install python-dotenv pymupdf pandas numpy scikit-learn umap-learn opencv-python-headless langchain langchain-community langchain-chroma langchain-huggingface chromadb faiss-cpu
```

Adjust packages if you prefer GPU-enabled FAISS or OpenCV.

### Environment Variables (`.env`)

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=<deployment>
AZURE_OPENAI_VERSION=2024-10-21
```

`tds_app1.py` validates these on startup.

## Data Preparation Workflow

1. **Place source documents** under `data/` (statutory PDFs, DOCX briefs, case laws, invoices, etc.). Metadata such as section type, source file, and chunk id is preserved in each chunk’s `metadata`.
2. **Chunk content**
   ```bash
   python unified_rag_pipeline.py
   ```
   What happens inside:
   - `advanced_parsing.AdvancedPDFParser` parses PDFs page by page, normalizes whitespace, detects legal structure (section heading, proviso, explanation, tables), and creates overlapping windows for long paragraphs.
   - Case-law PDFs are treated specially (pattern-based splits on numbered items) to retain holdings and ratios.
   - DOCX files are parsed via `pdf_objects.chunk_docx`, capturing paragraphs individually and concatenating all table rows into a single `table_cluster` entry to prevent row fragmentation.
   - Each chunk carries a short MD5 hash for dedupe and enough metadata (`chunk_type`, `page_number`, `source_file`, etc.) to cite later.
   - All chunks are serialized to `chunks.json` (leaf nodes for the later Raptor tree).
3. **Raptor tree & embeddings**
   - `recursive_embed_cluster_summarize` in `raptor_pipeline.py` feeds leaf texts into `raptor_setup.perform_clustering`, which reduces dimensionality with UMAP, runs Gaussian Mixture Models to identify overlapping clusters, and returns list-of-clusters per document.
   - For each cluster, the texts are concatenated and summarized via Azure OpenAI (`model.summarize_cluster`). The result is stored along with level and cluster id, then recursively clustered again to build higher-level summaries (default 4 levels).
   - Every summary and raw chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2`, then persisted to FAISS (`embeddings/`). A root summary (aggregated highest-level text) is appended for better top-level recall.
4. **Alternative ingestion**
   - If you already trust `chunks.json`, run:
     ```bash
     python vectorstore_ingest.py
     ```
     This replays the chunks and Raptor summaries, storing them into Chroma instead of FAISS while tagging each document with `type`, `level`, `cluster`, and unique ids.

## Retrieval Layer

`bfs_backtrack.py` opens the FAISS store and performs a hierarchical BFS:

- Starts at highest-level summaries, descends into matching clusters when cosine similarity exceeds a threshold, and backfills raw chunks.
- Always injects the most relevant `table_cluster` chunk to ensure tabular rates aren’t missed.
- Exported helper `bfs_search(query, threshold=0.5, max_docs=30)` powers both front-end apps.
- The BFS keeps a visited set using `(level, cluster/chunk_index)` tuples so summaries aren’t reprocessed, and augments results with longer paragraph chunks only if they exceed a `MIN_PARAGRAPH_LENGTH` guard to avoid noisy short snippets.
- Because summaries and chunks share cluster ids, `bfs_search` can “walk down” from high-level summary nodes into relevant children with one metadata comparison, avoiding fresh vector searches at each level.

## Applications

### Streamlit Assistant (`tds_app1.py`)

```bash
streamlit run tds_app1.py
```

- Drag/drop image, PDF, or DOCX invoices. Uploaded files start in memory, get converted to PIL, then pass through OpenCV deskewing, adaptive thresholding, denoising, and scaling to boost OCR accuracy (`enhance_image_for_ocr`).
- `pytesseract` extracts text that feeds the `INVOICE_EXTRACTION_PROMPT`, yielding structured JSON (state, city, amount, service description, vendor, invoice number, date, confidence).
- The app keeps the last six `st.session_state["messages"]` entries for conversational continuity, but performs a lightweight classification via `NON_TDS_DETECTION_PROMPT` to decide between conversational/safe responses and tax processing.
- For TDS section/rate questions, `bfs_search` supplies chunked context which is woven into `TDS_ANALYSIS_PROMPT`. The LLM returns JSON with primary/alternate sections (with sub-sections when relevant), rate splits (Individual/HUF vs Others), detailed rationale, confidence, and citations referencing chunk metadata.
- Non-rate or general queries fall back to `GENERAL_TDS_PROMPT`, and the response is forced into paragraph form to avoid overly rigid formatting.
- All responses cite the knowledge base chunk ids, and “unrelated” questions are politely declined per RULE 1 in the general prompt.

### CLI Persona (`tds_persona_query.py`)

```bash
python tds_persona_query.py
```

- Prompts for a question, fetches context with `bfs_search`, and forces the LLM to internally reason in JSON that is then converted to a Markdown table for terminal display.
- Enforces fallback responses for out-of-domain or unclear inputs (“I am sorry…” / “Could you please rephrase...”) per persona instructions.
- Pulls extra guidance when the chosen section is 194J/194C/194H/194I or “No TDS,” referencing linked knowledge-base DOCX files and ensuring sub-sections (e.g., 194J(1)(a)) are spelled out.

## Chunks & Embeddings Management

- `chunks.json` (~280 KB) already contains sample data; regenerate after adding documents.
- `embeddings/` stores FAISS index files; delete to rebuild if schema changes.
- `model.py` centralizes Azure LLM + embedding utilities used by Raptor summarization; ensure API quotas can handle repeated clustering runs.

## Troubleshooting

- **Missing `.env` vars**: `tds_app1.py` and `model.py` raise descriptive errors if Azure credentials are absent.
- **FAISS load failures**: Ensure `embeddings/` exists, matches the embedding model, and that `allow_dangerous_deserialization=True` is acceptable (toggle off in production). Rebuild the store after deleting stale artifacts.
- **OCR quality**: Install language packs and set `pytesseract.pytesseract.tesseract_cmd` if Streamlit cannot locate Tesseract.
- **Large PDFs**: Tune `chunk_size`/`overlap` in `AdvancedPDFParser` or chunk documents individually to manage memory.
- **Package gaps**: `requirements.txt` is minimal; install additional dependencies listed above if you hit `ModuleNotFoundError`.

## Next Steps

- Add Dockerfile/Makefile for reproducible setups.
- Expand automated tests for parsers and BFS retrieval.
- Instrument logging/metrics (latency, token usage) inside the Streamlit app.
- Consider migrating FAISS persistence to a managed vector DB for multi-user serving.
- Add monitoring around Azure OpenAI token usage during Raptor runs (summaries can be expensive) and explore batching or caching cluster summaries to cut costs.

