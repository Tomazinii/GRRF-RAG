

# GRRF-RAG

This repository contains a modular system for evaluating different retrieval strategies for legal documents using embeddings, reranking, and multiple ranking-based evaluation strategies. The system supports scalable chunking, reranking, and evaluation with metrics provided by the RAGAS framework.

---

## 🧠 Overview

The application is structured into three main stages:

### 1. **Document Chunking and Indexing**

* Uses sliding window and Markdown-based strategies to chunk legal documents.
* Computes vector embeddings for each chunk using the `BAAI/bge-m3` model via HuggingFace.
* Embeddings are stored in a PostgreSQL database using `Pgvector` for similarity search.

Scripts:

* `app_create_table.py`: Creates the necessary `Pgvector` tables for each document domain.
* `app_index.py`: Chunks and indexes documents into the vector database.

### 2. **Retrieval and Evaluation**

* Given a query, retrieves relevant document chunks based on vector similarity.
* Applies reranking using the `BAAI/bge-reranker-v2-m3` model (via `FlagEmbedding`) to refine the results.
  * **Naive**
  * **GRRF**

Main entry point: `app.py`
Evaluation is parallelized across queries and GPUs for scalability.

### 3. **Evaluation Metrics**

* Evaluation uses the [RAGAS](https://github.com/explodinggradients/ragas) framework.
* The Gemini 2.0 Flash model is used as both a query reformulator and evaluator.

---

## ⚙️ Minimum Environment Requirements

The experimental pipeline was developed in Python and has the following environment dependencies:

* **Programming Language**: Python 3.10+
* **Hardware**:

  * NVIDIA DGX workstation (used in experiments)
  * 2 × NVIDIA V100 GPUs (32 GB each)
* **Libraries & Frameworks**:

  * [`LangChain`](https://python.langchain.com/) for orchestration
  * [`Hugging Face Transformers`](https://huggingface.co/docs/transformers/index) for model loading
  * [`FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding) for reranking
  * [`Pgvector`](https://github.com/pgvector/pgvector) extension in PostgreSQL for dense vector similarity
  * [`Docling`](https://github.com/docling/docling) for PDF/document extraction
  * [`RAGAS`](https://github.com/explodinggradients/ragas) for retrieval quality evaluation
  * [`Pandas`](https://pandas.pydata.org/) for data manipulation
  * [`Tiktoken`](https://github.com/openai/tiktoken) for token counting


## 📝 Notes

* Ensure `CUDA_VISIBLE_DEVICES` is set appropriately before running GPU-based operations.
* PostgreSQL must be running with the `pgvector` extension installed and properly configured.
* A `.env` or exported environment variables must define: `HOST`, `PORT`, `USER`, `PASSWORD`, `DATABASE`.

---

Let me know if you'd like this as a `.md` file or tailored for deployment documentation.
