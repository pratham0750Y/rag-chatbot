# Domain-Specific RAG Chatbot for PDF Question Answering

Upload your PDFs and ask questions. The chatbot answers **only** from the retrieved passages and shows the source file and page. If the answer is not in the documents, it says so.

## How it works

```
Upload PDFs -> extract text per page -> split into chunks -> embeddings (MiniLM)
-> FAISS index -> question -> top 4 chunks -> Groq LLM -> answer + sources
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (upload, chat, sources) |
| `document_loader.py` | PDF validation and page-by-page text extraction |
| `vector_store.py` | chunking, embeddings and the FAISS index |
| `rag_pipeline.py` | retrieval + answer generation |
| `prompt.py` | strict prompt and the "not found" message |
| `evaluate.py` | runs the question-answer dataset and prints a report |
| `tests/test_cases.csv` | 22 evaluation questions for the PDFs in `documents/` |
| `tests/test_rag.py` | automatic tests (run offline) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Groq key in it (or keep your existing `.env`).

## Run

```bash
streamlit run app.py
```

Upload PDFs in the sidebar, click **Process documents** and start asking questions. Uploaded files are saved in `data/uploads/` and the index in `vector_store/saved_index/`, so they are still there after a restart.

## Evaluation

```bash
python evaluate.py
```

It indexes the PDFs in `documents/`, asks every question in `tests/test_cases.csv`, and reports retrieval accuracy (right file and page), answer correctness (expected keywords), correct refusals and response time. The sheet is saved as `evaluation_results.csv`. If good questions get refused, lower `MIN_SCORE` (default 0.15, can be set in `.env`).

## Tests

```bash
pip install pytest
pytest
```


