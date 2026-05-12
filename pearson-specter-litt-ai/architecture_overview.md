# Architecture Overview

```text
sample_inputs/
   |
   v
DocumentProcessor
   - text/PDF/image acceptance
   - PyMuPDF digital PDF extraction
   - optional Tesseract OCR
   - confidence warnings
   - structured legal JSON extraction
   |
   v
data/processed/*.json
   |
   v
Chunker + VectorStore
   - paragraph/window chunks
   - local sparse TF-IDF embeddings
   - JSON-backed inspectable index
   |
   v
Retriever
   - top-k evidence chunks
   - score, page, source, chunk id metadata
   |
   v
DraftGenerator
   - optional OpenAI LLM prompt
   - deterministic grounded fallback
   - chunk citations and gaps
   |
   v
sample_outputs/*.md + evidence JSON
   |
   v
FeedbackLearner
   - stores original and edited draft pairs
   - extracts reusable preferences
   - updates data/preferences.json
   |
   v
Future DraftGenerator runs apply learned preferences
```

## Component Mapping

1. Document Processing: `DocumentProcessor`, `TesseractOCR`, and `StructuredDataExtractor` in `src/legal_ai_workflow/ingestion.py`.
2. Grounded Retrieval: `Chunker`, `TfidfEmbeddingModel`, and `VectorStore` in `src/legal_ai_workflow/retrieval.py`.
3. Draft Generation: `DraftGenerator` in `src/legal_ai_workflow/generation.py`.
4. Improvement from Edits: `FeedbackLearner` and `PreferenceStore` in `src/legal_ai_workflow/feedback.py`.
5. Interfaces: CLI in `cli.py`; optional REST API in `api.py`.

## Traceability

Every retrieved chunk has an ID like `319c040caa6d-C001`. Draft claims include citations such as `[319c040caa6d-C001; p. 1]`, and each draft writes a separate evidence JSON file with chunk text, score, source file, page, extraction confidence, and extraction method.
