# Assumptions and Tradeoffs

## Assumptions

- The reviewer may run this without paid APIs or installed vector databases.
- Synthetic legal documents are acceptable for `sample_inputs/`.
- Legal correctness is out of scope; the goal is grounded, traceable first-pass drafting.
- Tesseract may not be installed. The system should flag OCR-dependent files instead of failing.
- Operator feedback can be represented as reusable drafting preferences plus stored edit examples.

## Simplifications

- The default embedding model is local sparse TF-IDF. Optional sentence-transformers embeddings are supported, but semantic recall still trails domain-tuned legal models.
- OCR confidence is a mix of Tesseract TSV confidence when available and conservative heuristics for embedded text.
- Confidence scoring is heuristic-only and not statistically calibrated; it is intended for demo triage rather than formal quality assurance.
- Structured extraction uses deterministic regex and keyword rules. It is easy to audit, but not as flexible as a document-understanding model.
- The deterministic draft generator is extractive. It prioritizes support and citations over polished language when no LLM key is available.
- The REST API accepts local paths for feedback rather than implementing multipart upload.

## Why This Design

- Each component is swappable: replace `TfidfEmbeddingModel` with OpenAI embeddings, `VectorStore` with FAISS/Chroma, or `OpenAIResponsesClient` with another LLM adapter without rewriting ingestion or feedback learning.
- Processed JSON is the contract between ingestion and retrieval. It includes raw text, structured fields, page metadata, warnings, and confidence values.
- Feedback learning affects future drafts directly. The sample run learns bullet preference, risk-section preference, strict citation behavior, and section order; the improved draft changes accordingly.

## Known Limits

- Handwriting recognition depends on external OCR quality. Without Tesseract or a cloud OCR provider, image-only samples are flagged as low/no evidence.
- Current chunking is paragraph/window based. For very long contracts, section-aware parsing would improve evidence precision.
- The local generator can overquote source-like sentences because it is extractive. With an LLM configured, the prompt can produce smoother prose while preserving citations.
- Learned preferences are global rather than per-reviewer or per-draft-type. A production system should scope preferences by user, client, jurisdiction, and draft type.

## Limitations and Future Work

- Add hybrid retrieval and reranking (BM25 + semantic + cross-encoder) to improve recall on long or ambiguous queries.
- Upgrade OCR with layout-aware parsing (tables, headers, stamps) and confidence calibration for better auditability.
- Introduce per-operator preference profiles and allow opt-in templates by draft type.
- Expand evaluation with human review rubrics, fact-level precision/recall, and longitudinal improvement metrics.
- Add structured field extraction models for parties, dates, and obligations beyond regex heuristics.
