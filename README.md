# Pearson Specter Litt Legal AI Workflow

End-to-end take-home assessment implementation for messy legal document ingestion, grounded retrieval, first-pass memo drafting, and improvement from operator edits.

The project is intentionally review-friendly:

- It runs offline with Python standard library retrieval and deterministic grounded drafting.
- It uses PyMuPDF/Pillow when available for PDF and image samples.
- It uses Tesseract OCR automatically when the `tesseract` binary is installed.
- It can call an LLM through `OPENAI_API_KEY`, but never requires secrets for the demo path.

## Quick Start

```bash
cd pearson-specter-litt-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m legal_ai_workflow.cli run-demo --reset
```

Generated artifacts:

- Processed document JSON: `data/processed/`
- Retrieval index: `data/index/`
- Baseline draft: `sample_outputs/baseline_draft.md`
- Simulated operator edit: `sample_outputs/operator_edited_draft.md`
- Learned preferences: `data/preferences.json`
- Improved draft: `sample_outputs/improved_draft_after_feedback.md`
- Second-round operator edit: `sample_outputs/operator_edited_draft_round2.md`
- Second-round improved draft: `sample_outputs/improved_draft_after_feedback_round2.md`
- Second-round evidence: `sample_outputs/improved_evidence_round2.json`

## Commands

Process one file or a directory:

```bash
PYTHONPATH=src python -m legal_ai_workflow.cli process sample_inputs --output-dir data/processed
```

Build the retrieval index:

```bash
PYTHONPATH=src python -m legal_ai_workflow.cli index \
  --processed-dir data/processed \
  --index-dir data/index \
  --target-tokens 60 \
  --overlap-tokens 20
```

Use optional semantic embeddings (sentence-transformers):

```bash
pip install sentence-transformers
PYTHONPATH=src python -m legal_ai_workflow.cli index \
  --processed-dir data/processed \
  --index-dir data/index \
  --embedding-model sentence-transformers \
  --embedding-model-name all-MiniLM-L6-v2
```

Generate a grounded first-pass internal memo:

```bash
PYTHONPATH=src python -m legal_ai_workflow.cli draft \
  "Prepare a first-pass internal memo summarizing title and notice risks." \
  --index-dir data/index \
  --preferences data/preferences.json \
  --output sample_outputs/custom_draft.md \
  --min-score 0.0 \
  --min-confidence 0.55
```

Learn from an operator-edited draft:

```bash
PYTHONPATH=src python -m legal_ai_workflow.cli feedback \
  --original sample_outputs/baseline_draft.md \
  --edited sample_outputs/operator_edited_draft.md \
  --preferences data/preferences.json
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Optional OCR

Install system Tesseract to OCR scanned PDFs and images:

```bash
sudo apt-get install tesseract-ocr
```

If Tesseract is absent, image/PDF pages with no embedded text are accepted, flagged, and excluded from evidence instead of crashing or hallucinating.

## Optional LLM

The generator includes a standard-library OpenAI Responses API client. Set:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

Then run `draft` without `--no-llm`. The prompt forces use of retrieved evidence only, requires chunk citations, and tells the model to state gaps instead of inventing unsupported claims. If the API call fails, the system falls back to deterministic grounded generation.

## Optional Semantic Retrieval

To switch from sparse TF-IDF to semantic embeddings, install `sentence-transformers` and
rebuild the index with `--embedding-model sentence-transformers`. The index metadata
stores the chosen model so `draft` and the REST API use the same backend.

## Optional REST API

Build an index first, then run:

```bash
PYTHONPATH=src python -m legal_ai_workflow.api --port 8000
```

Endpoints:

- `GET /health`
- `POST /draft` with `{"query": "...", "top_k": 6, "min_score": 0.0, "min_confidence": 0.55, "use_llm": false}`
- `POST /feedback` with `{"original": "path/to/original.md", "edited": "path/to/edited.md"}`

## Project Layout

```text
src/legal_ai_workflow/
  ingestion.py      Document extraction, OCR hooks, structured JSON
  retrieval.py      Chunking, local TF-IDF embeddings, JSON vector store
  generation.py     Grounded memo generation with citations
  feedback.py       Preference learning from operator edits
  llm.py            Optional OpenAI Responses API adapter
  cli.py            End-to-end command line workflow
  api.py            Small stdlib REST API
sample_inputs/      Synthetic messy legal documents
sample_outputs/     Baseline, operator edit, improved draft, evidence maps
tests/              Unit/integration tests
```

## Sample Inputs

The demo corpus currently includes:

- `handwritten_meeting_notes.txt`
- `lease_amendment_scan_transcript.txt`
- `low_res_notice_attachment.png`
- `notice_email_chain.txt`
- `notice_scan_transcript.txt`
- `synthetic_title_addendum.pdf`
- `title_commitment_messy.txt`
- `title_objection_notes.txt`

