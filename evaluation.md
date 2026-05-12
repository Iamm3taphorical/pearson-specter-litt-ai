# Evaluation

## Commands Run

```bash
python3 -m compileall src tests
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m legal_ai_workflow.cli run-demo --reset
```

Result: all tests passed.

```text
Ran 8 tests in 0.002s
OK
```

## Demo Corpus

The demo processed eight sample inputs:

- `title_commitment_messy.txt`
- `notice_scan_transcript.txt`
- `handwritten_meeting_notes.txt`
- `lease_amendment_scan_transcript.txt`
- `notice_email_chain.txt`
- `title_objection_notes.txt`
- `synthetic_title_addendum.pdf`
- `low_res_notice_attachment.png`

The PNG was accepted and flagged because Tesseract is not installed in this environment. That verifies graceful degradation for image-only documents.

## Processing Results

- Documents processed: 8
- Indexed chunks: 12 with the demo target token size (60).
- Embedding model: `local-tfidf`
- Structured extraction found dates, parties, matter/file numbers, key clauses, amounts, and low-confidence sections in the text/PDF samples.
- Low-confidence examples were preserved in processed JSON instead of silently removed.

## Grounding Results

The generated drafts cite every fact-bearing item using chunk IDs such as `[319c040caa6d-C001; p. 1]`.

Evidence trace files:

- `sample_outputs/baseline_evidence.json`
- `sample_outputs/improved_evidence.json`
- `sample_outputs/improved_evidence_round2.json`

Each evidence record includes source file, chunk ID, score, page, extraction confidence, extraction method, and passage text.

## Feedback Learning Results

The demo creates a simulated operator edit and learns these reusable preferences in `data/preferences.json`:

- `include_risk_section`
- `prefer_bullets`
- `strict_citations`
- `respect_operator_section_order`

Visible improvement:

- Baseline draft uses numbered facts and no dedicated `Risk Flags` section.
- Improved draft uses bullet formatting and adds a source-cited `Risk Flags` section.
- Future LLM prompts also receive the learned rules and recent edit examples.
- The demo runs a second simulated edit so preference weights increase and the improved draft evolves again.

## Draft Metrics

The demo prints lightweight metrics for the baseline, improved, and round-two drafts:

- Total character length
- Citation count
- Section count
- Bullet count

## Quality Checks

- Unit coverage verifies text ingestion, structured fields, retrieval-grounded drafting, and preference learning.
- Unsupported generation is minimized by returning an insufficient-evidence message when no chunks are retrieved.
- OCR-dependent failures are captured as warnings in processed JSON.

## Residual Risk

- Retrieval quality should be re-evaluated with a larger legal corpus and semantic embeddings.
- A human lawyer must review all draft outputs before use.
- Production deployment should add authentication, upload handling, per-user preference isolation, and stronger audit logging.
