from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legal_ai_workflow.feedback import FeedbackLearner, PreferenceStore
from legal_ai_workflow.generation import DraftGenerator
from legal_ai_workflow.ingestion import DocumentProcessor
from legal_ai_workflow.models import Chunk
from legal_ai_workflow.retrieval import Chunker, VectorStore


class WorkflowTests(unittest.TestCase):
    def test_text_ingestion_extracts_structured_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notice.txt"
            path.write_text(
                "NOTICE OF DEFAULT\nDate: May 1, 2026\nSeller: West 46th Street Holdings, Inc.\n"
                "Buyer: Ridge Harbor Capital LLC\nMatter No: PSL-2026-0142\n[illegible] stamp\n",
                encoding="utf-8",
            )
            document = DocumentProcessor().process_path(path)
        self.assertIn("May 1, 2026", document.structured_data["dates"])
        self.assertIn("PSL-2026-0142", document.structured_data["referenced_case_numbers"])
        self.assertTrue(document.structured_data["low_confidence_sections"])

    def test_retrieval_and_generation_are_grounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "title.txt"
            path.write_text(
                "Matter No: PSL-2026-0142\nBuyer: Ridge Harbor Capital LLC\n"
                "Utility easement recorded in Liber 4102 affects the rear five feet.\n"
                "Notice is effective on confirmed delivery.\n",
                encoding="utf-8",
            )
            document = DocumentProcessor().process_path(path)
            chunks = Chunker(target_tokens=40).chunk_document(document)
            store = VectorStore()
            store.build(chunks)
            evidence = store.query("utility easement and notice delivery", top_k=3)
            draft = DraftGenerator(use_llm=False).generate("Prepare memo", evidence, PreferenceStore())
        self.assertTrue(evidence)
        self.assertIn("-C", draft)
        self.assertIn("Evidence Map", draft)

    def test_feedback_preferences_change_future_draft_shape(self) -> None:
        original = "# Memo\n\nThe system wrote a long paragraph without a citation and with broad conclusions.\n"
        edited = (
            "# Memo\n\n## Key Supported Facts\n\n- Keep facts short [ABC-C001].\n\n"
            "## Risk Flags\n\n- Check unclear scans before relying on the draft.\n"
        )
        store = PreferenceStore()
        learned = FeedbackLearner().learn_from_text(original, edited, store)
        self.assertIn("include_risk_section", learned)
        self.assertTrue(store.is_enabled("prefer_bullets"))
        self.assertTrue(store.is_enabled("strict_citations"))

    def test_empty_document_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")
            document = DocumentProcessor().process_path(path)
        self.assertFalse(document.raw_text)
        self.assertTrue(any(warning.severity == "error" for warning in document.warnings))

    def test_min_confidence_filter_excludes_low_confidence_chunks(self) -> None:
        chunks = [
            Chunk(
                chunk_id="DOC-C001",
                document_id="DOC",
                source_path="sample.txt",
                text="Notice is effective upon delivery.",
                metadata={"confidence": 0.4},
            ),
            Chunk(
                chunk_id="DOC-C002",
                document_id="DOC",
                source_path="sample.txt",
                text="Utility easement recorded in Liber 4102.",
                metadata={"confidence": 0.9},
            ),
        ]
        store = VectorStore()
        store.build(chunks)
        results = store.query("easement", top_k=5, min_confidence=0.8)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "DOC-C002")

    def test_empty_query_returns_no_evidence(self) -> None:
        chunks = [
            Chunk(
                chunk_id="DOC-C001",
                document_id="DOC",
                source_path="sample.txt",
                text="Notice is effective upon delivery.",
                metadata={"confidence": 0.9},
            )
        ]
        store = VectorStore()
        store.build(chunks)
        results = store.query("   ")
        self.assertEqual(results, [])

    def test_no_evidence_returns_insufficient_message(self) -> None:
        draft = DraftGenerator(use_llm=False).generate("Prepare memo", [], PreferenceStore())
        self.assertIn("No relevant evidence was retrieved", draft)

    def test_preference_weights_accumulate(self) -> None:
        original = "# Memo\n\nThis paragraph has no citations and is long enough to trigger strict citations logic."
        edited = "# Memo\n\n- Short fact [ABC-C001].\n\n## Risk Flags\n- Review scans."
        store = PreferenceStore()
        learner = FeedbackLearner()
        learner.learn_from_text(original, edited, store)
        learner.learn_from_text(original, edited, store)
        self.assertGreaterEqual(store.preferences["prefer_bullets"].weight, 2)


if __name__ == "__main__":
    unittest.main()
