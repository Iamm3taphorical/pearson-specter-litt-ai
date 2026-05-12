"""Command line interface for the Pearson Specter Litt workflow."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from .feedback import FeedbackLearner, PreferenceStore
from .generation import DraftGenerator, write_evidence_json
from .ingestion import DocumentProcessor
from .retrieval import VectorStore, build_index_from_processed
from .utils import ensure_dir, write_text


DEFAULT_QUERY = (
    "Prepare a first-pass internal memo summarizing title, notice, party, date, "
    "and risk facts supported by the uploaded documents."
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded legal document workflow")
    sub = parser.add_subparsers()

    process = sub.add_parser("process", help="Extract text and structured JSON from files")
    process.add_argument("input", help="Input file or directory")
    process.add_argument("--output-dir", default="data/processed")
    process.set_defaults(func=cmd_process)

    index = sub.add_parser("index", help="Build retrieval index from processed JSON")
    index.add_argument("--processed-dir", default="data/processed")
    index.add_argument("--index-dir", default="data/index")
    index.add_argument("--target-tokens", type=int, default=120)
    index.add_argument("--overlap-tokens", type=int, default=25)
    index.set_defaults(func=cmd_index)

    draft = sub.add_parser("draft", help="Generate a grounded first-pass memo")
    draft.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    draft.add_argument("--index-dir", default="data/index")
    draft.add_argument("--preferences", default="data/preferences.json")
    draft.add_argument("--output", default="sample_outputs/draft.md")
    draft.add_argument("--evidence-output", default="sample_outputs/evidence.json")
    draft.add_argument("--top-k", type=int, default=6)
    draft.add_argument("--min-score", type=float, default=0.0)
    draft.add_argument("--min-confidence", type=float, default=None)
    draft.add_argument("--no-llm", action="store_true", help="Force deterministic local generation")
    draft.set_defaults(func=cmd_draft)

    feedback = sub.add_parser("feedback", help="Learn preferences from an operator edit")
    feedback.add_argument("--original", required=True)
    feedback.add_argument("--edited", required=True)
    feedback.add_argument("--preferences", default="data/preferences.json")
    feedback.add_argument("--log", default="data/feedback_events.jsonl")
    feedback.set_defaults(func=cmd_feedback)

    demo = sub.add_parser("run-demo", help="Run the full sample workflow end to end")
    demo.add_argument("--sample-inputs", default="sample_inputs")
    demo.add_argument("--processed-dir", default="data/processed")
    demo.add_argument("--index-dir", default="data/index")
    demo.add_argument("--preferences", default="data/preferences.json")
    demo.add_argument("--sample-outputs", default="sample_outputs")
    demo.add_argument("--reset", action="store_true", help="Clear generated data before running")
    demo.set_defaults(func=cmd_run_demo)

    return parser


def cmd_process(args: argparse.Namespace) -> int:
    processor = DocumentProcessor()
    input_path = Path(args.input)
    output_dir = ensure_dir(Path(args.output_dir))
    documents = processor.process_directory(input_path) if input_path.is_dir() else [processor.process_path(input_path)]
    for document in documents:
        output_path = processor.write_processed(document, output_dir)
        print(f"processed {document.file_name} -> {output_path}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    store = build_index_from_processed(
        args.processed_dir,
        args.index_dir,
        target_tokens=int(args.target_tokens),
        overlap_tokens=int(args.overlap_tokens),
    )
    print(f"indexed {len(store.chunks)} chunks -> {Path(args.index_dir) / 'index.json'}")
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    store = VectorStore.load(args.index_dir)
    evidence = store.query(
        args.query,
        top_k=args.top_k,
        min_score=float(args.min_score),
        min_confidence=args.min_confidence,
    )
    preferences = PreferenceStore.load(args.preferences)
    generator = DraftGenerator(use_llm=not args.no_llm)
    draft = generator.generate(args.query, evidence, preferences)
    generator.write_draft(args.output, draft)
    write_evidence_json(args.evidence_output, evidence)
    print(f"draft -> {args.output}")
    print(f"evidence -> {args.evidence_output}")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    learner = FeedbackLearner()
    store = learner.learn_from_files(args.original, args.edited, args.preferences, args.log)
    print(json.dumps(store.to_prompt_rules(), indent=2))
    return 0


def cmd_run_demo(args: argparse.Namespace) -> int:
    processed_dir = Path(args.processed_dir)
    index_dir = Path(args.index_dir)
    output_dir = ensure_dir(Path(args.sample_outputs))
    preferences_path = Path(args.preferences)
    if args.reset:
        for path in (processed_dir, index_dir):
            if path.exists():
                shutil.rmtree(path)
        if preferences_path.exists():
            preferences_path.unlink()

    cmd_process(argparse.Namespace(input=args.sample_inputs, output_dir=str(processed_dir)))
    cmd_index(
        argparse.Namespace(
            processed_dir=str(processed_dir),
            index_dir=str(index_dir),
            target_tokens=60,
            overlap_tokens=20,
        )
    )

    baseline_path = output_dir / "baseline_draft.md"
    baseline_evidence_path = output_dir / "baseline_evidence.json"
    cmd_draft(
        argparse.Namespace(
            query=DEFAULT_QUERY,
            index_dir=str(index_dir),
            preferences=str(preferences_path),
            output=str(baseline_path),
            evidence_output=str(baseline_evidence_path),
            top_k=6,
            min_score=0.0,
            min_confidence=None,
            no_llm=True,
        )
    )

    edited_path = output_dir / "operator_edited_draft.md"
    write_text(edited_path, _simulated_operator_edit(baseline_path.read_text(encoding="utf-8")))
    cmd_feedback(
        argparse.Namespace(
            original=str(baseline_path),
            edited=str(edited_path),
            preferences=str(preferences_path),
            log="data/feedback_events.jsonl",
        )
    )

    improved_path = output_dir / "improved_draft_after_feedback.md"
    improved_evidence_path = output_dir / "improved_evidence.json"
    cmd_draft(
        argparse.Namespace(
            query=DEFAULT_QUERY,
            index_dir=str(index_dir),
            preferences=str(preferences_path),
            output=str(improved_path),
            evidence_output=str(improved_evidence_path),
            top_k=6,
            min_score=0.0,
            min_confidence=None,
            no_llm=True,
        )
    )

    round_two_edit_path = output_dir / "operator_edited_draft_round2.md"
    write_text(round_two_edit_path, _simulated_operator_edit(improved_path.read_text(encoding="utf-8")))
    cmd_feedback(
        argparse.Namespace(
            original=str(improved_path),
            edited=str(round_two_edit_path),
            preferences=str(preferences_path),
            log="data/feedback_events.jsonl",
        )
    )

    improved_round_two_path = output_dir / "improved_draft_after_feedback_round2.md"
    improved_round_two_evidence_path = output_dir / "improved_evidence_round2.json"
    cmd_draft(
        argparse.Namespace(
            query=DEFAULT_QUERY,
            index_dir=str(index_dir),
            preferences=str(preferences_path),
            output=str(improved_round_two_path),
            evidence_output=str(improved_round_two_evidence_path),
            top_k=6,
            min_score=0.0,
            min_confidence=None,
            no_llm=True,
        )
    )

    _print_metrics("baseline", baseline_path.read_text(encoding="utf-8"))
    _print_metrics("improved", improved_path.read_text(encoding="utf-8"))
    _print_metrics("improved_round2", improved_round_two_path.read_text(encoding="utf-8"))

    print("demo complete")
    return 0


def _simulated_operator_edit(baseline: str) -> str:
    """Creates a deterministic reviewer edit for demo/evaluation purposes."""
    lines = [
        "# First-Pass Internal Memo",
        "",
        "## Matter Snapshot",
        "",
        "- Use bullet points by default so reviewers can scan source-backed facts quickly.",
        "- Keep unsupported legal conclusions out of the draft unless they are tied to a retrieved citation.",
        "",
        "## Key Supported Facts",
        "",
    ]
    fact_lines = _extract_section_lines(baseline, "Key Supported Facts")
    source_lines = fact_lines or baseline.splitlines()
    for line in source_lines:
        if reuses_supported_fact(line):
            normalized = line.strip()
            normalized = re.sub(r"^\d+\.\s+", "", normalized)
            normalized = re.sub(r"^[-*•]\s+", "", normalized)
            lines.append(f"- {normalized}")
    lines.extend(
        [
            "",
            "## Risk Flags",
            "",
            "- Check unclear or low-confidence source sections against the original images/PDFs before sending a final memo.",
            "- Treat defaults, termination language, title exceptions, and deadline language as review priorities.",
            "",
            "## Gaps / Unclear Items",
            "",
            "- Leave missing legal conclusions as gaps instead of inferring them.",
        ]
    )
    return "\n".join(lines) + "\n"


def reuses_supported_fact(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return False
    if stripped.startswith("#"):
        return False
    if stripped.lower().startswith("**drafting task:**"):
        return False
    return "-C" in stripped and ("[" in stripped and "]" in stripped)


def _extract_section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    heading_lower = heading.lower()
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith("## ") and heading_lower in line_stripped.lower():
            capture = True
            continue
        if capture and line_stripped.startswith("## "):
            break
        if capture:
            collected.append(line)
    return collected


def _draft_metrics(draft: str) -> dict[str, int]:
    citations = len(re.findall(r"\[[A-Za-z0-9_-]+-C\d{3}(?:;[^\]]+)?\]", draft))
    section_count = sum(1 for line in draft.splitlines() if line.strip().startswith("## "))
    bullet_count = sum(1 for line in draft.splitlines() if line.lstrip().startswith("- "))
    return {
        "chars": len(draft),
        "citations": citations,
        "sections": section_count,
        "bullets": bullet_count,
    }


def _print_metrics(label: str, draft: str) -> None:
    metrics = _draft_metrics(draft)
    print(
        f"metrics[{label}]: chars={metrics['chars']}, citations={metrics['citations']}, "
        f"sections={metrics['sections']}, bullets={metrics['bullets']}"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
