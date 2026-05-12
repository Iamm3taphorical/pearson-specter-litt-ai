"""Grounded draft generation with citations and fallback behavior."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .feedback import PreferenceStore
from .llm import OpenAIResponsesClient
from .models import RetrievalResult
from .utils import ensure_dir, short_quote, split_sentences, utc_now_iso, write_text


SYSTEM_PROMPT = """You are drafting an internal legal work product for Pearson Specter Litt.
Use only the evidence chunks provided by the user. Do not use outside facts or legal assumptions.
Every factual claim must include a source chunk citation like [DOC-C001].
If evidence is missing, write "Not found in provided evidence" or list it under Gaps.
Avoid legal advice; produce a first-pass review draft for a human operator."""


class DraftGenerator:
    """Generates first-pass internal memos from retrieved evidence."""

    def __init__(self, llm_client: OpenAIResponsesClient | None = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client or OpenAIResponsesClient()
        self.use_llm = use_llm

    def generate(
        self,
        query: str,
        evidence: list[RetrievalResult],
        preferences: PreferenceStore | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        preferences = preferences or PreferenceStore()
        if not evidence:
            return self._insufficient_evidence(query)

        if self.use_llm and self.llm_client.available:
            prompt = self._build_user_prompt(query, evidence, preferences)
            try:
                response = self.llm_client.generate(SYSTEM_PROMPT, prompt)
                if self._has_citations(response.text):
                    return response.text.strip() + f"\n\n_Generated with {response.provider}:{response.model} at {utc_now_iso()}._\n"
            except Exception as exc:
                fallback_note = f"\n\n> LLM generation failed, using local grounded fallback. Error: {exc}\n"
                return self._generate_local(query, evidence, preferences, metadata) + fallback_note
        return self._generate_local(query, evidence, preferences, metadata)

    def write_draft(self, output_path: str | Path, draft: str) -> None:
        write_text(Path(output_path), draft)

    def _build_user_prompt(self, query: str, evidence: list[RetrievalResult], preferences: PreferenceStore) -> str:
        evidence_block = "\n\n".join(
            f"{result.citation()} score={result.score:.3f}\n{result.chunk.text}" for result in evidence
        )
        preference_block = json.dumps(preferences.to_prompt_rules(), indent=2)
        return f"""Draft type: First-pass internal memo
Task/query: {query}

Operator preferences to apply when consistent with the evidence:
{preference_block}

Evidence:
{evidence_block}

Required structure:
- Matter snapshot
- Key supported facts
- Document-driven issues
- Gaps / unclear items
- Evidence map

Remember: cite every factual claim using the chunk id provided above."""

    def _generate_local(
        self,
        query: str,
        evidence: list[RetrievalResult],
        preferences: PreferenceStore,
        metadata: dict[str, Any] | None,
    ) -> str:
        prefer_bullets = preferences.is_enabled("prefer_bullets")
        include_risk = preferences.is_enabled("include_risk_section")
        strict_citations = preferences.is_enabled("strict_citations")
        concise = preferences.is_enabled("concise_default")

        lines: list[str] = [
            "# First-Pass Internal Memo",
            "",
            f"**Drafting task:** {query}",
            "",
            "## Matter Snapshot",
            "",
        ]

        top = evidence[0]
        snapshot = (
            f"The strongest available source is `{top.chunk.metadata.get('file_name', 'unknown')}` "
            f"with retrieval score {top.score:.3f} {top.citation()}."
        )
        if prefer_bullets:
            lines.append(f"- {snapshot}")
            lines.append("- This draft uses only retrieved source passages and flags gaps instead of filling them.")
        else:
            lines.append(snapshot)
            lines.append("")
            lines.append("This draft uses only retrieved source passages and flags gaps instead of filling them.")
        lines.extend(["", "## Key Supported Facts", ""])

        fact_groups = self._supported_fact_items(evidence, max_items=5 if concise else 7)
        for topic, items in fact_groups.items():
            lines.extend([f"### {topic}", ""])
            if prefer_bullets:
                lines.extend(f"- {item}" for item in items)
            else:
                lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))
            lines.append("")

        lines.extend(["", "## Document-Driven Issues", ""])
        issues = self._issue_items(evidence, strict_citations=strict_citations)
        lines.extend(f"- {item}" for item in issues)

        if include_risk:
            lines.extend(["", "## Risk Flags", ""])
            lines.extend(f"- {item}" for item in self._risk_items(evidence))

        lines.extend(["", "## Gaps / Unclear Items", ""])
        gaps = self._gap_items(evidence, metadata or {})
        lines.extend(f"- {item}" for item in gaps)

        lines.extend(["", "## Evidence Map", ""])
        lines.extend(["| Citation | Source | Score | Passage |", "|---|---:|---:|---|"])
        for result in evidence:
            source = result.chunk.metadata.get("file_name", Path(result.chunk.source_path).name)
            lines.append(f"| {result.citation()} | {source} | {result.score:.3f} | {short_quote(result.chunk.text, 180)} |")

        lines.extend(["", f"_Generated locally at {utc_now_iso()} using retrieval-only evidence._", ""])
        return "\n".join(lines)

    def _supported_fact_items(self, evidence: list[RetrievalResult], max_items: int) -> dict[str, list[str]]:
        items_by_topic: dict[str, list[str]] = {}
        seen: set[str] = set()
        total = 0
        for result in evidence:
            for sentence in split_sentences(result.chunk.text):
                if len(sentence) < 35:
                    continue
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                fingerprint = cleaned[:80].lower()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                topic = self._topic_for_sentence(cleaned)
                items_by_topic.setdefault(topic, []).append(f"{short_quote(cleaned, 240)} {result.citation()}")
                total += 1
                if total >= max_items:
                    return items_by_topic
        if not items_by_topic:
            items_by_topic["Other Facts"] = [
                f"The retrieved material is too short to extract stable fact statements {evidence[0].citation()}."
            ]
        return items_by_topic

    def _issue_items(self, evidence: list[RetrievalResult], strict_citations: bool) -> list[str]:
        issue_terms = ("default", "notice", "exception", "easement", "termination", "deadline", "closing", "unclear", "illegible", "payment")
        items: list[str] = []
        seen_terms: set[str] = set()
        for result in evidence:
            lower = result.chunk.text.lower()
            matched = self._first_matching_term(lower, issue_terms)
            if matched in seen_terms:
                continue
            if matched != "retrieved issue":
                seen_terms.add(matched)
                items.append(f"Review source language concerning {matched} before relying on the draft {result.citation()}.")
        if strict_citations:
            items.append("Do not add uncited factual or legal conclusions during review; unsupported assertions should remain in gaps.")
        return items[:5] or [f"No specific issue language was retrieved; human review should confirm whether the query needs broader evidence {evidence[0].citation()}."]

    def _topic_for_sentence(self, sentence: str) -> str:
        lower = sentence.lower()
        topic_keywords = [
            ("Notice / Default", ("notice", "default", "cure", "delivery")),
            ("Title / Easement", ("title", "easement", "exception", "schedule b", "lien", "liber")),
            ("Parties", ("buyer", "seller", "borrower", "lender", "landlord", "tenant", "plaintiff", "defendant", "grantor", "grantee", "client", "counterparty", "party")),
            ("Dates / Deadlines", ("date", "deadline", "closing", "effective", "recorded", "may", "june", "july", "aug", "sept", "oct", "nov", "dec")),
            ("Payments", ("payment", "amount", "fee", "rent", "$")),
        ]
        for topic, keywords in topic_keywords:
            if any(keyword in lower for keyword in keywords):
                return topic
        return "Other Facts"

    def _risk_items(self, evidence: list[RetrievalResult]) -> list[str]:
        risks: list[str] = []
        for result in evidence:
            lower = result.chunk.text.lower()
            if any(marker in lower for marker in ("illegible", "unclear", "[?]", "???", "smudged")):
                risks.append(f"Some source text is explicitly unclear or illegible and should be checked against the original document {result.citation()}.")
            if any(marker in lower for marker in ("default", "termination", "exception", "easement", "deadline")):
                risks.append(f"The retrieved evidence includes risk-bearing language that may affect obligations or title review {result.citation()}.")
        return risks[:5] or [f"No separate risk flags were found in retrieved passages; confirm against the full document set {evidence[0].citation()}."]

    def _gap_items(self, evidence: list[RetrievalResult], metadata: dict[str, Any]) -> list[str]:
        gaps = ["Not found in provided evidence: legal conclusion, court outcome, or external authority beyond the uploaded documents."]
        low_confidence = [result for result in evidence if float(result.chunk.metadata.get("confidence", 1.0)) < 0.55]
        if low_confidence:
            citations = ", ".join(result.citation() for result in low_confidence[:4])
            gaps.append(f"Low-confidence extraction appears in retrieved material: {citations}.")
        if metadata.get("warnings"):
            gaps.append("Processing warnings exist for one or more documents; inspect processed JSON before final use.")
        return gaps

    def _first_matching_term(self, text: str, terms: tuple[str, ...]) -> str:
        return next((term for term in terms if term in text), "retrieved issue")

    def _has_citations(self, text: str) -> bool:
        return bool(re.search(r"\[[A-Za-z0-9_-]+-C\d{3}(?:;[^\]]+)?\]", text))

    def _insufficient_evidence(self, query: str) -> str:
        return (
            "# First-Pass Internal Memo\n\n"
            f"**Drafting task:** {query}\n\n"
            "No relevant evidence was retrieved. The system cannot produce a grounded draft without source support.\n"
        )


def write_evidence_json(output_path: str | Path, evidence: list[RetrievalResult]) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps([item.to_dict() for item in evidence], indent=2), encoding="utf-8")
