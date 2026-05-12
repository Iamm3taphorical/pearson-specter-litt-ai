"""Feedback loop that learns reusable drafting preferences from edits."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_text, split_sentences, utc_now_iso


@dataclass
class LearnedPreference:
    preference_id: str
    description: str
    weight: int = 1
    examples: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def bump(self, description: str, example: str | None = None) -> None:
        self.description = description
        self.weight += 1
        self.updated_at = utc_now_iso()
        if example and example not in self.examples:
            self.examples.append(example)
            self.examples = self.examples[-5:]


class PreferenceStore:
    """JSON-backed rules and few-shot edit pairs."""

    def __init__(
        self,
        preferences: dict[str, LearnedPreference] | None = None,
        section_order: list[str] | None = None,
        examples: list[dict[str, str]] | None = None,
    ) -> None:
        self.preferences = preferences or {}
        self.section_order = section_order or []
        self.examples = examples or []

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceStore":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        preferences = {
            key: LearnedPreference(**value)
            for key, value in data.get("preferences", {}).items()
        }
        return cls(preferences, data.get("section_order", []), data.get("examples", []))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        data = {
            "version": 1,
            "preferences": {
                key: {
                    "preference_id": pref.preference_id,
                    "description": pref.description,
                    "weight": pref.weight,
                    "examples": pref.examples,
                    "updated_at": pref.updated_at,
                }
                for key, pref in sorted(self.preferences.items())
            },
            "section_order": self.section_order,
            "examples": self.examples[-10:],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_or_bump(self, preference_id: str, description: str, example: str | None = None) -> None:
        if preference_id in self.preferences:
            self.preferences[preference_id].bump(description, example)
        else:
            self.preferences[preference_id] = LearnedPreference(preference_id, description, examples=[example] if example else [])

    def is_enabled(self, preference_id: str, threshold: int = 1) -> bool:
        pref = self.preferences.get(preference_id)
        return bool(pref and pref.weight >= threshold)

    def to_prompt_rules(self) -> list[dict[str, Any]]:
        rules = [
            {"id": pref.preference_id, "description": pref.description, "weight": pref.weight}
            for pref in sorted(self.preferences.values(), key=lambda item: (-item.weight, item.preference_id))
        ]
        if self.section_order:
            rules.append({"id": "preferred_section_order", "description": "Use this section order when possible.", "order": self.section_order})
        if self.examples:
            rules.append({"id": "few_shot_edit_examples", "description": "Recent accepted edit pairs.", "examples": self.examples[-3:]})
        return rules


class FeedbackLearner:
    """Extracts reusable preferences from original and operator-edited drafts."""

    HEADING_RE = re.compile(r"^(?:#{1,6}\s*)?([A-Z][A-Za-z0-9 /&()-]{2,80})\s*$", re.MULTILINE)

    def learn_from_files(
        self,
        original_path: str | Path,
        edited_path: str | Path,
        preference_path: str | Path,
        feedback_log_path: str | Path | None = None,
    ) -> PreferenceStore:
        original = read_text(Path(original_path))
        edited = read_text(Path(edited_path))
        store = PreferenceStore.load(preference_path)
        self.learn_from_text(original, edited, store)
        store.examples.append(
            {
                "original_excerpt": original[:900],
                "edited_excerpt": edited[:900],
                "captured_at": utc_now_iso(),
            }
        )
        store.save(preference_path)
        if feedback_log_path:
            self._append_feedback_log(feedback_log_path, original_path, edited_path, original, edited, store)
        return store

    def learn_from_text(self, original: str, edited: str, store: PreferenceStore) -> list[str]:
        learned: list[str] = []
        original_lower = original.lower()
        edited_lower = edited.lower()

        if self._has_heading(edited, "risk") and not self._has_heading(original, "risk"):
            store.add_or_bump(
                "include_risk_section",
                "Include a distinct Risk Flags section when source evidence supports risks, uncertainty, deadlines, defaults, or exceptions.",
                "Operator added a risk section.",
            )
            learned.append("include_risk_section")

        if self._bullet_count(edited) >= self._bullet_count(original) + 2:
            store.add_or_bump(
                "prefer_bullets",
                "Prefer concise bullet points over dense paragraphs for reviewable legal drafts.",
                "Operator converted prose into bullets.",
            )
            learned.append("prefer_bullets")

        if len(edited) < len(original) * 0.85:
            store.add_or_bump(
                "concise_default",
                "Keep future drafts concise unless the query asks for detailed narrative.",
                "Operator shortened the draft.",
            )
            learned.append("concise_default")

        removed_uncited = self._removed_uncited_sentences(original, edited)
        if removed_uncited:
            store.add_or_bump(
                "strict_citations",
                "Avoid unsupported claims; every factual statement should carry a citation or be moved to gaps.",
                f"Operator removed uncited sentence: {removed_uncited[0][:160]}",
            )
            learned.append("strict_citations")

        edited_headings = self._headings(edited)
        original_headings = self._headings(original)
        if edited_headings and edited_headings != original_headings:
            store.section_order = edited_headings[:10]
            store.add_or_bump(
                "respect_operator_section_order",
                "Reuse the operator's accepted section order when generating the same draft type.",
                "Operator reordered headings.",
            )
            learned.append("respect_operator_section_order")

        return learned

    def _bullet_count(self, text: str) -> int:
        return sum(1 for line in text.splitlines() if line.lstrip().startswith(("-", "*", "•")))

    def _headings(self, text: str) -> list[str]:
        headings = []
        for match in self.HEADING_RE.finditer(text):
            heading = match.group(1).strip()
            if 3 <= len(heading) <= 70 and not heading.endswith("."):
                headings.append(heading)
        return headings[:12]

    def _has_heading(self, text: str, term: str) -> bool:
        term = term.lower()
        return any(term in heading.lower() for heading in self._headings(text))

    def _removed_uncited_sentences(self, original: str, edited: str) -> list[str]:
        original_sentences = split_sentences(original)
        edited_sentences = set(split_sentences(edited))
        removed = []
        for sentence in original_sentences:
            # Skip headings/metadata/table rows that are not factual claims.
            if self._is_non_factual_sentence(sentence):
                continue
            if "[" in sentence and "]" in sentence:
                continue
            if len(sentence) < 50:
                continue
            closest = difflib.get_close_matches(sentence, edited_sentences, n=1, cutoff=0.78)
            if not closest:
                removed.append(sentence)
        return removed

    def _is_non_factual_sentence(self, sentence: str) -> bool:
        stripped = sentence.strip()
        if not stripped:
            return True
        if stripped.startswith("#"):
            return True
        if stripped.startswith("|"):
            return True
        lower = stripped.lower()
        if lower.startswith("drafting task:"):
            return True
        if lower.startswith("generated "):
            return True
        if lower in {
            "matter snapshot",
            "key supported facts",
            "document-driven issues",
            "risk flags",
            "gaps / unclear items",
            "evidence map",
        }:
            return True
        return False

    def _append_feedback_log(
        self,
        feedback_log_path: str | Path,
        original_path: str | Path,
        edited_path: str | Path,
        original: str,
        edited: str,
        store: PreferenceStore,
    ) -> None:
        path = Path(feedback_log_path)
        ensure_dir(path.parent)
        event = {
            "captured_at": utc_now_iso(),
            "original_path": str(original_path),
            "edited_path": str(edited_path),
            "original_length": len(original),
            "edited_length": len(edited),
            "preferences_after_event": [rule["id"] for rule in store.to_prompt_rules()],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
