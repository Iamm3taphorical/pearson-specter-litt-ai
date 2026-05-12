"""Tiny stdlib REST API for local demos.

Run after building an index:
    python -m legal_ai_workflow.api --port 8000
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .feedback import FeedbackLearner, PreferenceStore
from .generation import DraftGenerator
from .retrieval import VectorStore


class WorkflowHandler(BaseHTTPRequestHandler):
    index_dir = Path("data/index")
    preferences_path = Path("data/preferences.json")

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/draft":
                self._draft(payload)
            elif parsed.path == "/feedback":
                self._feedback(payload)
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _draft(self, payload: dict[str, object]) -> None:
        query = str(payload.get("query") or "Prepare a first-pass internal memo.")
        top_k = int(payload.get("top_k") or 6)
        store = VectorStore.load(self.index_dir)
        evidence = store.query(query, top_k=top_k)
        preferences = PreferenceStore.load(self.preferences_path)
        draft = DraftGenerator(use_llm=bool(payload.get("use_llm", False))).generate(query, evidence, preferences)
        self._json({"draft": draft, "evidence": [item.to_dict() for item in evidence]})

    def _feedback(self, payload: dict[str, object]) -> None:
        original = str(payload["original"])
        edited = str(payload["edited"])
        store = FeedbackLearner().learn_from_files(original, edited, self.preferences_path, "data/feedback_events.jsonl")
        self._json({"preferences": store.to_prompt_rules()})

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def _json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--preferences", default="data/preferences.json")
    args = parser.parse_args()
    WorkflowHandler.index_dir = Path(args.index_dir)
    WorkflowHandler.preferences_path = Path(args.preferences)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), WorkflowHandler)
    print(f"serving on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
