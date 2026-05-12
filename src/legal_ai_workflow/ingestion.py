"""Document extraction and lightweight legal structure parsing."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import ExtractionWarning, PageText, ProcessedDocument
from .utils import ensure_dir, normalize_text, short_quote, stable_id

try:  # Optional dependency.
    import fitz  # type: ignore
except Exception:  # pragma: no cover - exercised when PyMuPDF is absent.
    fitz = None


TEXT_SUFFIXES = {".txt", ".md", ".csv", ".rtf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is requested but no OCR backend is available."""


class TesseractOCR:
    """Small subprocess wrapper around Tesseract OCR.

    The project does not require pytesseract. If the `tesseract` binary is
    installed on the reviewer machine, OCR is used automatically.
    """

    def __init__(self, language: str = "eng") -> None:
        self.language = language
        self.binary = shutil.which("tesseract")

    @property
    def available(self) -> bool:
        return bool(self.binary)

    def extract(self, image_path: Path) -> tuple[str, float]:
        if not self.binary:
            raise OCRUnavailableError("Tesseract is not installed or not on PATH.")

        command = [self.binary, str(image_path), "stdout", "-l", self.language, "--psm", "6", "tsv"]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "Tesseract OCR failed.")

        text_parts: list[str] = []
        confidences: list[float] = []
        for line in proc.stdout.splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 12:
                continue
            word = cols[11].strip()
            if word:
                text_parts.append(word)
            try:
                confidence = float(cols[10])
            except ValueError:
                continue
            if confidence >= 0:
                confidences.append(confidence / 100.0)
        text = normalize_text(" ".join(text_parts))
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.4
        return text, max(0.0, min(1.0, average_confidence))


class StructuredDataExtractor:
    """Extracts reviewable structured fields from messy legal text.

    This is intentionally conservative: missing values remain empty/null, and
    uncertain snippets are surfaced as low-confidence sections instead of being
    invented.
    """

    DATE_RE = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    )
    CASE_RE = re.compile(
        r"\b(?:Case|Docket|Matter|File|Reference|Ref)\b\s*(?:No\.?\s*:|No\.?|#|:)?\s*([A-Z0-9][A-Z0-9\-:/]{3,})",
        re.IGNORECASE,
    )
    MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
    CLAUSE_KEYWORDS = (
        "notice",
        "termination",
        "default",
        "closing",
        "title",
        "exceptions",
        "payment",
        "indemnity",
        "remedies",
        "confidentiality",
        "governing law",
        "assignment",
        "easement",
    )

    def extract(self, text: str, file_name: str) -> dict[str, Any]:
        normalized = normalize_text(text)
        lower = normalized.lower()
        document_type = self._document_type(lower, file_name)
        parties = self._parties(normalized)
        clauses = self._clauses(normalized)
        low_confidence_sections = self._low_confidence_sections(normalized)
        dates = sorted(set(match.group(0) for match in self.DATE_RE.finditer(normalized)))
        case_numbers = sorted(
            set(
                match.group(1).rstrip(".,;")
                for match in self.CASE_RE.finditer(normalized)
                if any(char.isdigit() for char in match.group(1))
            )
        )
        amounts = sorted(set(match.group(0).replace(" ", "") for match in self.MONEY_RE.finditer(normalized)))

        missing_fields = []
        for field_name, value in {
            "document_type": document_type,
            "dates": dates,
            "parties": parties,
            "case_numbers": case_numbers,
        }.items():
            if not value:
                missing_fields.append(field_name)

        return {
            "document_type": document_type,
            "dates": dates,
            "parties": parties,
            "key_clauses": clauses,
            "referenced_case_numbers": case_numbers,
            "amounts": amounts,
            "low_confidence_sections": low_confidence_sections,
            "missing_fields": missing_fields,
        }

    def _document_type(self, lower_text: str, file_name: str) -> str | None:
        haystack = f"{file_name.lower()}\n{lower_text[:1200]}"
        candidates = [
            ("title review", ("title commitment", "schedule b", "exceptions", "legal description")),
            ("notice-related summary", ("notice of default", "notice to cure", "default notice")),
            ("case fact record", ("complaint", "plaintiff", "defendant", "case no")),
            ("lease or contract", ("lease agreement", "lessor", "lessee", "agreement")),
            ("internal memo", ("memorandum", "memo", "re:")),
        ]
        for label, markers in candidates:
            if any(marker in haystack for marker in markers):
                return label
        return None

    def _parties(self, text: str) -> list[str]:
        parties: set[str] = set()
        label_re = re.compile(
            r"^(?:Seller|Buyer|Borrower|Lender|Landlord|Tenant|Plaintiff|Defendant|Owner|Grantor|Grantee|Client|Counterparty)\s*:\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in label_re.finditer(text):
            value = re.split(r"\s{2,}|;|\n", match.group(1).strip())[0]
            if value:
                parties.add(value.rstrip(".,"))

        between_re = re.compile(r"\bbetween\s+(.{2,80}?)\s+and\s+(.{2,80}?)(?:\.|,|\n)", re.IGNORECASE)
        for match in between_re.finditer(text):
            for item in match.groups():
                cleaned = item.strip(" .,:;")
                if cleaned:
                    parties.add(cleaned)

        caption_re = re.compile(r"\b([A-Z][A-Za-z0-9 &.,'-]{2,60})\s+v\.?\s+([A-Z][A-Za-z0-9 &.,'-]{2,60})")
        for match in caption_re.finditer(text):
            parties.add(match.group(1).strip(" .,:;"))
            parties.add(match.group(2).strip(" .,:;"))

        return sorted(parties)

    def _clauses(self, text: str) -> list[dict[str, str]]:
        clauses: list[dict[str, str]] = []
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        seen: set[str] = set()
        for paragraph in paragraphs:
            lower = paragraph.lower()
            matched = next((kw for kw in self.CLAUSE_KEYWORDS if kw in lower), None)
            if not matched:
                continue
            key = stable_id(paragraph, 10)
            if key in seen:
                continue
            seen.add(key)
            heading = self._heading(paragraph, matched)
            clauses.append({"topic": matched, "heading": heading, "excerpt": short_quote(paragraph, 280)})
        return clauses[:12]

    def _heading(self, paragraph: str, fallback: str) -> str:
        first_line = paragraph.splitlines()[0].strip(" #:")
        if 3 <= len(first_line) <= 90:
            return first_line
        return fallback.title()

    def _low_confidence_sections(self, text: str) -> list[dict[str, str]]:
        markers = ("[illegible]", "[unclear]", "[?]", "???", "____", "unreadable", "smudged")
        sections: list[dict[str, str]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            if any(marker in lower for marker in markers):
                sections.append({"line": str(line_no), "text": short_quote(line, 180), "reason": "explicit illegibility marker"})
            elif len(line) > 20:
                alpha = sum(1 for char in line if char.isalpha())
                strange = sum(1 for char in line if char in "#*@~^")
                if alpha / max(len(line), 1) < 0.25 or strange >= 4:
                    sections.append({"line": str(line_no), "text": short_quote(line, 180), "reason": "low alphabetic signal"})
        return sections[:20]


class DocumentProcessor:
    """Turns heterogeneous legal files into normalized text and structured JSON."""

    def __init__(self, ocr_language: str = "eng") -> None:
        self.ocr = TesseractOCR(ocr_language)
        self.structured_extractor = StructuredDataExtractor()

    def process_path(self, path: str | Path) -> ProcessedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            pages, warnings, metadata = self._extract_text_file(path)
        elif suffix in PDF_SUFFIXES:
            pages, warnings, metadata = self._extract_pdf(path)
        elif suffix in IMAGE_SUFFIXES:
            pages, warnings, metadata = self._extract_image(path)
        else:
            pages, warnings, metadata = self._extract_text_file(path, forced=True)
            warnings.append(
                ExtractionWarning(
                    message=f"Unknown file extension '{suffix}'. Attempted plain text extraction.",
                    severity="warning",
                )
            )

        raw_text = normalize_text("\n\n".join(page.text for page in pages if page.text))
        if not raw_text:
            warnings.append(
                ExtractionWarning(
                    message="No usable text was extracted. Downstream generation should report insufficient evidence.",
                    severity="error",
                )
            )
        structured_data = self.structured_extractor.extract(raw_text, path.name)
        return ProcessedDocument.create(path, raw_text, structured_data, pages, warnings, metadata)

    def process_directory(self, input_dir: str | Path) -> list[ProcessedDocument]:
        input_dir = Path(input_dir)
        documents: list[ProcessedDocument] = []
        for path in sorted(item for item in input_dir.iterdir() if item.is_file()):
            documents.append(self.process_path(path))
        return documents

    def write_processed(self, document: ProcessedDocument, output_dir: str | Path) -> Path:
        output_dir = ensure_dir(Path(output_dir))
        output_path = output_dir / f"{document.document_id}.json"
        output_path.write_text(json.dumps(document.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _extract_text_file(
        self, path: Path, forced: bool = False
    ) -> tuple[list[PageText], list[ExtractionWarning], dict[str, Any]]:
        warnings: list[ExtractionWarning] = []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            text = path.read_text(errors="replace")
        confidence = self._estimate_confidence(text, "plain_text")
        if forced and not text.strip():
            warnings.append(ExtractionWarning(message="Forced text extraction returned an empty document.", severity="error"))
        if confidence < 0.6:
            warnings.append(ExtractionWarning(message="Text appears noisy or partially illegible.", severity="warning"))
        return [PageText(1, normalize_text(text), confidence, "plain_text")], warnings, {"page_count": 1}

    def _extract_pdf(self, path: Path) -> tuple[list[PageText], list[ExtractionWarning], dict[str, Any]]:
        warnings: list[ExtractionWarning] = []
        if fitz is None:
            warnings.append(ExtractionWarning(message="PyMuPDF is not installed; PDF text could not be extracted.", severity="error"))
            return [], warnings, {"page_count": 0, "pdf_backend": "unavailable"}

        pages: list[PageText] = []
        with fitz.open(path) as pdf:
            for index, page in enumerate(pdf, start=1):
                digital_text = normalize_text(page.get_text("text") or "")
                if len(digital_text) >= 35:
                    confidence = self._estimate_confidence(digital_text, "pdf_text")
                    pages.append(PageText(index, digital_text, confidence, "pdf_text"))
                    if confidence < 0.6:
                        warnings.append(ExtractionWarning(message="PDF page text appears noisy.", page=index))
                    continue

                if self.ocr.available:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            image_path = Path(temp_dir) / f"page-{index}.png"
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                            pix.save(image_path)
                            ocr_text, ocr_confidence = self.ocr.extract(image_path)
                        pages.append(PageText(index, ocr_text, ocr_confidence, "pdf_ocr_tesseract"))
                        if ocr_confidence < 0.55:
                            warnings.append(ExtractionWarning(message="OCR confidence is low.", page=index, details={"confidence": ocr_confidence}))
                    except Exception as exc:
                        warnings.append(ExtractionWarning(message=f"OCR failed for PDF page: {exc}", severity="error", page=index))
                        pages.append(PageText(index, "", 0.0, "pdf_ocr_failed"))
                else:
                    warnings.append(
                        ExtractionWarning(
                            message="PDF page has little or no embedded text and Tesseract OCR is unavailable.",
                            severity="warning",
                            page=index,
                        )
                    )
                    pages.append(PageText(index, digital_text, 0.15, "pdf_text_empty_no_ocr"))
        return pages, warnings, {"page_count": len(pages), "pdf_backend": "pymupdf", "ocr_available": self.ocr.available}

    def _extract_image(self, path: Path) -> tuple[list[PageText], list[ExtractionWarning], dict[str, Any]]:
        warnings: list[ExtractionWarning] = []
        if not self.ocr.available:
            warnings.append(
                ExtractionWarning(
                    message="Image input requires OCR, but Tesseract is unavailable. File was accepted and flagged.",
                    severity="warning",
                )
            )
            return [PageText(1, "", 0.0, "image_no_ocr")], warnings, {"page_count": 1, "ocr_available": False}
        try:
            text, confidence = self.ocr.extract(path)
        except Exception as exc:
            warnings.append(ExtractionWarning(message=f"Image OCR failed: {exc}", severity="error"))
            return [PageText(1, "", 0.0, "image_ocr_failed")], warnings, {"page_count": 1, "ocr_available": True}
        if confidence < 0.55:
            warnings.append(ExtractionWarning(message="Image OCR confidence is low.", details={"confidence": confidence}))
        return [PageText(1, text, confidence, "image_ocr_tesseract")], warnings, {"page_count": 1, "ocr_available": True}

    def _estimate_confidence(self, text: str, method: str) -> float:
        cleaned = normalize_text(text)
        if not cleaned:
            return 0.0
        base = 0.94 if method in {"plain_text", "pdf_text"} else 0.74
        marker_penalty = 0.09 * sum(cleaned.lower().count(marker) for marker in ("[illegible]", "[unclear]", "[?]", "???", "____"))
        alpha_ratio = sum(1 for char in cleaned if char.isalpha()) / max(len(cleaned), 1)
        if alpha_ratio < 0.35:
            base -= 0.25
        if len(cleaned) < 40:
            base -= 0.2
        return max(0.05, min(0.99, base - marker_penalty))
