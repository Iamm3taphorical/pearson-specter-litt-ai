"""Generate synthetic PDF/image sample inputs.

This script is optional; text samples are already committed. It creates a
digital PDF and an image-style notice so reviewers can exercise PDF/image paths.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_inputs"


def create_pdf() -> None:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        print(f"Skipping PDF sample; PyMuPDF unavailable: {exc}")
        return

    text = (
        "SYNTHETIC TITLE REVIEW ADDENDUM\n\n"
        "Matter No: PSL-2026-0142\n"
        "Date: May 6, 2026\n"
        "Parties: Ridge Harbor Capital LLC and West 46th Street Holdings, Inc.\n\n"
        "Addendum: The title company confirmed the utility easement recorded in Liber 4102, Page 778 remains open.\n"
        "The addendum does not resolve the handwritten roof access notation. Counsel should verify the original scan.\n\n"
        "Notice Clause: courier and email delivery are both required before notice is effective.\n"
    )
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rect = fitz.Rect(72, 72, 540, 720)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv", align=0)
    doc.save(SAMPLE_DIR / "synthetic_title_addendum.pdf")
    doc.close()


def create_image() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"Skipping image sample; Pillow unavailable: {exc}")
        return

    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = [
        "LOW-RES NOTICE ATTACHMENT",
        "Matter No: PSL-2026-0142",
        "Courier receipt dated May 1, 2026",
        "Tracking: 1Z-9A7-[?]-4021",
        "Signature line is smudged / partially unreadable.",
    ]
    y = 50
    for line in lines:
        draw.text((55, y), line, fill="black", font=font)
        y += 65
    image.save(SAMPLE_DIR / "low_res_notice_attachment.png")


if __name__ == "__main__":
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    create_pdf()
    create_image()
