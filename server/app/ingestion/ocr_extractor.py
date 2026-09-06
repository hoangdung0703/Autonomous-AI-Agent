"""Gemini Vision OCR fallback for scanned PDFs where pdfplumber extracts
fewer than MIN_TEXT_CHARS of text (Section 6.1 of requirements.md).

Each page is rendered to a PNG image via pdfplumber's Page.to_image() (which
is itself backed by pypdfium2 — already a transitive dependency of
pdfplumber, so no new package is needed) and sent to Gemini Vision for
transcription, one page at a time, then concatenated in order.
"""

import io

import pdfplumber

from app.services import llm_service
from app.utils.logger import logger

OCR_PROMPT = "Transcribe all text from this document page exactly, preserving structure and headings."
RESOLUTION = 150


async def extract_via_ocr(file_path: str) -> str:
    pages_text: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            image_bytes = _render_page_png(page)
            try:
                text = await llm_service.generate_from_image(image_bytes, OCR_PROMPT)
            except Exception as exc:
                logger.warning("OCR failed on page %d of %s: %s", page_number, file_path, exc)
                text = ""
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _render_page_png(page: pdfplumber.page.Page) -> bytes:
    image = page.to_image(resolution=RESOLUTION).original
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
