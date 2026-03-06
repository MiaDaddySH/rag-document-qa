from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(file_path))

    pages = []
    full_text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = text.strip()

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
            }
        )

        if cleaned_text:
            full_text_parts.append(cleaned_text)

    full_text = "\n\n".join(full_text_parts)

    return {
        "filename": file_path.name,
        "page_count": len(reader.pages),
        "text": full_text,
        "pages": pages,
    }