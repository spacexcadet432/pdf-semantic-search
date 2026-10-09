
from pathlib import Path
import pymupdf

folder = Path(__file__).parent
pdf_path = folder / "book.pdf"
temp_path = folder / "book_trimmed.pdf"

with pymupdf.open(pdf_path) as source:
    if len(source) < 120:
        raise ValueError("The PDF has fewer than 120 pages.")

    with pymupdf.open() as output:
        output.insert_pdf(source, from_page=0, to_page=119)
        output.save(temp_path)

temp_path.replace(pdf_path)

print("PDF trimmed successfully.")
with pymupdf.open(pdf_path) as pdf:
    print("Final page count:", len(pdf))
    print("Text sample:", pdf[4].get_text()[:200])
