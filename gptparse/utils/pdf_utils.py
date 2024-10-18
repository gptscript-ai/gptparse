from PyPDF2 import PdfReader, PdfWriter
from typing import List
import io


def split_pdf_into_chunks(pdf_path: str, chunk_size: int = 10) -> List[bytes]:
    """Split a PDF into chunks of specified size."""
    reader = PdfReader(pdf_path)
    chunks = []
    for i in range(0, len(reader.pages), chunk_size):
        writer = PdfWriter()
        for j in range(i, min(i + chunk_size, len(reader.pages))):
            writer.add_page(reader.pages[j])
        with io.BytesIO() as bytes_stream:
            writer.write(bytes_stream)
            chunks.append(bytes_stream.getvalue())
    return chunks
