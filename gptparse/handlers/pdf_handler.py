from typing import List
import os
from PIL import Image
from pdf2image import convert_from_bytes
from .base import FileHandler
from ..utils.pdf_utils import split_pdf_into_chunks


class PDFHandler(FileHandler):
    """Handler for PDF files."""

    def supports_file(self) -> bool:
        """Check if the file is a PDF."""
        return self.file_path.lower().endswith('.pdf')

    def get_images(self) -> List[Image.Image]:
        """Convert PDF pages to images."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        images = []
        pdf_chunks = split_pdf_into_chunks(self.file_path)
        
        for chunk in pdf_chunks:
            # Convert each chunk to images with reasonable DPI for OCR
            chunk_images = convert_from_bytes(
                chunk,
                thread_count=4,
                dpi=300,
                fmt='PNG',
                grayscale=False,
                size=(None, None)  # Maintain original size
            )
            images.extend(chunk_images)
        
        return images

    @property
    def is_multi_page(self) -> bool:
        return True
