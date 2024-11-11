from pathlib import Path
from typing import Optional, List
from PIL import Image

from docling.document_converter import DocumentConverter

from .base import FileHandler


class DoclingHandler(FileHandler):
    """Handler for OCR using docling"""

    def __init__(self, file_path: str, abort_on_error: bool = False):
        super().__init__(file_path)  # Call parent constructor with file_path
        self.abort_on_error = abort_on_error
        self.converter = DocumentConverter()
        self._images = None

    def get_images(self) -> List[Image.Image]:
        """Convert input file to a list of PIL Images."""
        # Simple implementation to satisfy abstract method
        return []

    def supports_file(self) -> bool:
        """Check if this handler supports the given file type."""
        ext = Path(self.file_path).suffix.lower()
        return ext in [".pdf", ".png", ".jpg", ".jpeg"]

    @property
    def is_multi_page(self) -> bool:
        """Return True if the file type supports multiple pages."""
        ext = Path(self.file_path).suffix.lower()
        return ext == ".pdf"

    def process(self, output_file: Optional[str] = None) -> str:
        """Process a file using docling

        Args:
            output_file: Optional path to output file

        Returns:
            Extracted text content
        """
        try:
            # Convert document
            result = self.converter.convert(self.file_path)

            # Get markdown content
            text = result.document.export_to_markdown()

            # Save to output file if specified
            if output_file:
                Path(output_file).write_text(text)

            return text

        except Exception as e:
            if self.abort_on_error:
                raise e
            return f"Error processing file: {str(e)}"
