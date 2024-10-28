from typing import Type
from .base import FileHandler
from .pdf_handler import PDFHandler
from .image_handler import ImageHandler


def get_handler(file_path: str) -> FileHandler:
    """
    Factory function to get the appropriate handler for a file.
    
    Args:
        file_path: Path to the file to be processed
        
    Returns:
        An instance of the appropriate FileHandler
        
    Raises:
        ValueError: If no handler supports the file type
    """
    handlers: List[Type[FileHandler]] = [PDFHandler, ImageHandler]
    
    for handler_class in handlers:
        handler = handler_class(file_path)
        if handler.supports_file():
            return handler
            
    raise ValueError(
        f"Unsupported file type. Supported formats: PDF, PNG, JPG, JPEG"
    )


__all__ = ['FileHandler', 'PDFHandler', 'ImageHandler', 'get_handler']
