from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class FileHandler(ABC):
    """Base class for file handlers that convert different file types to images."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def get_images(self) -> List[Image.Image]:
        """Convert input file to a list of PIL Images."""
        pass

    @abstractmethod
    def supports_file(self) -> bool:
        """Check if this handler supports the given file type."""
        pass

    @property
    def is_multi_page(self) -> bool:
        """Return True if the file type supports multiple pages."""
        return False
