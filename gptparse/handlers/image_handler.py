from typing import List
import os
from PIL import Image
from .base import FileHandler
from ..utils.image_utils import resize_image


class ImageHandler(FileHandler):
    """Handler for image files (PNG, JPG, JPEG)."""

    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg')

    def supports_file(self) -> bool:
        """Check if the file is a supported image format."""
        return self.file_path.lower().endswith(self.SUPPORTED_FORMATS)

    def get_images(self) -> List[Image.Image]:
        """Load and process the image file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Image file not found: {self.file_path}")

        try:
            image = Image.open(self.file_path)
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Resize if the image is too large
            image = resize_image(image, max_size=4096)
            
            return [image]
        except Exception as e:
            raise ValueError(f"Error processing image {self.file_path}: {str(e)}")

    @property
    def is_multi_page(self) -> bool:
        return False
