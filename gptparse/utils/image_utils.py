from PIL import Image
import io
import base64


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize the image to fit within max_size x max_size while maintaining aspect ratio."""
    original_width, original_height = image.size

    if original_width <= max_size and original_height <= max_size:
        return image

    if original_width > original_height:
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))

    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
