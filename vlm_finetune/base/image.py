from PIL import Image


class ImageProcessor:

    def __init__(self, image_size: tuple[int, int] | None = None):
        self.image_size = image_size

    def process_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size, Image.LANCZOS)
        return image