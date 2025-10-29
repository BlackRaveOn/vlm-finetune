"""
Модуль `image_processor` реализует класс `ImageProcessor` для подготовки изображений
к подаче в мультимодели Vision-Language (VLM), такие как LLaVA или Qwen.

Основные возможности:
    • Загрузка изображения с диска и приведение к RGB.
    • Изменение размера изображения при необходимости.
    • Унифицированная подготовка изображений для моделей и процессоров `transformers`.

Классы:
    ImageProcessor — класс для обработки и масштабирования изображений.

Исключения:
    FileNotFoundError — если указанный путь к изображению не существует.
"""

from PIL import Image


class ImageProcessor:
    """
    Класс для загрузки и предварительной обработки изображений перед подачей в модель VLM.

    Атрибуты:
        image_size (tuple[int, int] | None): размер изображения после масштабирования. 
            Если None, сохраняется исходный размер.
    """

    def __init__(self, image_size: tuple[int, int] | None = None):
        """
        Инициализация процессора изображений.

        Параметры:
            image_size: кортеж (ширина, высота) для масштабирования изображений. 
                        Если None, изображения остаются в исходном размере.
        """
        self.image_size = image_size

    def process_image(self, image_path: str) -> Image.Image:
        """
        Загружает изображение с диска, приводит его к RGB и изменяет размер при необходимости.

        Параметры:
            image_path: путь к изображению на диске.

        Возвращает:
            PIL.Image.Image: обработанное изображение в формате RGB.

        Исключения:
            FileNotFoundError: если файл по указанному пути не найден.
        """
        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size, Image.LANCZOS)
        return image
