"""
Модуль реализует абстрактный базовый класс `VlmDataset` для подготовки
и токенизации данных мультимоделей Vision-Language Models (VLM), таких как LLaVA и Qwen.

Основная цель:
    • Унификация работы с JSON-данными, содержащими изображения и текстовые промпты;
    • Преобразование данных в формат, совместимый с процессорами `transformers`.

Классы:
    VlmDataset — базовый абстрактный класс PyTorch Dataset для подготовки данных.

Особенности:
    • Может использоваться для разных VLM (LLava, Qwen, и т.д.);
    • Дочерние классы реализуют метод `__getitem__` для конкретной токенизации;
    • Поддерживает кастомный препроцессор изображений (`ImageProcessor`).

Исключения:
    FileNotFoundError — если указанный JSON-файл не существует.
    json.JSONDecodeError — если JSON-файл имеет некорректный формат.
"""

import json
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from transformers import ProcessorMixin

from vlm_finetune.base.image import ImageProcessor


class VlmDataset(Dataset, ABC):
    """
    Абстрактный базовый класс датасета для мультимоделей Vision-Language Models (VLM).

    Класс загружает данные из JSON-файла, содержащего пути к изображениям, текстовые
    запросы и ответы, и предоставляет интерфейс для преобразования их в тензоры,
    совместимые с процессорами `transformers`.

    Пример структуры входного JSON:
        [
            {
                "image_path": "path/to/image1.jpg",
                "prompt": "Текстовый запрос к модели",
                "answer": "Текстовый ответ модели"
            },
            ...
        ]

    Атрибуты:
        data (list[dict[str, str]]): Загруженные данные из JSON-файла.
        processor (ProcessorMixin): Процессор `transformers` для токенизации текста
            и обработки изображений.
        prompt (str | None): Общий системный промпт, используемый, если он не задан
            для конкретного примера.
        image_processor (ImageProcessor): Препроцессор изображений, выполняющий
            преобразование изображений перед подачей в модель.
    """

    def __init__(
        self,
        dataset_path: str,
        processor: ProcessorMixin,
        image_processor: ImageProcessor,
        prompt: str | None
    ):
        """
        Инициализация экземпляра датасета.

        Параметры:
            dataset_path (str): путь к JSON-файлу с данными.
            processor (ProcessorMixin): процессор из `transformers`, выполняющий
                токенизацию текста и обработку изображений.
            image_processor (ImageProcessor): кастомный препроцессор изображений.
            prompt (str | None): базовый текстовый промпт, если он не задан в примере.

        Исключения:
            FileNotFoundError: если указанный JSON-файл отсутствует.
            json.JSONDecodeError: если JSON-файл имеет некорректный формат.
        """
        with open(dataset_path, encoding="utf-8") as f:
            self.data: list[dict[str, str]] = json.load(f)
        self.processor = processor
        self.prompt = prompt
        self.image_processor = image_processor

    def __len__(self) -> int:
        """
        Возвращает количество элементов в датасете.

        Возвращает:
            int: Количество примеров в датасете.
        """
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Абстрактный метод для получения одного элемента датасета.

        Метод должен быть реализован в дочерних классах и возвращать словарь с тензорами,
        готовыми для обучения конкретной VLM-модели.

        Параметры:
            idx (int): индекс запрашиваемого примера.

        Возвращает:
            Возвращает элемент датасета, преобразованный в формат, пригодный для обучения конкретной модели.
        """
        pass
