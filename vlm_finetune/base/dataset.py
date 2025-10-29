

"""
Модуль для подготовки и токенизации данных в формате LLaVA (Large Language and Vision Assistant)
для обучения и дообучения мультимодальных моделей.

Основная цель — преобразовать сырые данные (изображения и текстовые промпты) в формат,
удобный для подачи в модель `LlavaModel` из библиотеки `transformers`.  
Каждый элемент датасета представляет собой диалог пользователя и ассистента, где
вход включает изображение и текстовый запрос, а выход — текстовый ответ модели.

Классы:
    LLavaDataset: Класс PyTorch Dataset для обработки и токенизации данных LLaVA.
"""

import json
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from transformers import ProcessorMixin
from vlm_finetune.base.image import ImageProcessor


class VlmDataset(Dataset, ABC):
    """
    Класс датасета для обучения мультимодели LLaVA.

    Данный класс загружает данные из JSON, содержащего пути к изображениям, тексты запросов и ответов,
    и с помощью процессора `ProcessorMixin` преобразует их в тензоры, совместимые с моделью LLaVA.

    Пример структуры входного JSON:
    [
        {
            "image_path": "path/to/image1.jpg",
            "prompt": "Что изображено на фото?",
            "answer": "На фото изображена собака."
        },
        ...
    ]

    Атрибуты:
        data (list[dict[str, str]]): Загруженные из JSON данные.
        processor (LlavaProcessor): Процессор для токенизации и преобразования изображений.
        prompt (str | None): Общий промпт, если он не задан для конкретного примера.
    """

    def __init__(self, dataset_path: str, processor: ProcessorMixin,  image_processor: ImageProcessor, prompt: str | None):
        """
        Инициализация датасета.

        Параметры:
            dataset_path: str
                Путь к JSON-файлу с данными датасета.
            processor: ProcessorMixin
                Процессор из библиотеки `transformers`, выполняющий токенизацию текста
                и преобразование изображений.
            prompt: str | None
                Базовый текстовый промпт, который будет использован, если в примере отсутствует свой.
            image_size: tuple[int, int] | None = None
                Размер картинки, если не указан - то берётся вся картинка. В ином случае ожидается кортеж нового размера картинки
        
        Исключения:
            FileNotFoundError:
                Если указанный JSON-файл не существует.
            json.JSONDecodeError:
                Если JSON-файл имеет некорректный формат.
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
            int: Количество примеров (строк) в датасете.
        """
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Возвращает один элемент датасета, преобразованный в формат, пригодный для обучения модели."""
        pass
