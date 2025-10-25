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
from PIL import Image
from torch.utils.data import Dataset
from transformers.models.llava.processing_llava import LlavaProcessor


class LLavaDataset(Dataset):
    """
    Класс датасета для обучения мультимодели LLaVA.

    Данный класс загружает данные из JSON, содержащего пути к изображениям, тексты запросов и ответов,
    и с помощью процессора `LlavaProcessor` преобразует их в тензоры, совместимые с моделью LLaVA.

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

    def __init__(self, dataset_path: str, processor: LlavaProcessor, prompt: str | None):
        """
        Инициализация датасета.

        Параметры:
            dataset_path: str
                Путь к JSON-файлу с данными датасета.
            processor: LlavaProcessor
                Процессор из библиотеки `transformers`, выполняющий токенизацию текста
                и преобразование изображений.
            prompt: str | None
                Базовый текстовый промпт, который будет использован, если в примере отсутствует свой.
        
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

    def __len__(self) -> int:
        """
        Возвращает количество элементов в датасете.

        Возвращает:
            int: Количество примеров (строк) в датасете.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Возвращает один элемент датасета, преобразованный в формат, пригодный для обучения модели.

        Параметры:
            idx: int
                Индекс запрашиваемого примера.

        Возвращает:
            dict[str, torch.Tensor]:
                Словарь с полями:
                    - "input_ids": тензор токенов входного текста;
                    - "attention_mask": маска внимания;
                    - "pixel_values": закодированные значения изображения;
                    - "labels": целевые токены для обучения (всё до "ASSISTANT:" замаскировано -100).

        Исключения:
            KeyError:
                Если в элементе отсутствует ключ `image_path`.
            FileNotFoundError:
                Если указанный путь к изображению не существует.
            ValueError:
                Если процессор не смог корректно обработать текст или изображение.
        """
        item = self.data[idx]
        image_path = item["image_path"]
        answer = item["answer"]
        prompt = item.get("prompt", self.prompt)

        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer + self.processor.tokenizer.eos_token},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoding: dict[str, torch.Tensor] = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=False
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)

        # 🔍 Находим позицию начала "ASSISTANT:"
        tokenized_assistant: torch.Tensor = self.processor.tokenizer(
            "ASSISTANT:",
            add_special_tokens=False
        )["input_ids"]

        start_idx = None
        for i in range(len(input_ids) - len(tokenized_assistant)):
            if torch.equal(input_ids[i:i + len(tokenized_assistant)], torch.tensor(tokenized_assistant)):
                start_idx = i + len(tokenized_assistant)
                break

        if start_idx is None:
            start_idx = 0  # fallback

        labels = input_ids.clone()
        labels[:start_idx] = -100  # маскируем всё до начала ответа

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
