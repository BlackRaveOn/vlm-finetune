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

import torch
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

from vlm_finetune import ImageProcessor
from vlm_finetune.base import VlmDataset


class QwenDataset(VlmDataset):
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

    def __init__(self, dataset_path: str, processor: Qwen2_5_VLProcessor, image_processor: ImageProcessor, prompt: str | None):
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
            image_size: tuple[int, int] | None = None
                Размер картинки, если не указан - то берётся вся картинка. В ином случае ожидается кортеж нового размера картинки
        
        Исключения:
            FileNotFoundError:
                Если указанный JSON-файл не существует.
            json.JSONDecodeError:
                Если JSON-файл имеет некорректный формат.
        """
        super().__init__(
            dataset_path=dataset_path, 
            processor=processor, 
            prompt=prompt, 
            image_processor=image_processor
        )

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

        image = self.image_processor.process_image(image_path=image_path)
        # Создаём диалог
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
                    {"type": "text", "text": answer},
                ],
            },
        ]

        # Получаем текстовую часть
        text: str = self.processor.apply_chat_template(
            conversation,
            tokenize=False
        )

        # Токенизация
        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        input_ids: torch.Tensor = encoding["input_ids"][0]

        # Находим индекс начала ассистента
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start != -1:
            # Токенизируем до этого места, чтобы узнать, где обрезать
            prefix = self.processor.tokenizer(text[:assistant_start], return_tensors="pt")
            cutoff = prefix["input_ids"].size(1)
        else:
            cutoff = 0

        labels = input_ids.clone()
        labels[:cutoff] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = 151655
        labels[labels == image_token_id] = -100

        encoding["labels"] = labels.unsqueeze(0)

        return {k: v.squeeze(0) for k, v in encoding.items()}
