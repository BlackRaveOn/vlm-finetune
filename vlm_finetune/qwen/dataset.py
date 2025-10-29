"""
Модуль для подготовки и токенизации данных в формате Qwen2.5-VL
для обучения мультимодальных моделей (Vision-Language Models).

Основная задача — преобразовать сырые данные (изображения и текстовые диалоги)
в формат, совместимый с `Qwen2_5_VLProcessor` и ожидаемым протоколом обучения
моделей семейства Qwen-VL из библиотеки `transformers`.

Каждый элемент датасета представляет собой диалог:
    • Пользователь (user) передаёт изображение + текстовый запрос
    • Модель (assistant) генерирует текстовый ответ

Формат данных основан на chat-template Qwen-VL:
    [{"role": "user", "content":[{"type":"image"}, {"type":"text","text":"..."}]},
     {"role": "assistant", "content":[{"type":"text","text":"..."}]}]

Класс:
    QwenDataset — PyTorch Dataset, готовящий батч из текста и изображения,
                  формируя корректные `input_ids`, `pixel_values`, `attention_mask`, `labels`
"""
import torch
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

from vlm_finetune import ImageProcessor
from vlm_finetune.base import VlmDataset


class QwenDataset(VlmDataset):
    """
    Датасет для обучения мультимодели Qwen-VL (Qwen2.5-VL).

    Датасет считывает JSON с путями к изображениям, пользовательскими промптами и ответами модели,
    затем формирует корректный чат-диалог и токенизирует его через `Qwen2_5_VLProcessor`.

    Формат входного JSON:
    [
        {
            "image_path": "path/to/image.jpg",
            "prompt":  "Что на изображении?",
            "answer":  "Собака сидит на траве."
        },
        ...
    ]

    Атрибуты:
        data: list[dict]
            Загруженные записи датасета (image_path, prompt, answer).
        processor: Qwen2_5_VLProcessor
            Процессор HuggingFace, отвечающий за токенизацию и подготовку изображения.
        prompt: str | None
            Общий промпт, используется, если в элементе отсутствует свой.
        image_processor: ImageProcessor
            Кастомный компонент для чтения/преобразования изображений перед подачей в модель.
    """

    def __init__(self, dataset_path: str, processor: Qwen2_5_VLProcessor, image_processor: ImageProcessor, prompt: str | None):
        """
        Инициализация датасета.

        Параметры:
            dataset_path: str
                Путь к JSON-файлу с образцами.
            processor: Qwen2_5_VLProcessor
                Процессор токенизации + обработки картинки под Qwen-VL.
            image_processor: ImageProcessor
                Пользовательский препроцессинг изображений (resize, normalize и т.д.).
            prompt: str | None
                Общий промпт, если поле `prompt` отсутствует у примера.

        Исключения:
            FileNotFoundError — если JSON не найден.
            json.JSONDecodeError — если JSON некорректен.
        """
        super().__init__(
            dataset_path=dataset_path,
            processor=processor,
            prompt=prompt,
            image_processor=image_processor
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Подготавливает единичный обучающий пример для Qwen-VL.

        Действия:
            1) Загружает изображение
            2) Формирует чат-структуру формата Qwen-VL
            3) Применяет chat-template → чистый текст
            4) Токенизирует текст + изображение
            5) Создаёт `labels`, маскируя всё до роли `assistant`

        Параметры:
            idx: int — индекс элемента

        Возвращает:
            dict[str, torch.Tensor], содержащий:
                input_ids: ids токенов текста
                attention_mask: маска внимания
                pixel_values: закодированное изображение
                labels: тензор целевых токенов (prefix masked to -100)

        Исключения:
            KeyError — отсутствует ключ в JSON (например image_path)
            FileNotFoundError — если отсутствует изображение
            ValueError — если токенизация/обработка не удалась
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

        text: str = self.processor.apply_chat_template(
            conversation,
            tokenize=False
        )

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        input_ids: torch.Tensor = encoding["input_ids"][0]

        # Определяем границу между user-частью и assistant-ответом
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start != -1:
            prefix = self.processor.tokenizer(text[:assistant_start], return_tensors="pt")
            cutoff = prefix["input_ids"].size(1)
        else:
            cutoff = 0

        labels = input_ids.clone()
        labels[:cutoff] = -100  # скрываем prompt/user часть
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Маскируем спец-токены изображения (Qwen-VL style)
        image_token_id = 151655
        labels[labels == image_token_id] = -100

        encoding["labels"] = labels.unsqueeze(0)

        return {k: v.squeeze(0) for k, v in encoding.items()}
