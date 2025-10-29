"""
Модуль `llava_dataset` содержит реализацию датасета `LLavaDataset` для подготовки данных
в формате LLaVA (Large Language and Vision Assistant). Модуль обеспечивает преобразование
сырых данных (изображения и текстовых пар «вопрос — ответ») в формат, совместимый
с мультимодальными моделями LLaVA из библиотеки `transformers`.

Основные задачи:
    • чтение примеров из JSON-файла или списка словарей;
    • загрузка и предварительная обработка изображений;
    • формирование диалоговой структуры пользователь → ассистент;
    • токенизация текста и кодирование изображения;
    • подготовка `labels` с маскированием пользовательской части.

Структура входных данных:
    [
        {
            "image_path": "path/to/file.jpg",
            "prompt": "Что изображено на фото?",
            "answer": "На изображении находится собака."
        }
    ]

Датасет используется для обучения LLaVA-совместимых моделей на собственных данных.
"""

import torch
from transformers.models.llava.processing_llava import LlavaProcessor

from vlm_finetune import ImageProcessor
from vlm_finetune.base import VlmDataset


class LLavaDataset(VlmDataset):
    """
    Датасет для обучения мультимодели LLaVA.

    Расширяет `VlmDataset` и формирует корректные обучающие примеры для задач
    визуально-лингвистического обучения.

    Особенности:
        • автоматически добавляет изображение в prompt-часть;
        • строит диалоговый формат в стиле LLaVA;
        • маскирует все токены до начала текста ассистента (`labels[:start] = -100`);
        • совместим с `Trainer` из HuggingFace.

    Атрибуты:
        data (list[dict[str, str]]): загруженные элементы датасета
        processor (LlavaProcessor): токенайзер и image-encoder
        prompt (str | None): базовый промпт, если не указан в элементе
        image_processor (ImageProcessor): util для чтения/обработки изображений
    """

    def __init__(
        self,
        dataset_path: str,
        processor: LlavaProcessor,
        image_processor: ImageProcessor,
        prompt: str | None
    ):
        """
        Инициализация датасета.

        Параметры:
            dataset_path (str):
                Путь к JSON или список примеров.
            processor (LlavaProcessor):
                Процессор `transformers` для текста и изображений.
            image_processor (ImageProcessor):
                Компонент для чтения/предобработки изображения.
            prompt (str | None):
                Базовый текстовый шаблон, если в примере отсутствует свой.

        Исключения:
            FileNotFoundError — если JSON-файл отсутствует
            json.JSONDecodeError — некорректный формат JSON
        """
        super().__init__(
            dataset_path=dataset_path,
            processor=processor,
            prompt=prompt,
            image_processor=image_processor
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Возвращает один преобразованный пример.

        Строит последовательность вида:
            USER: <image> <prompt>
            ASSISTANT: <answer>

        Затем:
            • токенизирует текст;
            • кодирует изображение;
            • формирует labels, маскируя часть prompt (`-100`);
            • возвращает тензоры для Trainer.

        Параметры:
            idx: int
                Индекс запрашиваемого примера.

        Возвращает:
            dict[str, torch.Tensor]:
                {
                    "input_ids": Tensor,
                    "attention_mask": Tensor,
                    "pixel_values": Tensor,
                    "labels": Tensor
                }

        Исключения:
            KeyError — отсутствует `image_path` или `answer`
            FileNotFoundError — изображение не найдено
            ValueError — ошибка при обработке текста/изображения
        """
        item = self.data[idx]
        image_path = item["image_path"]
        answer = item["answer"]
        prompt = item.get("prompt", self.prompt)

        image = self.image_processor.process_image(image_path=image_path)

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

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=False
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)

        # Находим позицию начала "ASSISTANT:"
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
