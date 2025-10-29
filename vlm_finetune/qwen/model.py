"""
Модуль `qwen_model` реализует класс `QwenModel`, предназначенный для
дообучения и инференса мультимодальной модели Qwen-VL (Qwen2.5-VL).

Функциональность:
    ─ Загрузка модели через фабрику `AutoVlmModel`
    ─ Дообучение на пользовательских данных с использованием LoRA
    ─ Генерация ответов по изображению и текстовому запросу

Модель использует архитектуру Qwen-VL, поддерживающую совместную
обработку изображений и текста и формат chat-инструкций, применяемый
в `Qwen2_5_VLProcessor`.

Класс:
    QwenModel — интерфейс обучения и инференса мультимодели Qwen-VL

Исключения:
    ValueError — при некорректных параметрах обучения или датасете
"""

from typing import Any, override

from PIL import Image
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

from vlm_finetune import AutoVlmModel, ImageProcessor
from vlm_finetune.base import VlmModel
from vlm_finetune.qwen.dataset import QwenDataset
from vlm_finetune.utils import set_logger

logger = set_logger(__name__)


@AutoVlmModel.register("qwen")
class QwenModel(VlmModel):
    """
    Класс `QwenModel` предоставляет интерфейс для дообучения и инференса
    модели Qwen-VL (Qwen2.5-VL).

    Служит адаптером над базовой моделью: подготавливает датасет,
    вызывает тренировочный pipeline, а также реализует удобный
    предикт-интерфейс для диалогового формата.

    Атрибуты:
        model — загруженная мультимодальная модель Qwen-VL
        processor — процессор для изображений и токенизации чата
        image_processor — объект для предобработки изображений
    """

    def __init__(
        self,
        model: Qwen2_5_VLModel,
        processor: Qwen2_5_VLProcessor,
        image_processor: ImageProcessor | None = None
    ) -> None:
        """
        Инициализация модели.

        Параметры:
            model — предобученная Qwen-VL модель
            processor — процессор токенизации и обработки изображений
            image_processor — пользовательский препроцессор изображений
        """
        super().__init__(model=model, processor=processor, image_processor=image_processor)

    @override
    def finetune(
        self,
        dataset_path: list[dict[str, str]],
        output_dir: str,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        prompt: str | None = None,
        lora_params: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Дообучает модель Qwen-VL на пользовательском датасете с LoRA-адаптацией.

        Ожидается JSON-датасет вида:
        {
            "image_path": "...",
            "prompt": "текст запроса",
            "answer": "ответ модели"
        }

        Параметры:
            dataset_path — путь к данным или список объектов
            output_dir — директория сохранения модели
            learning_rate — LR для обучения
            num_train_epochs — количество эпох
            prompt — глобальный промпт (если не задан в данных)
            lora_params — параметры LoRA
            training_params — доп. параметры обучения

        Исключения:
            ValueError — при пустом или неверном датасете
        """
        dataset = QwenDataset(
            dataset_path=dataset_path,
            processor=self.processor,
            prompt=prompt,
            image_processor=self.image_processor
        )

        super().finetune(
            dataset=dataset,
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lora_params=lora_params,
            training_params=training_params
        )

    @override
    def predict(self, image: Image.Image | str, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Генерирует текстовый ответ на изображение и пользовательский запрос.

        Параметры:
            image — путь или PIL.Image
            prompt — текстовый запрос
            max_new_tokens — ограничение длины ответа

        Возвращает:
            Строку — ответ модели
        """
        model_reponse = super().predict(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        # Разбор chat-формата, возвращаем только текст ассистента
        model_answer = model_reponse[0].split("assistant\n")[1]
        return model_answer
