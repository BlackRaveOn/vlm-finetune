"""
Модуль `llava_model` реализует класс `LLavaModel`, предназначенный для дообучения и
инференса мультимодальной модели LLaVA (Large Language and Vision Assistant).

Основные возможности:
    • Загрузка и настройка модели через фабрику `AutoVlmModel`.
    • Дообучение модели с использованием LoRA (Low-Rank Adaptation).
    • Генерация текстовых ответов по изображениям и промптам.

Классы:
    LLavaModel — класс, реализующий дообучение и предсказание для модели LLaVA.

Исключения:
    ValueError — при некорректных данных в датасете или ошибках в параметрах обучения.
"""

from typing import Any, override

from PIL import Image
from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava.processing_llava import LlavaProcessor

from vlm_finetune import AutoVlmModel, ImageProcessor
from vlm_finetune.base import VlmModel
from vlm_finetune.llava.dataset import LLavaDataset
from vlm_finetune.utils import set_logger

logger = set_logger(__name__)

@AutoVlmModel.register("llava")
class LLavaModel(VlmModel):
    """
    Класс `LLavaModel` представляет собой адаптер для модели LLaVA,
    обеспечивающий удобный интерфейс для обучения и инференса.

    Атрибуты:
        model: экземпляр базовой модели LLaVA.
        processor: процессор для преобразования данных (изображений и текста).
    """

    def __init__(
        self,
        model: LlavaModel,
        processor: LlavaProcessor,
        image_processor: ImageProcessor | None = None
    ) -> None:
        """
        Инициализация экземпляра `LLavaModel`.

        Параметры:
            model: предобученная модель LLaVA.
            processor: процессор для обработки изображений и текстов.
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
        Дообучает модель LLaVA на пользовательском датасете с использованием LoRA.

        Параметры:
            dataset_path: путь к датасету или список объектов, содержащих пути к изображениям и текстам.
            output_dir: директория для сохранения результатов обучения.
            learning_rate: скорость обучения (по умолчанию 2e-4).
            num_train_epochs: количество эпох обучения.
            prompt: шаблон текстового промпта, если требуется.
            lora_params: словарь параметров для LoRA (по умолчанию `DEFAULT_LORA`).
            training_params: параметры обучения (по умолчанию `DEFAULT_TRAINING`).

        Возвращает:
            None

        Исключения:
            ValueError: если датасет пуст или некорректен.
        """
        dataset = LLavaDataset(dataset_path=dataset_path, processor=self.processor, prompt=prompt, image_processor=self.image_processor)
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
        Выполняет генерацию текстового ответа по изображению и текстовому запросу.

        Параметры:
            image: изображение (объект PIL.Image или путь к файлу).
            prompt: текстовый запрос пользователя.
            max_new_tokens: максимальное количество генерируемых токенов (по умолчанию 256).

        Возвращает:
            Текстовый ответ модели.

        Исключения:
            RuntimeError: при ошибке генерации ответа.
        """
        model_reponse = super().predict(image=image, prompt=prompt, max_new_tokens=max_new_tokens)
        model_answer = model_reponse[0].split("ASSISTANT:")[1].strip()
        return model_answer
