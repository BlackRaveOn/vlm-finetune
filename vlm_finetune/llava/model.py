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
        model (LlavaModel): инстанс предобученной модели LLaVA.
        processor (LlavaProcessor): процессор для преобразования изображений и текста.
        image_processor (ImageProcessor | None): опциональный препроцессор изображений,
            позволяющий использовать пользовательские преобразования перед подачей в модель.

    Особенности:
        • поддержка LoRA-дообучения;
        • единый интерфейс для тренировки и инференса;
        • поддержка изображений как PIL.Image, так и string-пути;
        • совместим с фабрикой `AutoVlmModel` и абстракцией `VlmModel`.

    Пример:
        >>> model = AutoVlmModel.create("llava", ...)
        >>> model.finetune(dataset_path, output_dir="./ckpt")
        >>> model.predict("cat.png", "Что за животное на фото?")
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
            model (LlavaModel): предобученная модель LLaVA.
            processor (LlavaProcessor): процессор для обработки изображений и текстов.
            image_processor (ImageProcessor | None): кастомный препроцессор изображений.

        Особенности реализации:
            • вызывает базовый конструктор `VlmModel`;
            • позволяет переопределить пайплайн обработки изображений.
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
            dataset_path (list[dict[str,str]]): список объектов с путями к изображениям и текстом.
            output_dir (str): директория для сохранения чекпоинтов.
            learning_rate (float): скорость обучения (default: 2e-4).
            num_train_epochs (int): количество эпох обучения.
            prompt (str | None): шаблон системного промпта для датасета.
            lora_params (dict[str,Any] | None): параметры LoRA-адаптации.
            training_params (dict[str,Any] | None): параметры тренировки (trainer, scheduler и т.д.).

        Возвращает:
            None

        Исключения:
            ValueError: если датасет пуст или содержит некорректные записи.

        Детали:
            • создаёт `LLavaDataset` с процессором и опциональным image_processor;
            • делегирует обучение базовому `VlmModel.finetune`.

        Использование:
            >>> model.finetune(dataset, "./output", learning_rate=1e-4)
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
            image (PIL.Image | str): изображение или путь к изображению.
            prompt (str): текстовый запрос.
            max_new_tokens (int): ограничение длины ответа модели.

        Возвращает:
            str: текстовый ответ модели без служебных префиксов (`ASSISTANT:`).

        Исключения:
            RuntimeError: при ошибке генерации или обработке изображения.

        Детали:
            • вызывает базовый метод `VlmModel.predict`;
            • post-processing: вырезает часть ответа после `ASSISTANT:`.

        Пример:
            >>> model.predict("dog.jpg", "Что изображено?")
            'На фото собака.'
        """
        model_reponse = super().predict(image=image, prompt=prompt, max_new_tokens=max_new_tokens)
        model_answer = model_reponse[0].split("ASSISTANT:")[1].strip()
        return model_answer
