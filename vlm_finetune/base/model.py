"""
Модуль `vlm_model` реализует базовый абстрактный класс `VlmModel` для обучения и
инференса мультимодальных моделей Vision-Language Models (VLM), таких как LLaVA и Qwen.

Основные возможности:
    • Поддержка дообучения модели на пользовательских датасетах с использованием LoRA (Low-Rank Adaptation).
    • Удобный интерфейс для генерации текстовых ответов по изображениям и промптам.
    • Унификация работы с процессорами `transformers` и препроцессорами изображений.

Классы:
    VlmModel — базовый абстрактный класс для обучения и инференса VLM.

Исключения:
    ValueError — при некорректных данных в датасете или ошибках в параметрах обучения.
"""

from abc import ABC, abstractmethod
from typing import Any

from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedModel, ProcessorMixin, Trainer, TrainingArguments

from vlm_finetune.base.config import DEFAULT_LORA, DEFAULT_TRAINING
from vlm_finetune.base.image import ImageProcessor
from vlm_finetune.utils import set_logger

logger = set_logger(__name__)


class VlmModel(ABC):
    """
    Абстрактный базовый класс для мультимоделей Vision-Language Models (VLM),
    обеспечивающий интерфейс для дообучения и предсказаний.

    Атрибуты:
        model (PreTrainedModel): экземпляр предобученной модели VLM.
        processor (ProcessorMixin): процессор для токенизации текста и преобразования изображений.
        image_processor (ImageProcessor): препроцессор изображений для подготовки к модели.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        image_processor: ImageProcessor | None
    ) -> None:
        """
        Инициализация экземпляра базовой VLM-модели.

        Параметры:
            model: предобученная модель VLM.
            processor: процессор для токенизации текста и обработки изображений.
            image_processor: препроцессор изображений (если None, используется стандартный).
        """
        self.model = model
        self.processor = processor
        self.image_processor = image_processor or ImageProcessor()

    @abstractmethod
    def finetune(
        self,
        dataset: Dataset,
        output_dir: str,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        lora_params: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Дообучение модели на пользовательском датасете с использованием LoRA.

        Параметры:
            dataset: объект Dataset с изображениями и текстовыми примерами.
            output_dir: директория для сохранения результатов обучения.
            learning_rate: скорость обучения (по умолчанию 2e-4).
            num_train_epochs: количество эпох обучения (по умолчанию 3).
            lora_params: словарь параметров LoRA (по умолчанию DEFAULT_LORA).
            training_params: дополнительные параметры обучения (по умолчанию DEFAULT_TRAINING).

        Возвращает:
            None

        Исключения:
            ValueError: если датасет пуст или некорректен.
        """
        lora_params = lora_params or DEFAULT_LORA
        lora_config = LoraConfig(**lora_params)
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()

        training_params = training_params or DEFAULT_TRAINING
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            **training_params,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Используется устройство: %s", self.model.device)
        logger.info("Количество обучаемых параметров: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
        logger.info("Начинаем finetune...")

        trainer.train()
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        logger.info("Модель успешно дообучена и сохранена в %s", output_dir)
        self.model = trainer.model
        logger.info("!Для предсказаний будет использоваться дообученная модель!")

    @abstractmethod
    def predict(self, image: Image.Image | str, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Генерация текстового ответа по изображению и текстовому запросу.

        Параметры:
            image: объект PIL.Image или путь к файлу изображения.
            prompt: текстовый запрос пользователя.
            max_new_tokens: максимальное количество генерируемых токенов (по умолчанию 256).

        Возвращает:
            str: текстовый ответ модели.

        Исключения:
            RuntimeError: при ошибке генерации ответа.
        """
        if isinstance(image, str):
            image = self.image_processor.process_image(image_path=image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        model_response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        return model_response
