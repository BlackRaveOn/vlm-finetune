"""
Модуль `auto` предоставляет универсальный интерфейс для загрузки и регистрации
визуально-лингвистических моделей (Vision-Language Models, VLM) через библиотеку `transformers`.

Основные возможности:
    • Регистрация собственных подклассов моделей через декоратор `@register`.
    • Автоматическая загрузка процессора и модели по имени.
    • Поддержка 4-битной квантизации (BitsAndBytes) для экономии видеопамяти.
    • Единый интерфейс для различных VLM-архитектур (LLaVA, Qwen-VL, etc.).

Классы:
    AutoVlmModel — базовый класс-фабрика для создания и регистрации VLM-моделей.

Константы:
    DEVICE — устройство вычислений: "cuda" или "cpu".
    BNB_CONFIG — конфигурация BitsAndBytesConfig для 4-битной загрузки.

Исключения:
    OSError — если происходит попытка прямой инициализации базового класса.
    ValueError — если указан не зарегистрированный тип модели.
"""

from typing import Any, TypeVar

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)

from vlm_finetune.base import ImageProcessor

T = TypeVar("T", bound="AutoVlmModel")

DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BNB_CONFIG: BitsAndBytesConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


class AutoVlmModel:
    """
    Базовый класс для фабричной загрузки VLM-моделей.

    Этот класс нельзя инициализировать напрямую — модели создаются через
    `AutoVlmModel.from_name(...)`, который подбирает зарегистрированный подкласс
    по префиксу модели (например, `llava`, `qwen`, `blip`).

    Атрибуты класса:
        _registry: словарь сопоставления:
            <строковый префикс модели> -> <подкласс AutoVlmModel>

    Каждый зарегистрированный подкласс обязан принимать параметры:
        model: torch.nn.Module — загруженная модель
        processor: PreTrainedProcessor — процессор для токенизации и обработки изображений
        image_processor: ImageProcessor | None — внешний обработчик изображений (опционально)
    """

    _registry: dict[str, type["AutoVlmModel"]] = {}

    def __init__(self):
        """
        Базовый конструктор запрещён — используйте `from_name`.
        """
        raise OSError(
            "AutoVlmModel должен быть создан через `AutoVlmModel.from_name(name, ...)`, "
            "а не через прямой вызов конструктора."
        )

    @classmethod
    def register(cls, name: str):
        """
        Декоратор для регистрации подклассов модели.

        Параметры:
            name: str — имя модели, используемое в `AutoVlmModel.from_name(...)`.

        Пример:
            >>> @AutoVlmModel.register("llava")
            ... class LlavaModel(AutoVlmModel):
            ...     def __init__(self, model, processor, image_processor=None):
            ...         self.model = model
            ...         self.processor = processor
            ...         self.image_processor = image_processor
        """
        def decorator(subclass: type["AutoVlmModel"]) -> type["AutoVlmModel"]:
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(
        cls: type[T],
        model_name: str,
        model_path: str,
        model_dtype: torch.dtype = torch.float16,
        bnb_config: BitsAndBytesConfig = BNB_CONFIG,
        device: str = DEVICE,
        device_map: str | None = None,
        model_params: dict[str, Any] | None = None,
        processor_params: dict[str, Any] | None = None,
        image_processor: ImageProcessor | None = None
    ) -> T:
        """
        Фабрика для загрузки VLM-модели и процессора по имени и пути.

        Параметры:
            model_name: str — имя/префикс модели (например, "llava").
            model_path: str — путь/ID модели в HuggingFace Hub или локально.
            model_dtype: torch.dtype — dtype для загрузки (по умолчанию FP16).
            bnb_config: BitsAndBytesConfig | None — конфиг 4-битной загрузки.
            device: "cuda" | "cpu" — устройство для размещения модели.
            device_map: dict | str | None — карта устройств для accelerate (если надо).
            model_params: dict — дополнительные параметры в модель.
            processor_params: dict — параметры в AutoProcessor.
            image_processor: ImageProcessor | None — внешняя обработка изображений.

        Возвращает:
            Экземпляр зарегистрированного подкласса модели.

        Исключения:
            ValueError: если `model_name` не найден в `AutoVlmModel._registry`.
        """
        subclass = cls._registry.get(model_name)
        processor_params = processor_params or {}
        model_params = model_params or {}

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_path,
            use_fast=True,
            **processor_params
        )

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_path,
            dtype=model_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
            **model_params
        )
        model = model.to(device)

        return subclass(
            model=model,
            processor=processor,
            image_processor=image_processor
        )
