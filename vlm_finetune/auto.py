"""
Модуль `auto` предоставляет универсальный интерфейс для загрузки и регистрации
визуально-лингвистических моделей (Vision-Language Models, VLM) через библиотеку `transformers`.

Основные возможности:
    • Регистрация собственных подклассов моделей.
    • Автоматическая загрузка процессора и модели по имени.
    • Поддержка 4-битной квантизации с помощью BitsAndBytes.

Классы:
    AutoVlmModel — базовый класс для создания и регистрации VLM-моделей.

Переменные:
    DEVICE — устройство вычислений ("cuda" или "cpu").
    BNB_CONFIG — конфигурация для 4-битной квантизации модели.

Исключения:
    OSError — при попытке прямой инициализации класса AutoVlmModel.
    ValueError — при указании неизвестного типа модели.
"""

from typing import Any, TypeVar

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

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
    Базовый класс для загрузки визуально-лингвистических моделей (VLM)
    и управления их регистрацией.

    Класс предоставляет фабричный метод `from_name` для автоматической инициализации
    нужного подкласса модели на основе её префикса (например, "llava:", "blip:" и т.п.).

    Атрибуты:
        _registry: словарь зарегистрированных подклассов моделей.
    """

    _registry: dict[str, type["AutoVlmModel"]] = {}

    def __init__(self):
        """
        Запрещает прямое создание экземпляров класса.

        Исключения:
            OSError: если выполняется попытка создать экземпляр напрямую.
        """
        raise OSError(
            "AutoVlmModel должен быть создан через `AutoVlmModel.from_name(name)`."
        )

    @classmethod
    def register(cls, name: str):
        """
        Декоратор для регистрации нового подкласса модели.

        Параметры:
            name: имя модели (префикс, например, "llava"),
                  используемое при вызове `from_name`.

        Возвращает:
            Декоратор, который добавляет подкласс модели в реестр.

        Пример:
            >>> @AutoVlmModel.register("llava")
            ... class LlavaModel(AutoVlmModel):
            ...     pass
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
    ) -> T:
        """
        Фабричный метод для загрузки модели и процессора по имени и пути.

        Параметры:
            model_name: имя модели (например, "llava").
            model_path: путь к предобученной модели.
            model_dtype: тип данных (по умолчанию float16).
            bnb_config: конфигурация BitsAndBytes для 4-битной загрузки.
            device: устройство для загрузки модели ("cuda" или "cpu").
            device_map: карта распределения модели по устройствам (опционально).
            model_params: дополнительные параметры при инициализации модели.
            processor_params: дополнительные параметры при инициализации процессора.

        Возвращает:
            Экземпляр зарегистрированного подкласса `AutoVlmModel`.

        Исключения:
            ValueError: если указанный префикс модели не зарегистрирован.
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

        if subclass is None:
            raise ValueError(f"Неизвестный тип VLM модели: {model_name}")

        return subclass(model=model, processor=processor)  # type: ignore
