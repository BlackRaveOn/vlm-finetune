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



T = TypeVar("T", bound="AutoVlmModel")


class AutoVlmModel:
    _registry: dict[str, type["AutoVlmModel"]] = {}

    def __init__(self):
        raise OSError(
            "AutoVlmModel is designed to be instantiated via `AutoVlmModel.from_name(name)`."
        )

    @classmethod
    def register(cls, name: str):
        """Decorator to register subclasses"""
        def decorator(subclass: type["AutoVlmModel"]) -> type["AutoVlmModel"]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def from_name(
        cls: type[T], 
        model_name: str,            
        model_path: str,   
        model_dtype=torch.float16,
        bnb_config: BitsAndBytesConfig = BNB_CONFIG, 
        device: str = DEVICE, 
        device_map: str | None = None,
        model_params: dict[str, Any] = None,
        processor_params: dict[str, Any] = None
    ) -> T:
        """Factory: choose proper subclass"""
        prefix = model_name.split(":")[0].lower()
        subclass = cls._registry.get(prefix)
        processor_params = processor_params or {}
        model_params = model_params or {}
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=True, **processor_params)
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_path,
            dtype=model_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
            **model_params
        )
        model = model.to(device)
        if subclass is None:
            raise ValueError(f"Unknown VLM model type: {prefix}")
        return subclass(model=model, processor=processor)  # type: ignore
