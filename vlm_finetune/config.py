"""
Модуль `config` содержит стандартные конфигурации для обучения и дообучения моделей VLM.

Переменные:
    DEFAULT_LORA — параметры адаптации LoRA.
    DEFAULT_TRAINING — параметры обучения для Trainer.
"""

import torch

DEFAULT_LORA: dict[str, object] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

DEFAULT_TRAINING: dict[str, object] = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "logging_steps": 10,
    "save_steps": 100,
    "remove_unused_columns": False,
    "push_to_hub": False,
    "dataloader_pin_memory": False,
    "gradient_checkpointing": False,
    "dataloader_drop_last": True,
}
