from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava.processing_llava import LlavaProcessor

from vlm_finetune import AutoVlmModel
from vlm_finetune.llava.dataset import LLavaDataset

DEFAULT_LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


DEFAULT_TRAINING = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Пробуем bf16 если доступно
    "logging_steps": 10,
    "save_steps": 100,
    "remove_unused_columns": False,
    "push_to_hub": False,
    "dataloader_pin_memory": False,
    "gradient_checkpointing": False,
    "dataloader_drop_last": True,
}


@AutoVlmModel.register("llava")
class LLavaModel:

    def __init__(
            self,
            model: LlavaModel,
            processor: LlavaProcessor
        ):
        self.model = model
        self.processor = processor

    def finetune(
            self, 
            dataset_path: list[dict[str, str]], 
            output_dir: str, 
            learning_rate=2e-4, 
            num_train_epochs=3, 
            promt: str | None = None, 
            lora_params: dict[str, Any] | None = None, 
            training_params: dict[str, Any] | None = None
        ):
        dataset = LLavaDataset(dataset_path=dataset_path, processor=self.processor, promt=promt)

        lora_params = lora_params or DEFAULT_LORA
        lora_config = LoraConfig(**lora_params)
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()

        training_params = training_params or DEFAULT_TRAINING
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            **training_params
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        print(f"Используется устройство: {self.model.device}")
        print(f"Количество обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print("Начинаем обучение...")
        trainer.train()
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        self.model = trainer.model

    def predict(self, image, promt, max_new_tokens: int = 256):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": promt},
                ],
            },
        ] 
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        model_answer = response[0].split("ASSISTANT:")[1].strip()
        return model_answer
