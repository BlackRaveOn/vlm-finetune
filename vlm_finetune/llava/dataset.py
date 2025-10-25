import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers.models.llava.processing_llava import LlavaProcessor


class LLavaDataset(Dataset):
    def __init__(self, dataset_path: list[dict[str, str]], processor: LlavaProcessor, prompt: str | None):
        with open(dataset_path, encoding="utf-8") as f:
            dataset_data = json.load(f)
        self.data = dataset_data
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image_path = item["image_path"]
        promt = item.get("prompt", self.prompt)
        answer = item.get("answer", self.prompt)

        image = Image.open(image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": promt},
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

        encoding: dict[str, torch.Tensor] = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=False
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)

        # 🔍 Находим позицию начала "ASSISTANT:" в токенах
        tokenized_assistant: torch.Tensor = self.processor.tokenizer(
            "ASSISTANT:",
            add_special_tokens=False
        )["input_ids"]

        # Ищем подстроку в input_ids
        start_idx = None
        for i in range(len(input_ids) - len(tokenized_assistant)):
            if torch.equal(input_ids[i:i+len(tokenized_assistant)], torch.tensor(tokenized_assistant)):
                start_idx = i + len(tokenized_assistant)
                break

        if start_idx is None:
            start_idx = 0  # fallback

        # 🎯 Маскируем всё до начала ответа
        labels = input_ids.clone()
        labels[:start_idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }