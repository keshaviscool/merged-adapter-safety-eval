"""Batched response generation with configurable inference settings."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

from bench.config import GenerationConfig
from bench.datasets import Prompt
from bench.utils import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class Generation:
    prompt: Prompt
    response: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Generation":
        return cls(prompt=Prompt(**data["prompt"]), response=data["response"])


class ResponseGenerator:
    def __init__(self, model, tokenizer, config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.uses_chat_template = tokenizer.chat_template is not None

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    def _format_prompt(self, user_text: str) -> str:
        """Wraps the text in the model's chat template (or leaves it raw for base LMs)."""
        if not self.uses_chat_template:
            return user_text

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_text})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Generation settings
    # ------------------------------------------------------------------
    def _generation_kwargs(self) -> dict:
        cfg = self.config
        kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "repetition_penalty": cfg.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # Sampling parameters only make sense when sampling.
        if cfg.do_sample:
            kwargs["temperature"] = cfg.temperature
            kwargs["top_p"] = cfg.top_p
            kwargs["top_k"] = cfg.top_k
        return kwargs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_batch(self, formatted_prompts: list[str]) -> list[str]:
        # The chat template already adds special tokens such as BOS, so don't add them twice.
        encoded = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not self.uses_chat_template,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **self._generation_kwargs())

        # Keep only the newly generated tokens (drop the prompt).
        new_token_ids = output_ids[:, encoded["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    def generate(self, prompts: list[Prompt]) -> list[Generation]:
        set_global_seed(self.config.seed)

        # Sort by length so each batch has similar-sized prompts (less padding).
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].text))
        responses: list[str | None] = [None] * len(prompts)

        batch_size = self.config.batch_size
        for start in tqdm(range(0, len(order), batch_size), desc="Generating"):
            batch_indices = order[start : start + batch_size]
            formatted = [self._format_prompt(prompts[i].text) for i in batch_indices]
            batch_responses = self._generate_batch(formatted)
            for index, response in zip(batch_indices, batch_responses):
                responses[index] = response

        # Restore the original dataset order.
        return [Generation(prompt=p, response=r) for p, r in zip(prompts, responses)]
