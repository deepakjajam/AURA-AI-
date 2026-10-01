"""AURA local inference provider."""

from abc import ABC, abstractmethod
from functools import lru_cache
import threading

from app.core.config import settings


AURA_SYSTEM_PROMPT = """
You are AURA, an AI assistant running locally as part of the AURA AI project.

Identity:
- Your name is AURA.
- You are not Anthropic, Claude, OpenAI, ChatGPT, Google, Gemini, or Meta.
- If asked who you are or who created you, identify yourself as AURA.
- Never claim that Anthropic or another external AI company created you.

Behavior:
- Answer accurately and directly.
- Keep answers concise. Do not repeat sentences or restate the same point.
- If the user asks in Hindi, answer in Hindi.
- If the user asks in English, answer in English.
- For simple arithmetic, calculate correctly.
"""


class InferenceProvider(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str):
        raise NotImplementedError


class DemoProvider(InferenceProvider):
    """Local placeholder that needs no model files or ML dependencies."""

    @property
    def model_name(self) -> str:
        return 'aura-demo'

    def generate(self, prompt: str) -> str:
        return f'AURA demo reply: {prompt[:500]}'

    def generate_stream(self, prompt: str):
        yield self.generate(prompt)


def identity_response(prompt: str):
    q = prompt.strip().lower()

    identity_words = (
        "who are you",
        "who created you",
        "who made you",
        "are you anthropic",
        "are you claude",
        "are you chatgpt",
        "आप कौन हो",
        "तुम कौन हो",
        "आपको किसने बनाया",
        "तुम्हें किसने बनाया",
    )

    if any(word in q for word in identity_words):
        return (
            "I am AURA, an AI assistant running locally as part of "
            "the AURA AI project. I am not Anthropic or Claude."
        )

    return None


class HuggingFaceDevProvider(InferenceProvider):
    """Local Qwen development model."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        model_id = settings.dev_model_id

        print(f"[AURA] Loading local model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self._torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # This cached model is shared by all requests in the process. Serialize
        # generation so simultaneous chat requests cannot interfere with one another.
        self._generation_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return settings.dev_model_id

        print("[AURA] Local model loaded successfully.")

    def _inputs(self, prompt: str):
        messages = [
            {
                "role": "system",
                "content": AURA_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return {
            key: value.to("cpu")
            for key, value in inputs.items()
        }

    def _generation_options(self):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        return {
            "max_new_tokens": 64,
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            # Discourage token loops and repeated short phrases in greedy output.
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 3,
        }

    def generate(self, prompt: str) -> str:

        identity = identity_response(prompt)

        if identity:
            return identity

        inputs = self._inputs(prompt)

        input_length = inputs["input_ids"].shape[-1]

        with self._generation_lock, self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **self._generation_options(),
            )

        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return response or "I am AURA."


    def generate_stream(self, prompt: str):

        identity = identity_response(prompt)

        if identity:
            yield identity
            return

        from transformers import TextIteratorStreamer

        inputs = self._inputs(prompt)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            **self._generation_options(),
            "streamer": streamer,
        }

        def run_generation():
            try:
                with self._generation_lock, self._torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as exc:
                print(f"[AURA] Streaming generation failed: {exc}")
                streamer.on_finalized_text(
                    "Sorry, AURA could not complete that reply.",
                    stream_end=True,
                )

        thread = threading.Thread(
            target=run_generation,
            daemon=True,
        )

        thread.start()

        for text in streamer:
            if text:
                yield text


@lru_cache(maxsize=1)
def get_provider() -> InferenceProvider:
    backend = settings.inference_backend.strip().lower()
    if backend == 'demo':
        return DemoProvider()
    if backend == 'huggingface-dev':
        return HuggingFaceDevProvider()
    raise ValueError(f'Unsupported inference backend: {settings.inference_backend}')
