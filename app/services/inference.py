"""AURA AI inference providers: offline local model + AURA online API."""

from abc import ABC, abstractmethod
from functools import lru_cache
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from app.core.config import settings


AURA_SYSTEM_PROMPT = """
You are AURA, an AI assistant created as part of the AURA AI project.

Identity:
- Your name is AURA.
- Do not claim to be Anthropic, Claude, OpenAI, ChatGPT, Google, Gemini, or Meta.
- If asked who you are, identify yourself as AURA.

Behavior:
- Answer accurately and directly.
- If the user asks in Hindi, answer in Hindi.
- If the user asks in English, answer in English.
- Keep answers concise and useful.
- Do not invent facts.
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
            "I am AURA, an AI assistant created as part of the AURA AI project. "
            "I am not Anthropic, Claude, OpenAI, or ChatGPT."
        )

    return None


class DemoProvider(InferenceProvider):

    @property
    def model_name(self) -> str:
        return "aura-demo"

    def generate(self, prompt: str) -> str:
        return f"AURA demo reply: {prompt[:500]}"


class LocalQwenProvider(InferenceProvider):
    """Offline local Qwen model."""

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
        )

        self.model.to("cpu")
        self.model.eval()

        self._generation_lock = threading.Lock()

        print("[AURA] Local model loaded successfully.")

    @property
    def model_name(self) -> str:
        return settings.dev_model_id

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
                print(f"[AURA] Local streaming failed: {exc}")
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


class AuraOnlineProvider(InferenceProvider):
    """Client for the future AURA Online API.

    The API is intentionally external to this application. No OpenAI
    dependency is used here.
    """

    @property
    def model_name(self) -> str:
        return settings.online_model_id

    def generate(self, prompt: str) -> str:
        identity = identity_response(prompt)

        if identity:
            return identity

        api_url = settings.aura_api_url.strip()

        if not api_url:
            raise RuntimeError(
                "AURA Online API is not configured yet. "
                "Set AURA_API_URL when the online AURA API is ready."
            )

        payload = json.dumps({
            "message": prompt,
            "system": AURA_SYSTEM_PROMPT.strip(),
        }).encode("utf-8")

        request = Request(
            api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"AURA Online API request failed: {exc}") from exc

        data = json.loads(raw)

        reply = data.get("reply") or data.get("response") or data.get("text")

        if not reply:
            raise RuntimeError("AURA Online API returned no reply.")

        return str(reply)

    def generate_stream(self, prompt: str):
        yield self.generate(prompt)


def needs_online(prompt: str) -> bool:
    """Detect queries where current/network information may be useful."""

    q = prompt.strip().lower()

    online_terms = (
        "latest",
        "current",
        "today",
        "tonight",
        "tomorrow",
        "yesterday",
        "news",
        "live",
        "weather",
        "price",
        "stock price",
        "share price",
        "market today",
        "right now",
        "recent",
        "अभी",
        "आज",
        "कल",
        "ताज़ा",
        "ताजा",
        "न्यूज़",
        "समाचार",
        "मौसम",
        "कीमत",
        "भाव",
        "वर्तमान",
    )

    return any(term in q for term in online_terms)


class HybridProvider(InferenceProvider):
    """AURA router: OFFLINE, ONLINE, or AUTO."""

    def __init__(self):
        self.local = LocalQwenProvider()
        self.online = AuraOnlineProvider()

    @property
    def model_name(self) -> str:
        return "aura-hybrid"

    def _mode(self) -> str:
        return settings.aura_mode.strip().lower()

    def _select(self, prompt: str):
        mode = self._mode()

        if mode == "offline":
            return self.local

        if mode == "online":
            return self.online

        if mode == "auto":
            if needs_online(prompt) and settings.aura_api_url.strip():
                return self.online
            return self.local

        raise ValueError(
            f"Unsupported AURA mode: {settings.aura_mode}"
        )

    def generate(self, prompt: str) -> str:
        identity = identity_response(prompt)

        if identity:
            return identity

        provider = self._select(prompt)

        print(
            f"[AURA] mode={self._mode()} provider={provider.model_name}"
        )

        return provider.generate(prompt)

    def generate_stream(self, prompt: str):
        identity = identity_response(prompt)

        if identity:
            yield identity
            return

        provider = self._select(prompt)

        print(
            f"[AURA] mode={self._mode()} provider={provider.model_name}"
        )

        yield from provider.generate_stream(prompt)


@lru_cache(maxsize=1)
def get_provider() -> InferenceProvider:
    backend = settings.inference_backend.strip().lower()

    if backend == "demo":
        return DemoProvider()

    if backend == "huggingface-dev":
        return LocalQwenProvider()

    if backend == "hybrid":
        return HybridProvider()

    raise ValueError(
        f"Unsupported inference backend: {settings.inference_backend}"
    )
