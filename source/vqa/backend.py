"""Async client for a vLLM OpenAI-compatible server (`vllm serve ...`).
Many requests are kept in flight so vLLM can batch them.
"""

import base64
import io

import httpx
from PIL import Image


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, enable_thinking: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Thinking is off for all models: with thinking on, Qwen3.5-style
        # models spend the token budget on reasoning before the answer.
        self.enable_thinking = enable_thinking
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, image: Image.Image | None, prompt: str, max_new_tokens: int = 128) -> str:
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        resp = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
