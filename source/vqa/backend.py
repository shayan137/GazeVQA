"""Answer clients.

VLLMClient talks to a vLLM OpenAI-compatible server (`vllm serve ...`) with many requests in
flight so vLLM can batch them. HFLocalClient runs a Hugging Face model in-process, one request
at a time, for smoke tests on a machine without vLLM. Both expose

    await client.generate_ex(prompt, image=None, video_path=None, max_new_tokens=...) -> Generation
    await client.generate(image, prompt, max_new_tokens=...) -> str        (text only)

A Generation also carries the token counts, so the logs show whether an image or a whole video
actually reached the model.
"""

import asyncio
import base64
import re
import io
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    extra: dict = field(default_factory=dict)  # backend specific details worth logging


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def video_to_url(path: str | Path, transport: str = "data") -> str:
    """'data': the mp4 is embedded in the request; 'file': the server reads the path itself and
    must run with --allowed-local-media-path covering it."""
    path = Path(path)
    if transport == "file":
        return f"file://{path.resolve()}"
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_TAIL = re.compile(r"<\|(?:return|end|call|start)\|>.*$", re.S)


def parse_harmony(raw: str) -> tuple[str, bool]:
    """Final-channel text of a harmony-format reply (GPT-OSS based models, e.g. InternVL3.5-GPT-OSS).
    The reply looks like "<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|return|>";
    returns (ANSWER, True). Without a final channel (cut off while reasoning) it returns the text with the
    special tokens removed and False, so the row is still logged and can be told apart."""
    if HARMONY_FINAL in raw:
        return HARMONY_TAIL.sub("", raw.rsplit(HARMONY_FINAL, 1)[1]).strip(), True
    return re.sub(r"<\|[a-z_]+\|>", " ", raw).strip(), False


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0, enable_thinking: bool = False,
                 video_transport: str = "data"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Thinking is off for all models: with thinking on, Qwen3.5-style
        # models spend the token budget on reasoning before the answer.
        self.enable_thinking = enable_thinking
        self.video_transport = video_transport
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate_ex(self, prompt: str, image: Image.Image | None = None, video_path=None,
                          max_new_tokens: int = 128, image_path=None) -> Generation:
        content = [{"type": "text", "text": prompt}]
        if video_path is not None:
            content.insert(0, {"type": "video_url", "video_url": {"url": video_to_url(video_path, self.video_transport)}})
        elif image_path is not None:  # a pre-rendered JPEG is sent as it is, without re-encoding
            url = "data:image/jpeg;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})
        elif image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        harmony = "gpt-oss" in self.model.lower()
        if harmony:
            # keep the channel markers (otherwise the reply starts with a bare "final") and cut the answer out ourselves
            payload["skip_special_tokens"] = False
        resp = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage") or {}
        text = body["choices"][0]["message"]["content"]
        extra = {"finish_reason": body["choices"][0].get("finish_reason")}
        if harmony:
            extra["raw_text"] = text
            text, extra["harmony_final"] = parse_harmony(text)
        return Generation(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            extra=extra,
        )

    async def generate(self, image: Image.Image | None, prompt: str, max_new_tokens: int = 128) -> str:
        return (await self.generate_ex(prompt, image=image, max_new_tokens=max_new_tokens)).text

    async def close(self) -> None:
        await self._client.aclose()


class HFLocalClient:
    """In-process Qwen3-VL / Qwen3.5 style model. Videos are decoded with OpenCV and handed to the
    processor with their metadata (fps, frame indices), so the "<t seconds>" timestamps in the
    prompt follow the container fps and every written frame is kept (no resampling)."""

    def __init__(self, model_path: str, name: str, device: str = "cuda", enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = name
        self.enable_thinking = enable_thinking
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.net = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
        self._lock = asyncio.Lock()  # one generation at a time on the one GPU

    @staticmethod
    def _read_video(path):
        import cv2
        import numpy as np
        from transformers.video_utils import VideoMetadata

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        meta = VideoMetadata(total_num_frames=len(frames), fps=fps, duration=len(frames) / fps,
                             frames_indices=list(range(len(frames))))
        return np.stack(frames), meta

    def _generate(self, prompt, image, video_path, max_new_tokens) -> Generation:
        import torch

        part = {"type": "video"} if video_path is not None else ({"type": "image"} if image is not None else None)
        content = ([part] if part else []) + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        kwargs = {}
        if video_path is not None:
            frames, meta = self._read_video(video_path)
            kwargs = {"videos": [frames], "video_metadata": [meta], "videos_kwargs": {"do_sample_frames": False}}
        elif image is not None:
            kwargs = {"images": [image.convert("RGB")]}
        inputs = self.processor(text=[text], return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            out = self.net.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        extra = {}
        if "video_grid_thw" in inputs:
            extra["video_grid_thw"] = inputs["video_grid_thw"][0].tolist()
        return Generation(self.processor.decode(new, skip_special_tokens=True), int(inputs["input_ids"].shape[1]),
                          int(new.shape[0]), extra)

    async def generate_ex(self, prompt: str, image: Image.Image | None = None, video_path=None,
                          max_new_tokens: int = 128, image_path=None) -> Generation:
        if image is None and image_path is not None:
            image = Image.open(image_path)
        async with self._lock:
            return await asyncio.to_thread(self._generate, prompt, image, video_path, max_new_tokens)

    async def generate(self, image: Image.Image | None, prompt: str, max_new_tokens: int = 128) -> str:
        return (await self.generate_ex(prompt, image=image, max_new_tokens=max_new_tokens)).text

    async def close(self) -> None:
        pass
