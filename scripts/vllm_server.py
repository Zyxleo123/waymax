from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Gemma-4 via vLLM in a separate process.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-4-31b-it",
        help="Hugging Face model id to load.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for the HTTP server.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="GPU ids visible to this server process, for example '1'.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of visible GPU memory to reserve for vLLM. Lower this for 2x48GB A6000 setups.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length. Lower this to reduce KV-cache memory usage.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences to serve at once.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture to reduce memory pressure.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size. Keep this at 1 when reserving a single GPU for vLLM.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Uvicorn log level.",
    )
    return parser.parse_args()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model.")
    image_base64: str | None = Field(default=None, description="Base64-encoded RGB/RGBA image.")
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, gt=0)
    image_placeholder: str | None = Field(
        default=None,
        description="Optional explicit multimodal placeholder to insert into the prompt. Defaults to <image> when an image is provided.",
    )


class GenerateResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    cuda_visible_devices: str | None


args = parse_args()

# Keep the vLLM worker isolated to a single GPU so another GPU can be reserved for JAX.
if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Running this file directly makes its own directory shadow the installed `vllm` package.
# Remove the script directory from sys.path so Python imports the real vLLM package.
if __name__ == "__main__":
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in os.sys.path:
        os.sys.path.remove(script_dir)

import torch
from vllm import LLM, SamplingParams

app = FastAPI(title="Waymax vLLM Gemma Server")
llm: LLM | None = None


def _decode_image(image_base64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(raw))
        return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc


@app.on_event("startup")
def _load_model() -> None:
    global llm
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, so vLLM cannot start the Gemma server.")

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=args.model_name,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    image = None
    if request.image_base64 is not None:
        image = _decode_image(request.image_base64)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    prompt = request.prompt
    if image is not None:
        placeholder = request.image_placeholder or "<image>"
        if placeholder not in prompt:
            prompt = f"{prompt}\n{placeholder}".strip()

    payload: dict[str, Any] = {"prompt": prompt}
    if image is not None:
        payload["multi_modal_data"] = {"image": image}

    outputs = llm.generate([payload], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text.strip()
    return GenerateResponse(text=text)


if __name__ == "__main__":
    uvicorn.run(
        "scripts.vllm_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
        factory=False,
    )
