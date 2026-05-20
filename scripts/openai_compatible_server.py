#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for a local Transformers model."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ModelState:
    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        threads: int,
        default_max_tokens: int,
        max_context_length: int,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        torch.set_num_threads(threads)

        self.model_dir = model_dir
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()
        model_max_context = int(getattr(self.model.config, "max_position_embeddings", 0) or max_context_length)
        self.max_context_length = min(max_context_length, model_max_context)
        self.eos_token_ids = normalize_token_ids(
            getattr(self.model.generation_config, "eos_token_id", None)
            or getattr(self.model.config, "eos_token_id", None)
            or self.tokenizer.eos_token_id
        )
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else getattr(self.model.generation_config, "pad_token_id", None)
        )
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_ids[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "models/Qwen3-Coder-30B-A3B-Instruct-full"),
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", "qwen3-coder-30b-a3b-instruct-local"),
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_THREADS", "12")))
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=int(os.environ.get("DEFAULT_MAX_TOKENS", "512")),
        help="Default max generated tokens when the request omits max_tokens.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=int(os.environ.get("MAX_CONTEXT_LENGTH", "32768")),
        help="Server-side prompt + completion context cap.",
    )
    return parser.parse_args()


def openai_error(message: str, code: str = "bad_request", status: int = 400) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def normalize_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def chat_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generation_kwargs(payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    temperature = payload.get("temperature", 0)
    top_p = payload.get("top_p", 1.0)
    kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature and float(temperature) > 0:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
    else:
        kwargs["do_sample"] = False
    return kwargs


def requested_max_tokens(payload: dict[str, Any], default_max_tokens: int, remaining_context: int) -> int:
    explicit = payload.get("max_tokens")
    if explicit is None:
        explicit = payload.get("max_completion_tokens")
    if explicit is None:
        return min(default_max_tokens, remaining_context)
    return int(explicit)


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalOpenAICompat/0.1"
    protocol_version = "HTTP/1.1"
    state: ModelState

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "model": self.state.model_id})
            return
        if self.path in {"/v1/models", "/models"}:
            now = int(time.time())
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.state.model_id,
                            "object": "model",
                            "created": now,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self.send_json(*openai_error(f"Unknown route: {self.path}", "not_found", 404))

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json(*openai_error(f"Unknown route: {self.path}", "not_found", 404))
            return

        try:
            payload = self.read_json()
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                self.send_json(*openai_error("messages must be a non-empty list"))
                return

            if payload.get("stream", False):
                self.handle_stream(payload, messages)
            else:
                self.handle_completion(payload, messages)
        except Exception as exc:
            self.send_json(*openai_error(str(exc), "server_error", 500))

    def handle_completion(self, payload: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        prompt = chat_prompt(self.state.tokenizer, messages)
        inputs = self.state.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[-1])
        remaining_context = self.state.max_context_length - prompt_tokens
        if remaining_context <= 0:
            self.send_json(
                *openai_error(
                    f"Context length exceeded: prompt has {prompt_tokens} tokens, "
                    f"server max context is {self.state.max_context_length}.",
                    "context_length_exceeded",
                    400,
                )
            )
            return
        max_tokens = requested_max_tokens(payload, self.state.default_max_tokens, remaining_context)
        if max_tokens > remaining_context:
            self.send_json(
                *openai_error(
                    f"Requested max_tokens={max_tokens}, but only {remaining_context} tokens remain "
                    f"in the server context window ({self.state.max_context_length}).",
                    "context_length_exceeded",
                    400,
                )
            )
            return
        kwargs = generation_kwargs(payload, max_tokens)

        with self.state.lock, torch.inference_mode():
            output_ids = self.state.model.generate(
                **inputs,
                **kwargs,
                eos_token_id=self.state.eos_token_ids,
                pad_token_id=self.state.pad_token_id,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[-1] :]
        content = self.state.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        self.send_json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": payload.get("model") or self.state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(generated_ids.shape[-1]),
                    "total_tokens": int(output_ids.shape[-1]),
                },
            },
        )

    def handle_stream(self, payload: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        prompt = chat_prompt(self.state.tokenizer, messages)
        inputs = self.state.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[-1])
        remaining_context = self.state.max_context_length - prompt_tokens
        if remaining_context <= 0:
            self.send_json(
                *openai_error(
                    f"Context length exceeded: prompt has {prompt_tokens} tokens, "
                    f"server max context is {self.state.max_context_length}.",
                    "context_length_exceeded",
                    400,
                )
            )
            return
        max_tokens = requested_max_tokens(payload, self.state.default_max_tokens, remaining_context)
        if max_tokens > remaining_context:
            self.send_json(
                *openai_error(
                    f"Requested max_tokens={max_tokens}, but only {remaining_context} tokens remain "
                    f"in the server context window ({self.state.max_context_length}).",
                    "context_length_exceeded",
                    400,
                )
            )
            return
        streamer = TextIteratorStreamer(
            self.state.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = generation_kwargs(payload, max_tokens)

        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def run_generate() -> None:
            with self.state.lock, torch.inference_mode():
                self.state.model.generate(
                    **inputs,
                    **kwargs,
                    streamer=streamer,
                    eos_token_id=self.state.eos_token_ids,
                    pad_token_id=self.state.pad_token_id,
                )

        thread = Thread(target=run_generate, daemon=True)
        thread.start()

        for text in streamer:
            if not text:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.get("model") or self.state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload.get("model") or self.state.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        thread.join(timeout=1)
        try:
            self.connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    print(f"Loading model from {model_dir}", flush=True)
    Handler.state = ModelState(
        model_dir=model_dir,
        model_id=args.model_id,
        threads=args.threads,
        default_max_tokens=args.default_max_tokens,
        max_context_length=args.max_context_length,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenAI-compatible server listening on http://{args.host}:{args.port}", flush=True)
    print(
        f"Model id: {args.model_id}; max_context_length={Handler.state.max_context_length}; "
        f"default_max_tokens={Handler.state.default_max_tokens}; "
        f"eos_token_ids={Handler.state.eos_token_ids}; pad_token_id={Handler.state.pad_token_id}",
        flush=True,
    )
    print("Routes: GET /health, GET /v1/models, POST /v1/chat/completions", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
