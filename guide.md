# Running Guide

This guide explains how to prepare the local model files, verify the
`safetensors` checkpoint, run the CPU-only OpenAI-compatible eLLM server, and
run the eLLM Rust BF16 CPU test path. The target runtime in this repository is
CPU-only eLLM execution; GPU offload is not used.

## 1. Requirements

| Requirement | Notes |
| --- | --- |
| OS | Linux x86_64 |
| CPU | AVX-512 BF16 is required for the Rust BF16 path |
| GPU | Not required and not used |
| RAM | 128 GiB was used for the reported Qwen3-Coder-30B-A3B BF16 run |
| Python | A virtual environment is recommended |
| Rust | Nightly toolchain, configured by `rust-toolchain.toml` |

Install Python dependencies in a local virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors accelerate huggingface_hub
```

If your virtual environment does not include `pip`, use `uv`:

```bash
uv pip install --python .venv/bin/python \
  torch transformers safetensors accelerate huggingface_hub
```

Use the PyTorch install command that matches your platform if you need a
specific CPU wheel. Do not install or select a CUDA wheel for these documented
runs.

## 2. Prepare a Model

The full model weights are not stored in this repository.

For the BF16 Qwen3-Coder test path, put the Hugging Face `safetensors` files
under this path:

```text
models/Qwen3-Coder-30B-A3B-Instruct-full/
```

The directory should contain files like:

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00016.safetensors
...
```

For the AWQ 4-bit Qwen3.6 compatibility target, download it to a local ignored
directory:

```bash
hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --local-dir models/Qwen3.6-35B-A3B-AWQ-4bit
```

The AWQ model uses `compressed-tensors` quantization. The current eLLM CPU path
includes an experimental reference executor for this mixed Qwen3.6/Qwen3.5 MoE
architecture. It computes token ids on CPU only, but it is intentionally
conservative and not optimized; the current loader dequantizes AWQ expert
weights to BF16 in host memory.

## 3. Verify Safetensors Loading

Run the Python verification script first for the BF16 Qwen3-Coder baseline. It
checks shard metadata and generates a short answer on the host CPU. This is a
CPU-only helper validation step, not the benchmark source and not the Qwen3.6
AWQ eLLM runtime path.

```bash
.venv/bin/python scripts/verify_safetensors_generation.py \
  --model-dir models/Qwen3-Coder-30B-A3B-Instruct-full \
  --threads 12 \
  --max-new-tokens 48
```

Expected output includes:

```text
=== safetensors ===
=== load ===
=== generated ===
generated_tokens: ...
```

## 4. Start the CPU-Only OpenAI-Compatible eLLM Server

Start the local chat server with model name `local_model`, API key `EMPTY`, host
`0.0.0.0`, and port `8000`. This server mode loads the Qwen3.6 AWQ model through
the Rust eLLM CPU reference executor. GPU offload is not used.

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin main

CUDA_VISIBLE_DEVICES= \
ELLM_OPENAI_SERVER=1 \
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_MODEL_ID=local_model \
ELLM_API_KEY=EMPTY \
ELLM_HOST=0.0.0.0 \
ELLM_PORT=8000 \
ELLM_MAX_CONTEXT=2048 \
ELLM_DEFAULT_MAX_TOKENS=128 \
./target/release/main
```

`ELLM_MAX_CONTEXT` controls the server-side prompt + completion window allocated
inside the eLLM reference executor. Larger values use more host memory. This path
is correctness-oriented and slow; it is not the final optimized AWQ kernel path.

Health check:

```bash
curl http://localhost:8000/health
```

List models:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

Chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [
      {"role": "user", "content": "Write a short Python example."}
    ],
    "max_tokens": 128
  }'
```

Streaming chat completion:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [
      {"role": "user", "content": "Explain BF16 in one paragraph."}
    ],
    "max_tokens": 128,
    "stream": true
}'
```

For a minimal CPU smoke test that avoids a long chat-template prefill, the server
also accepts a local eLLM extension field, `prompt_token_ids`. This is not part
of the OpenAI API; it is only useful for checking the loaded eLLM model path:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [{"role": "user", "content": "smoke"}],
    "prompt_token_ids": [0],
    "max_tokens": 1
  }'
```

On the documented test system, this server smoke loaded 733 tensors, reported
`runtime=ellm-qwen35-cpu-reference` from `/health`, returned model id
`local_model` from `/v1/models`, and handled a normal `messages` request with
`prompt_tokens=9`. The short `prompt_token_ids=[0]` smoke generated content
` sanz`, and streaming mode ended with `data: [DONE]`.

## 5. Run the eLLM Rust BF16 CPU Path

The Rust binary is the eLLM CPU execution path. It loads the config and, when
`ELLM_SAFETENSORS_DIR` is set, loads BF16 weights from the local safetensors
directory. It does not use a GPU.

First check that it compiles:

```bash
RUSTFLAGS='-C target-cpu=native' cargo check --bin main
```

Run a short BF16 test:

```bash
ELLM_CONFIG=models/Qwen3-Coder-30B-A3B-Instruct-full/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3-Coder-30B-A3B-Instruct-full \
ELLM_GENERATE_TOKENS=1 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --release --bin main
```

The binary currently prints generated token ids rather than a decoded text
response.

## 6. Run the Qwen3.6 AWQ CPU Reference Path

Run the same Rust entry point with the Qwen3.6 AWQ config. The mixed
`linear_attention`/`full_attention` config is detected automatically and routed
to the CPU reference executor:

```bash
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_GENERATE_TOKENS=1 \
ELLM_PROMPT='Write one short sentence about BF16.' \
ELLM_MAX_CONTEXT=64 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --release --bin main
```

`ELLM_PROMPT` is encoded with `tokenizer.json` from `ELLM_SAFETENSORS_DIR` and is
wrapped in a minimal Qwen chat prompt by default. Set `ELLM_RAW_PROMPT=1` to
encode the prompt exactly as provided. `ELLM_PROMPT_IDS` is still available for
comma-separated token ids and takes precedence over `ELLM_PROMPT`.

The binary prints generated token ids and, when a tokenizer is available, decoded
generated text. Expect high memory use and slow throughput in this path because
it is a correctness-oriented CPU reference path, not the final optimized AWQ
kernel path. EOS ids are loaded from `generation_config.json` when available, so
Qwen `<|im_end|>` can stop generation correctly.

On the documented test system, the command above completed without GPU offload
with `ELLM_PROMPT_IDS=0`, `ELLM_GENERATE_TOKENS=1`, and `ELLM_MAX_CONTEXT=2` in
55.54 seconds, generated token id `222033`, decoded it as ` sanz`, and reached
83,363,796 KB peak RSS.

To test the AWQ unpacking logic itself:

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib model_loader
```

To test the Qwen3.6/Qwen3.5 CPU reference math on small synthetic tensors:

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib qwen3_moe::reference_cpu
```

---

# 실행 가이드

이 문서는 로컬 모델 파일 준비, `safetensors` 검증, CPU 전용 OpenAI 호환 eLLM
서버 실행, Rust BF16 테스트 실행 방법을 정리합니다. 이 저장소의 기준 실행
대상은 eLLM CPU 실행이며, GPU offload는 사용하지 않습니다.

## 1. 요구 사항

| 요구 사항 | 설명 |
| --- | --- |
| OS | Linux x86_64 |
| CPU | Rust BF16 경로 실행에는 AVX-512 BF16 필요 |
| GPU | 필요하지 않으며 사용하지 않음 |
| RAM | 보고된 Qwen3-Coder-30B-A3B BF16 실행은 128 GiB 환경에서 테스트 |
| Python | 가상 환경 사용 권장 |
| Rust | `rust-toolchain.toml`에서 nightly 사용 |

Python 의존성은 로컬 가상 환경에 설치하는 것을 권장합니다.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch transformers safetensors accelerate huggingface_hub
```

가상 환경에 `pip`가 없다면 `uv`를 사용할 수 있습니다.

```bash
uv pip install --python .venv/bin/python \
  torch transformers safetensors accelerate huggingface_hub
```

CPU 전용 wheel이 따로 필요하면 본인 환경에 맞는 PyTorch 설치 명령을 사용하면
됩니다. 이 문서의 실행 경로에서는 CUDA wheel을 설치하거나 선택하지 않습니다.

## 2. 모델 준비

전체 모델 가중치는 이 저장소에 포함되어 있지 않습니다.

BF16 Qwen3-Coder 테스트 경로는 Hugging Face `safetensors` 파일을 아래 경로에
배치합니다.

```text
models/Qwen3-Coder-30B-A3B-Instruct-full/
```

디렉터리에는 다음과 같은 파일들이 있어야 합니다.

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00016.safetensors
...
```

AWQ 4-bit Qwen3.6 호환성 점검 대상은 Git에서 제외되는 로컬 디렉터리에 먼저
내려받을 수 있습니다.

```bash
hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --local-dir models/Qwen3.6-35B-A3B-AWQ-4bit
```

이 AWQ 모델은 `compressed-tensors` quantization을 사용합니다. 현재 eLLM CPU
경로에는 이 mixed Qwen3.6/Qwen3.5 MoE 구조를 위한 실험적 reference executor가
추가되어 있습니다. CPU만 사용해 token id를 생성하지만, 아직 최적화된 경로는
아닙니다. 현재 loader는 AWQ expert weight를 host memory에서 BF16으로
dequantize합니다.

## 3. Safetensors 로딩 검증

먼저 BF16 Qwen3-Coder 기준선에 대해 Python 검증 스크립트를 실행합니다. 이
스크립트는 shard 메타데이터를 확인하고 host CPU에서 짧은 답변을 생성합니다.
이는 CPU 전용 보조 검증 단계이며, 성능 측정 기준이나 Qwen3.6 AWQ의 eLLM 실행
경로가 아닙니다.

```bash
.venv/bin/python scripts/verify_safetensors_generation.py \
  --model-dir models/Qwen3-Coder-30B-A3B-Instruct-full \
  --threads 12 \
  --max-new-tokens 48
```

정상 실행 시 다음과 같은 구간이 출력됩니다.

```text
=== safetensors ===
=== load ===
=== generated ===
generated_tokens: ...
```

## 4. CPU 전용 OpenAI 호환 eLLM 서버 실행

모델 이름은 `local_model`, API key는 `EMPTY`, host는 `0.0.0.0`, port는 `8000`으로
실행합니다. 이 서버 모드는 Rust eLLM CPU reference executor로 Qwen3.6 AWQ
모델을 로드합니다. GPU offload는 사용하지 않습니다.

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin main

CUDA_VISIBLE_DEVICES= \
ELLM_OPENAI_SERVER=1 \
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_MODEL_ID=local_model \
ELLM_API_KEY=EMPTY \
ELLM_HOST=0.0.0.0 \
ELLM_PORT=8000 \
ELLM_MAX_CONTEXT=2048 \
ELLM_DEFAULT_MAX_TOKENS=128 \
./target/release/main
```

`ELLM_MAX_CONTEXT`는 eLLM reference executor 내부에 할당할 prompt + completion
context window를 정합니다. 값을 키우면 host memory 사용량도 늘어납니다. 이
경로는 정확성 확인을 위한 reference 경로라 느리며, 최종 최적화 AWQ kernel
경로는 아닙니다.

Health check:

```bash
curl http://localhost:8000/health
```

모델 목록 확인:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

Chat completion 요청:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [
      {"role": "user", "content": "짧은 Python 예제를 작성해줘."}
    ],
    "max_tokens": 128
  }'
```

Streaming 요청:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [
      {"role": "user", "content": "BF16을 한 문단으로 설명해줘."}
    ],
    "max_tokens": 128,
    "stream": true
}'
```

긴 chat template prefill을 피해서 CPU 경로만 짧게 확인하려면 로컬 eLLM 확장
필드인 `prompt_token_ids`도 사용할 수 있습니다. 이 필드는 OpenAI API 표준이
아니며, 로드된 eLLM 모델 경로를 확인하기 위한 용도입니다.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [{"role": "user", "content": "smoke"}],
    "prompt_token_ids": [0],
    "max_tokens": 1
  }'
```

문서화한 테스트 시스템에서는 이 서버 smoke run에서 733개 tensor를 로드했고,
`/health`는 `runtime=ellm-qwen35-cpu-reference`를 반환했으며, `/v1/models`는
model id `local_model`을 반환했습니다. 일반 `messages` 요청은
`prompt_tokens=9`로 처리됐습니다. 짧은 `prompt_token_ids=[0]` smoke의 생성
content는 ` sanz`였고, streaming 모드는 마지막에 `data: [DONE]`으로
종료됐습니다.

## 5. eLLM Rust BF16 CPU 경로 실행

Rust 바이너리는 eLLM CPU 실행 경로입니다. `ELLM_SAFETENSORS_DIR`를 지정하면
로컬 safetensors 디렉터리에서 BF16 가중치를 로드합니다. GPU는 사용하지
않습니다.

먼저 컴파일을 확인합니다.

```bash
RUSTFLAGS='-C target-cpu=native' cargo check --bin main
```

짧은 BF16 테스트 실행:

```bash
ELLM_CONFIG=models/Qwen3-Coder-30B-A3B-Instruct-full/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3-Coder-30B-A3B-Instruct-full \
ELLM_GENERATE_TOKENS=1 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --release --bin main
```

현재 Rust 바이너리는 디코딩된 텍스트가 아니라 생성된 token id를 출력합니다.

## 6. Qwen3.6 AWQ CPU reference 경로 실행

Qwen3.6 AWQ config로 같은 Rust entry point를 실행합니다. mixed
`linear_attention`/`full_attention` config는 자동으로 감지되어 CPU reference
executor로 들어갑니다.

```bash
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_GENERATE_TOKENS=1 \
ELLM_PROMPT='BF16에 대해 한 문장으로 설명해줘.' \
ELLM_MAX_CONTEXT=64 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --release --bin main
```

`ELLM_PROMPT`는 `ELLM_SAFETENSORS_DIR`의 `tokenizer.json`으로 encode되며,
기본적으로 간단한 Qwen chat prompt 형식으로 감쌉니다. 입력 문자열을 그대로
encode하려면 `ELLM_RAW_PROMPT=1`을 설정합니다. 쉼표로 구분한 token id를 직접
넣는 `ELLM_PROMPT_IDS`도 계속 사용할 수 있고, `ELLM_PROMPT`보다 우선합니다.

바이너리는 생성된 token id와, tokenizer가 있으면 디코딩된 생성 텍스트를 함께
출력합니다. 이 경로는 최종 최적화 AWQ kernel 경로가 아니라 정확성 확인을 위한
CPU reference 경로이므로 메모리 사용량이 크고 처리 속도도 느릴 수 있습니다.
EOS id는 가능한 경우 `generation_config.json`에서 읽으므로 Qwen `<|im_end|>`도
정상 종료 토큰으로 처리됩니다.

문서화한 테스트 시스템에서는 `ELLM_PROMPT_IDS=0`, `ELLM_GENERATE_TOKENS=1`,
`ELLM_MAX_CONTEXT=2`로 실행한 smoke run이 GPU offload 없이 55.54초에 완료됐고,
token id `222033`을 생성해 ` sanz`로 decode했으며 peak RSS는 83,363,796
KB였습니다.

AWQ unpack 로직 자체는 아래 명령으로 테스트할 수 있습니다.

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib model_loader
```

Qwen3.6/Qwen3.5 CPU reference 연산 자체는 작은 synthetic tensor로 아래처럼
검증할 수 있습니다.

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib qwen3_moe::reference_cpu
```
