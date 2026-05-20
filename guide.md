# Running Guide

This guide explains how to prepare the local model files, verify the
`safetensors` checkpoint, run the CPU-only OpenAI-compatible wrapper, and run the
eLLM Rust BF16 CPU test path. The target runtime in this repository is CPU-only
eLLM execution; GPU offload is not used.

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
can inspect the config and unpack compressed expert weights, but it cannot run
generation yet because the model requires GatedDeltaNet `linear_attention` to be
integrated into the eLLM CPU executor.

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

## 4. Start the CPU-Only OpenAI-Compatible Wrapper

Start the local chat server with model name `local_model`, API key `EMPTY`, host
`0.0.0.0`, and port `8000`. The wrapper explicitly hides CUDA devices and loads
the model on CPU. The benchmark target remains the eLLM Rust CPU path in the next
section.

```bash
.venv/bin/python scripts/openai_compatible_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-dir models/Qwen3-Coder-30B-A3B-Instruct-full \
  --model-id local_model \
  --threads 12 \
  --default-max-tokens 512 \
  --max-context-length 32768
```

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

## 6. Check the Qwen3.6 AWQ CPU Status

Run the same Rust entry point with the Qwen3.6 AWQ config to verify the current
CPU/eLLM compatibility boundary:

```bash
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_GENERATE_TOKENS=1 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --bin main
```

Expected result today: the program exits before loading all weights and reports
that 30 `linear_attention` layers require the GatedDeltaNet CPU path to be wired
into the executor. This is intentional; Qwen3.6 AWQ should be considered
unsupported in eLLM CPU generation until that integration is complete.

To test the AWQ unpacking logic itself:

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib model_loader
```

---

# 실행 가이드

이 문서는 로컬 모델 파일 준비, `safetensors` 검증, CPU 전용 OpenAI 호환 wrapper
실행, Rust BF16 테스트 실행 방법을 정리합니다. 이 저장소의 기준 실행 대상은
eLLM CPU 실행이며, GPU offload는 사용하지 않습니다.

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
경로는 config 점검과 압축된 expert weight unpack까지는 가능하지만,
GatedDeltaNet `linear_attention` 경로가 아직 eLLM CPU 실행기에 통합되지 않아
실제 생성 실행은 불가능합니다.

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

## 4. CPU 전용 OpenAI 호환 wrapper 실행

모델 이름은 `local_model`, API key는 `EMPTY`, host는 `0.0.0.0`, port는 `8000`으로
실행합니다. 이 wrapper는 CUDA 장치를 숨기고 모델을 CPU에 로드합니다. 성능 측정
기준은 다음 섹션의 eLLM Rust CPU 경로입니다.

```bash
.venv/bin/python scripts/openai_compatible_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-dir models/Qwen3-Coder-30B-A3B-Instruct-full \
  --model-id local_model \
  --threads 12 \
  --default-max-tokens 512 \
  --max-context-length 32768
```

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

## 6. Qwen3.6 AWQ CPU 상태 확인

Qwen3.6 AWQ config로 같은 Rust entry point를 실행하면 현재 eLLM CPU 호환성
경계를 확인할 수 있습니다.

```bash
ELLM_CONFIG=models/Qwen3.6-35B-A3B-AWQ-4bit/config.json \
ELLM_SAFETENSORS_DIR=models/Qwen3.6-35B-A3B-AWQ-4bit \
ELLM_GENERATE_TOKENS=1 \
RUSTFLAGS='-C target-cpu=native' \
cargo run --bin main
```

현재 예상 결과는 전체 weight를 로드하기 전에 종료하면서, 30개의
`linear_attention` layer에 GatedDeltaNet CPU 경로 통합이 필요하다고 출력하는
것입니다. 이는 의도한 동작이며, 해당 통합이 완료되기 전까지 Qwen3.6 AWQ는 eLLM
CPU 생성에서 미지원 상태로 봐야 합니다.

AWQ unpack 로직 자체는 아래 명령으로 테스트할 수 있습니다.

```bash
RUSTFLAGS='-C target-cpu=native' cargo test --lib model_loader
```
