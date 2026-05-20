# Running Guide

This guide explains how to prepare the local model files, verify the
`safetensors` checkpoint, run the OpenAI-compatible server, and run the Rust BF16
test path.

## 1. Requirements

| Requirement | Notes |
| --- | --- |
| OS | Linux x86_64 |
| CPU | AVX-512 BF16 is required for the Rust BF16 path |
| RAM | 128 GiB was used for the reported Qwen3-Coder-30B-A3B run |
| Python | A virtual environment is recommended |
| Rust | Nightly toolchain, configured by `rust-toolchain.toml` |

Install Python dependencies in a local virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch transformers safetensors accelerate
```

Use the PyTorch install command that matches your platform if you need a
specific CPU or CUDA wheel.

## 2. Prepare the Model Directory

The full model weights are not stored in this repository. Put the Hugging Face
Qwen3-Coder BF16 `safetensors` files under this path:

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

## 3. Verify Safetensors Loading

Run the Python verification script first. It checks the shard metadata, loads the
model through Transformers, and generates a short answer.

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

## 4. Start the OpenAI-Compatible Server

Start the local chat server with model name `local_model`, API key `EMPTY`, host
`0.0.0.0`, and port `8000`:

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

## 5. Run the Rust BF16 Test Path

The Rust binary is a lower-level execution test. It loads the config and, when
`ELLM_SAFETENSORS_DIR` is set, loads BF16 weights from the local safetensors
directory.

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

## 6. Measure Host Memory Bandwidth

For the host memory reference number used in the README:

```bash
perf bench mem memcpy -f default -s 2GB -l 5
```

The tested host reported:

```text
27.014688 GB/sec
```

---

# 실행 가이드

이 문서는 로컬 모델 파일 준비, `safetensors` 검증, OpenAI 호환 서버 실행, Rust
BF16 테스트 실행 방법을 정리합니다.

## 1. 요구 사항

| 요구 사항 | 설명 |
| --- | --- |
| OS | Linux x86_64 |
| CPU | Rust BF16 경로 실행에는 AVX-512 BF16 필요 |
| RAM | 보고된 Qwen3-Coder-30B-A3B 실행은 128 GiB 환경에서 테스트 |
| Python | 가상 환경 사용 권장 |
| Rust | `rust-toolchain.toml`에서 nightly 사용 |

Python 의존성은 로컬 가상 환경에 설치하는 것을 권장합니다.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch transformers safetensors accelerate
```

CPU 전용 또는 CUDA wheel이 따로 필요하면 본인 환경에 맞는 PyTorch 설치 명령을
사용하면 됩니다.

## 2. 모델 디렉터리 준비

전체 모델 가중치는 이 저장소에 포함되어 있지 않습니다. Hugging Face
Qwen3-Coder BF16 `safetensors` 파일을 아래 경로에 배치합니다.

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

## 3. Safetensors 로딩 검증

먼저 Python 검증 스크립트를 실행합니다. 이 스크립트는 shard 메타데이터를 확인한
뒤 Transformers로 모델을 로드하고 짧은 답변을 생성합니다.

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

## 4. OpenAI 호환 서버 실행

모델 이름은 `local_model`, API key는 `EMPTY`, host는 `0.0.0.0`, port는 `8000`으로
실행합니다.

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

## 5. Rust BF16 테스트 실행

Rust 바이너리는 더 낮은 수준의 실행 테스트입니다. `ELLM_SAFETENSORS_DIR`를
지정하면 로컬 safetensors 디렉터리에서 BF16 가중치를 로드합니다.

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

## 6. Host Memory 대역폭 측정

README에 적은 host memory 참고값은 아래 명령으로 측정했습니다.

```bash
perf bench mem memcpy -f default -s 2GB -l 5
```

테스트 호스트에서는 다음 값이 나왔습니다.

```text
27.014688 GB/sec
```
