# eLLM BF16 Safetensors Experiment

This fork is an experimental validation of running a Qwen3-Coder MoE model from
Hugging Face `safetensors` on a consumer AMD Ryzen platform.

The original project was mainly oriented around server CPU paths such as AMX.
This version explores a Ryzen 9000-series path using AVX-512 BF16, because
consumer CPUs in this class do not provide the same FP16 execution path that
server platforms expose.

## Scope

| Area | Status |
| --- | --- |
| Model format | Hugging Face `safetensors` shards |
| Test model | Qwen3-Coder-30B-A3B-Instruct BF16 |
| Runtime target | CPU, AVX-512 BF16 |
| OpenAI-compatible API | Included as a local Python reference server |
| Purpose | Proof of operation and baseline performance measurement |

This repository is not intended as a production inference server yet. It is a
working experiment to confirm that the model can be loaded, executed, and served
locally on the tested host.

## Test System

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 5 9600X |
| Memory | 128 GiB DDR5 |
| DIMMs | 2 x 64 GB CORSAIR CMH128GX5M2B6400C42 |
| Configured memory speed | DDR5-5800, CL46 |
| Host memory bandwidth reference | `memcpy` 2 GB: 27.014688 GB/s |
| Batch size | 1 for the reported text-generation test |

The model weights are not committed to this repository. The local test used a
full Qwen3-Coder BF16 `safetensors` directory under:

```text
models/Qwen3-Coder-30B-A3B-Instruct-full/
```

## Measured Performance

| Phase | Result | Notes |
| --- | ---: | --- |
| Prefill (PP) | ~60 tokens/s | Short prompt, batch size 1 |
| Token generation (TG/TP) | ~5.4 tokens/s | CPU path, BF16 weights |
| Host `memcpy` bandwidth | 27.014688 GB/s | 2 GB buffer, `perf bench mem memcpy` |

Longer prompts and longer context windows are expected to reduce throughput,
especially during token generation. The current result should be read as a
small-batch baseline, not as a full context-length benchmark.

## OpenAI-Compatible Server

The repository includes a minimal local OpenAI-compatible chat server:

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

Use `EMPTY` as the API key for local clients:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [{"role": "user", "content": "Write a short Python example."}],
    "max_tokens": 128
  }'
```

## Notes

- The reported output quality was confirmed to be normal for short test prompts.
- The current work focuses on BF16 safetensors loading and CPU execution.
- The local model directory is intentionally ignored by Git because the full
  weight set is tens of GiB.
- This is an experimental baseline; performance numbers depend heavily on memory
  speed, CPU clocks, prompt length, and context length.

---

# eLLM BF16 Safetensors 실험

이 포크는 Hugging Face `safetensors` 형식의 Qwen3-Coder MoE 모델을 소비자용
AMD Ryzen 시스템에서 실행해 보기 위한 실험용 저장소입니다.

원본 프로젝트는 주로 AMX 같은 서버 CPU 경로를 기준으로 작성되어 있었습니다.
이 버전은 Ryzen 9000 시리즈에서 사용할 수 있는 AVX-512 BF16 경로를 실험합니다.
소비자용 CPU에서는 서버 플랫폼과 같은 FP16 실행 경로를 기대하기 어렵기 때문에,
BF16 기반 포팅 가능성을 확인하는 데 초점을 두었습니다.

## 범위

| 항목 | 상태 |
| --- | --- |
| 모델 형식 | Hugging Face `safetensors` 샤드 |
| 테스트 모델 | Qwen3-Coder-30B-A3B-Instruct BF16 |
| 실행 대상 | CPU, AVX-512 BF16 |
| OpenAI 호환 API | 로컬 Python 레퍼런스 서버 포함 |
| 목적 | 동작 검증 및 기본 성능 측정 |

이 저장소는 아직 프로덕션용 추론 서버를 목표로 한 것은 아닙니다. 테스트한
호스트에서 모델을 로드하고, 실행하고, 로컬 API로 서빙할 수 있는지를 확인하기
위한 실험 단계입니다.

## 테스트 시스템

| 구성 | 값 |
| --- | --- |
| CPU | AMD Ryzen 5 9600X |
| 메모리 | 128 GiB DDR5 |
| DIMM | 2 x 64 GB CORSAIR CMH128GX5M2B6400C42 |
| 설정된 메모리 속도 | DDR5-5800, CL46 |
| Host memory 대역폭 참고값 | `memcpy` 2 GB: 27.014688 GB/s |
| 배치 크기 | 보고된 텍스트 생성 테스트 기준 1 |

모델 가중치는 이 저장소에 커밋하지 않았습니다. 로컬 테스트에서는 전체
Qwen3-Coder BF16 `safetensors` 디렉터리를 아래 경로에 두고 사용했습니다.

```text
models/Qwen3-Coder-30B-A3B-Instruct-full/
```

## 측정 성능

| 단계 | 결과 | 비고 |
| --- | ---: | --- |
| Prefill (PP) | 약 60 tokens/s | 짧은 프롬프트, 배치 크기 1 |
| Token generation (TG/TP) | 약 5.4 tokens/s | CPU 경로, BF16 가중치 |
| Host `memcpy` 대역폭 | 27.014688 GB/s | 2 GB 버퍼, `perf bench mem memcpy` |

프롬프트가 길어지거나 컨텍스트 길이가 늘어나면 특히 token generation 단계의
성능은 더 낮아질 수 있습니다. 위 수치는 전체 컨텍스트 길이 벤치마크가 아니라,
짧은 입력과 작은 배치에서 확인한 기준값으로 보는 것이 좋습니다.

## OpenAI 호환 서버

이 저장소에는 로컬 OpenAI 호환 chat 서버가 포함되어 있습니다.

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

로컬 클라이언트에서는 API key로 `EMPTY`를 사용하면 됩니다.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local_model",
    "messages": [{"role": "user", "content": "짧은 Python 예제를 작성해줘."}],
    "max_tokens": 128
  }'
```

## 참고

- 짧은 테스트 프롬프트 기준으로 답변 품질은 정상적으로 나오는 것을 확인했습니다.
- 현재 작업은 BF16 safetensors 로딩과 CPU 실행 경로 확인에 초점을 둡니다.
- 전체 모델 가중치는 수십 GiB 규모이므로 Git에서 제외했습니다.
- 성능 수치는 메모리 속도, CPU 클럭, 프롬프트 길이, 컨텍스트 길이에 크게 영향을
  받습니다.
