# eLLM CPU BF16 Safetensors Experiment

This fork is an experimental validation of running a Qwen3-Coder MoE model from
Hugging Face `safetensors` on a consumer AMD Ryzen platform through the eLLM CPU
runtime. GPU offload is not used for the documented runs.

The original repository mainly targets server-class CPU lineups and is designed
around an AVX-512 FP16 execution path. Ryzen 9000-series CPUs expose AVX-512
BF16 instead, while many current LLM checkpoints are already distributed in
BF16. Because of that, this fork focuses on using BF16 weights directly rather
than building the experiment around an AVX-512 FP16 path.

## Scope

| Area | Status |
| --- | --- |
| Model format | Hugging Face `safetensors` shards and compressed-tensors AWQ |
| BF16 test model | Qwen3-Coder-30B-A3B-Instruct BF16 |
| AWQ test target | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit compatibility inspection |
| Primary runtime target | eLLM Rust CPU executor, AVX-512 BF16 |
| GPU usage | Not used |
| OpenAI-compatible API | CPU-only local compatibility wrapper for the BF16 baseline |
| Purpose | eLLM CPU operation validation and baseline performance measurement |

This repository is not intended as a production inference server yet. It is a
working experiment to confirm that the model can be loaded and executed locally
on the tested host through the CPU path.

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
| Prefill (PP) | ~60 tokens/s | eLLM CPU path, short prompt, batch size 1 |
| Token generation (TG/TP) | ~5.4 tokens/s | eLLM CPU path, BF16 weights |

Longer prompts and longer context windows are expected to reduce throughput,
especially during token generation. The current result should be read as a
small-batch baseline, not as a full context-length benchmark.

## Running

See [guide.md](guide.md) for setup and execution instructions, including model
placement, safetensors verification, the CPU-only OpenAI-compatible wrapper, and
the eLLM Rust BF16 test path.

## Notes

- The reported output quality was confirmed to be normal for short test prompts.
- The benchmark target is the eLLM Rust CPU path. The Python scripts are
  CPU-only helper tools for verification and OpenAI API compatibility checks.
- The Qwen3.6 AWQ target is not runnable in the eLLM CPU executor yet. Its
  compressed-tensors expert weights can be unpacked, and the nested config is
  parsed, but 30 of 40 text layers use GatedDeltaNet `linear_attention`, which
  still needs to be integrated into the eLLM CPU executor.
- The local model directory is intentionally ignored by Git because the full
  weight set is tens of GiB.
- This is an experimental baseline; performance numbers depend heavily on memory
  speed, CPU clocks, prompt length, and context length.

---

# eLLM CPU BF16 Safetensors 실험

이 포크는 Hugging Face `safetensors` 형식의 Qwen3-Coder MoE 모델을 소비자용
AMD Ryzen 시스템에서 eLLM CPU 런타임으로 실행해 보기 위한 실험용 저장소입니다.
문서화된 실행 경로에서는 GPU offload를 사용하지 않습니다.

원본 리포지토리는 서버급 CPU 라인업을 주 대상으로 하며, AVX-512 FP16 실행
경로를 기반으로 설계되어 있습니다. 반면 Ryzen 9000 시리즈는 AVX-512 BF16을
지원하고, 최근 LLM 체크포인트도 BF16 형식으로 배포되는 경우가 많습니다. 따라서
이 포크는 AVX-512 FP16 경로를 전제로 하기보다 BF16 가중치를 직접 사용하는
방향에 초점을 둡니다.

## 범위

| 항목 | 상태 |
| --- | --- |
| 모델 형식 | Hugging Face `safetensors` 샤드 및 compressed-tensors AWQ |
| BF16 테스트 모델 | Qwen3-Coder-30B-A3B-Instruct BF16 |
| AWQ 테스트 대상 | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit 호환성 점검 |
| 주 실행 대상 | eLLM Rust CPU executor, AVX-512 BF16 |
| GPU 사용 | 사용하지 않음 |
| OpenAI 호환 API | BF16 기준선 확인용 CPU 전용 로컬 호환 wrapper |
| 목적 | eLLM CPU 동작 검증 및 기본 성능 측정 |

이 저장소는 아직 프로덕션용 추론 서버를 목표로 한 것은 아닙니다. 테스트한
호스트에서 모델을 로드하고 CPU 경로로 실행할 수 있는지를 확인하기 위한 실험
단계입니다.

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
| Prefill (PP) | 약 60 tokens/s | eLLM CPU 경로, 짧은 프롬프트, 배치 크기 1 |
| Token generation (TG/TP) | 약 5.4 tokens/s | eLLM CPU 경로, BF16 가중치 |

프롬프트가 길어지거나 컨텍스트 길이가 늘어나면 특히 token generation 단계의
성능은 더 낮아질 수 있습니다. 위 수치는 전체 컨텍스트 길이 벤치마크가 아니라,
짧은 입력과 작은 배치에서 확인한 기준값으로 보는 것이 좋습니다.

## 실행 방법

모델 배치, safetensors 검증, CPU 전용 OpenAI 호환 wrapper 실행, eLLM Rust BF16
테스트 실행 방법은 [guide.md](guide.md)에 정리되어 있습니다.

## 참고

- 짧은 테스트 프롬프트 기준으로 답변 품질은 정상적으로 나오는 것을 확인했습니다.
- 성능 측정 기준은 eLLM Rust CPU 경로입니다. Python 스크립트는 safetensors
  검증과 OpenAI API 호환성 확인을 위한 CPU 전용 보조 도구입니다.
- Qwen3.6 AWQ 대상은 아직 eLLM CPU 실행기에서 실제 생성까지 실행할 수
  없습니다. compressed-tensors expert weight unpack과 중첩 config 파싱은
  추가했지만, text layer 40개 중 30개가 GatedDeltaNet `linear_attention`을
  사용하므로 eLLM CPU 실행기에 해당 경로를 통합하는 작업이 더 필요합니다.
- 전체 모델 가중치는 수십 GiB 규모이므로 Git에서 제외했습니다.
- 성능 수치는 메모리 속도, CPU 클럭, 프롬프트 길이, 컨텍스트 길이에 크게 영향을
  받습니다.
