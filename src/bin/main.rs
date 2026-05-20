use ellm::bfloat16::Bf16;
use ellm::memory::allocator::allocate_init;
use ellm::memory::model_loader::SafeTensorsLoader;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::qwen3_moe::reference_cpu::{supports_config as supports_qwen35_cpu, Qwen35CpuModel};
use ellm::serving::start::start;
use serde_json::Value;
use std::env;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn main() {
    println!("Initializing...");

    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        panic!("AVX512 BF16 is required for this run");
    }

    let sequence_length = env::var("ELLM_GENERATE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(30);
    let sequence_chunk_size = 1;
    let batch_size = 3;
    let topk_size = 8;

    let config_path = env::var("ELLM_CONFIG")
        .unwrap_or_else(|_| String::from("models/Qwen3-Coder-30B-A3B-Instruct/config1.json"));
    let config = Config::load_from_file(&config_path).unwrap();
    println!(
        "Using {}, dtype=bf16, avx512bf16=true, generated_tokens={}, batch_size={}",
        config_path, sequence_length, batch_size
    );

    if supports_qwen35_cpu(&config) {
        let weights_dir = env::var("ELLM_SAFETENSORS_DIR")
            .expect("ELLM_SAFETENSORS_DIR is required for the Qwen3.6/Qwen3.5 CPU reference path");
        let tokenizer = load_tokenizer(&weights_dir);
        let eos_token_ids = load_eos_token_ids(&weights_dir, config.eos_token_id);
        println!("Using Qwen3.6/Qwen3.5 CPU reference executor");
        println!("eos_token_ids={:?}", eos_token_ids);
        println!("Loading safetensors from {}", weights_dir);
        let weights = SafeTensorsLoader::new(&weights_dir)
            .unwrap()
            .load_all_weights_bf16_packed_moe(config.num_experts)
            .unwrap();
        println!("Loaded {} tensors from safetensors", weights.len());

        let prompt_tokens = qwen35_prompt_tokens(tokenizer.as_ref());
        let max_context = env::var("ELLM_MAX_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(prompt_tokens.len() + sequence_length + 1);

        println!(
            "prompt_ids={:?}, max_context={}, max_new_tokens={}",
            prompt_tokens, max_context, sequence_length
        );
        let mut model =
            Qwen35CpuModel::with_eos_token_ids(config, weights, max_context, eos_token_ids)
                .unwrap();
        let generated = model.generate_greedy(&prompt_tokens, sequence_length);
        println!("Generated token ids:");
        for (idx, token) in generated.iter().enumerate() {
            println!("{:02}: {}", idx + 1, token);
        }
        if let Some(tokenizer) = tokenizer.as_ref() {
            let generated_ids = generated
                .iter()
                .map(|&token| u32::try_from(token).expect("token id exceeds u32"))
                .collect::<Vec<_>>();
            match tokenizer.decode(&generated_ids, true) {
                Ok(text) => println!("Decoded generated text:\n{}", text),
                Err(err) => eprintln!("tokenizer decode failed: {}", err),
            }
        }
        return;
    }

    let unsupported_layer_types = config.unsupported_layer_types();
    if !unsupported_layer_types.is_empty() {
        let preview = unsupported_layer_types
            .iter()
            .take(8)
            .map(|(idx, layer_type)| format!("{}:{}", idx, layer_type))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "This eLLM CPU executor currently supports full_attention Qwen3-MoE layers only. \
             The loaded config contains unsupported layer types ({} total; first: {}). \
             Qwen3.6/Qwen3.5 MoE models require a GatedDeltaNet linear_attention CPU operator before they can run here.",
            unsupported_layer_types.len(),
            preview
        );
        std::process::exit(2);
    }

    let mut model = if let Ok(weights_dir) = env::var("ELLM_SAFETENSORS_DIR") {
        println!("Loading safetensors from {}", weights_dir);
        let weights = SafeTensorsLoader::new(&weights_dir)
            .unwrap()
            .load_all_weights_bf16_packed_moe(config.num_experts)
            .unwrap();
        println!("Loaded {} tensors from safetensors", weights.len());
        Model::<Bf16>::new_with_parameters(
            &config,
            sequence_length,
            sequence_chunk_size,
            batch_size,
            topk_size,
            weights,
        )
    } else {
        Model::<Bf16>::new(
            &config,
            sequence_length,
            sequence_chunk_size,
            batch_size,
            topk_size,
        )
    };

    let sequences = allocate_init::<usize>((sequence_length + 1) * batch_size, 0);
    let _ = model.forward(sequences);

    start(model.operator_queue.take(), sequence_length, batch_size);

    println!("Generated token ids for batch 0:");
    for position in 0..sequence_length {
        let token = unsafe { *sequences.add((position + 1) * batch_size) };
        println!("{:02}: {}", position + 1, token);
    }
}

fn parse_token_ids(value: &str, env_name: &str) -> Vec<usize> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid {} token id: {}", env_name, part.trim()))
        })
        .collect()
}

fn load_tokenizer(weights_dir: &str) -> Option<Tokenizer> {
    let tokenizer_path = env::var("ELLM_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(weights_dir).join("tokenizer.json"));
    if !tokenizer_path.exists() {
        return None;
    }

    Some(Tokenizer::from_file(&tokenizer_path).unwrap_or_else(|err| {
        panic!(
            "failed to load tokenizer from {}: {}",
            tokenizer_path.display(),
            err
        )
    }))
}

fn qwen35_prompt_tokens(tokenizer: Option<&Tokenizer>) -> Vec<usize> {
    if let Ok(prompt_ids) = env::var("ELLM_PROMPT_IDS") {
        let tokens = parse_token_ids(&prompt_ids, "ELLM_PROMPT_IDS");
        if !tokens.is_empty() {
            return tokens;
        }
    }

    if let Ok(prompt) = env::var("ELLM_PROMPT") {
        let tokenizer = tokenizer.expect("ELLM_PROMPT requires tokenizer.json or ELLM_TOKENIZER");
        let prompt_text = if env::var("ELLM_RAW_PROMPT").ok().as_deref() == Some("1") {
            prompt
        } else {
            qwen_chat_prompt(&prompt)
        };
        let encoding = tokenizer
            .encode(prompt_text, true)
            .unwrap_or_else(|err| panic!("tokenizer encode failed: {}", err));
        let tokens = encoding
            .get_ids()
            .iter()
            .map(|&token| token as usize)
            .collect::<Vec<_>>();
        assert!(!tokens.is_empty(), "tokenizer produced an empty prompt");
        return tokens;
    }

    vec![0]
}

fn qwen_chat_prompt(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

fn load_eos_token_ids(weights_dir: &str, default_eos: usize) -> Vec<usize> {
    if let Ok(raw) = env::var("ELLM_EOS_TOKEN_IDS") {
        let mut ids = parse_token_ids(&raw, "ELLM_EOS_TOKEN_IDS");
        if !ids.is_empty() {
            ids.sort_unstable();
            ids.dedup();
            return ids;
        }
    }

    let mut ids = vec![default_eos];
    let path = Path::new(weights_dir).join("generation_config.json");
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(root) = serde_json::from_str::<Value>(&data) {
            match root.get("eos_token_id") {
                Some(Value::Number(value)) => {
                    if let Some(value) = value.as_u64() {
                        ids.push(value as usize);
                    }
                }
                Some(Value::Array(values)) => {
                    for value in values {
                        if let Some(value) = value.as_u64() {
                            ids.push(value as usize);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    ids.sort_unstable();
    ids.dedup();
    ids
}
