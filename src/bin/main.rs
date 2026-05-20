use ellm::bfloat16::Bf16;
use ellm::memory::allocator::allocate_init;
use ellm::memory::model_loader::SafeTensorsLoader;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
use ellm::qwen3_moe::reference_cpu::{supports_config as supports_qwen35_cpu, Qwen35CpuModel};
use ellm::serving::start::start;
use std::env;

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
        println!("Using Qwen3.6/Qwen3.5 CPU reference executor");
        println!("Loading safetensors from {}", weights_dir);
        let weights = SafeTensorsLoader::new(&weights_dir)
            .unwrap()
            .load_all_weights_bf16_packed_moe(config.num_experts)
            .unwrap();
        println!("Loaded {} tensors from safetensors", weights.len());

        let prompt_tokens = env::var("ELLM_PROMPT_IDS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| {
                        part.trim()
                            .parse::<usize>()
                            .expect("invalid ELLM_PROMPT_IDS token id")
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|tokens| !tokens.is_empty())
            .unwrap_or_else(|| vec![0]);
        let max_context = env::var("ELLM_MAX_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(prompt_tokens.len() + sequence_length + 1);

        println!(
            "prompt_ids={:?}, max_context={}, max_new_tokens={}",
            prompt_tokens, max_context, sequence_length
        );
        let mut model = Qwen35CpuModel::new(config, weights, max_context).unwrap();
        let generated = model.generate_greedy(&prompt_tokens, sequence_length);
        println!("Generated token ids:");
        for (idx, token) in generated.iter().enumerate() {
            println!("{:02}: {}", idx + 1, token);
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
