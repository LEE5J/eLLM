use ellm::bfloat16::Bf16;
use ellm::memory::allocator::allocate_init;
use ellm::memory::model_loader::SafeTensorsLoader;
use ellm::qwen3_moe::config::Config;
use ellm::qwen3_moe::model::Model;
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
