use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub eos_token_id: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub initializer_range: f32,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub output_router_logits: bool,
    #[serde(default)]
    pub qkv_bias: bool,
    #[serde(default)]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, String>>,
    #[serde(default)]
    pub rope_theta: usize,
    #[serde(default)]
    pub router_aux_loss_coef: f32,
    #[serde(default)]
    pub shared_experts_intermediate_size: usize,
    #[serde(default, alias = "shared_expert_intermediate_size")]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub torch_dtype: String,
    #[serde(default)]
    pub transformers_version: String,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let root: Value = serde_json::from_reader(reader)?;
        let mut config: Config = serde_json::from_value(
            root.get("text_config")
                .cloned()
                .unwrap_or_else(|| root.clone()),
        )?;

        if let Some(architectures) = root.get("architectures") {
            config.architectures = serde_json::from_value(architectures.clone())?;
        }
        if let Some(model_type) = root.get("model_type").and_then(Value::as_str) {
            config.model_type = model_type.to_string();
        }
        if config.torch_dtype.is_empty() {
            if let Some(dtype) = root
                .get("torch_dtype")
                .or_else(|| root.get("text_config").and_then(|v| v.get("dtype")))
                .and_then(Value::as_str)
            {
                config.torch_dtype = dtype.to_string();
            }
        }
        if let Some(transformers_version) = root.get("transformers_version").and_then(Value::as_str)
        {
            config.transformers_version = transformers_version.to_string();
        }
        if config.shared_experts_intermediate_size == 0 {
            config.shared_experts_intermediate_size = config.shared_expert_intermediate_size;
        }
        if config.rope_theta == 0 {
            if let Some(rope_theta) = root
                .get("text_config")
                .and_then(|v| v.get("rope_parameters"))
                .and_then(|v| v.get("rope_theta"))
                .and_then(Value::as_u64)
            {
                config.rope_theta = rope_theta as usize;
            }
        }
        Ok(config)
    }

    pub fn unsupported_layer_types(&self) -> Vec<(usize, String)> {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, layer_type)| layer_type.as_str() != "full_attention")
            .map(|(idx, layer_type)| (idx, layer_type.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_file() {
        let config = Config::load_from_file(r"models/Qwen3-Coder-30B-A3B-Instruct/config.json");
        match config {
            Ok(cfg) => println!("{:?}", cfg),
            Err(e) => println!("Error loading config: {}", e),
        }
    }
}
