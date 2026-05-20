use crate::bfloat16::Bf16;
use std::collections::HashMap;
use std::f16;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

struct PackedExpertTensor {
    data: Vec<Bf16>,
    filled: Vec<bool>,
    per_expert_len: usize,
}

impl PackedExpertTensor {
    fn new(num_experts: usize, per_expert_len: usize) -> Self {
        Self {
            data: vec![Bf16::default(); num_experts * per_expert_len],
            filled: vec![false; num_experts],
            per_expert_len,
        }
    }

    fn insert(&mut self, expert_idx: usize, data: &[Bf16], tensor_name: &str) -> Result<()> {
        if expert_idx >= self.filled.len() {
            return Err(anyhow!(
                "Expert index {} out of range for {} with {} experts",
                expert_idx,
                tensor_name,
                self.filled.len()
            ));
        }
        if data.len() != self.per_expert_len {
            return Err(anyhow!(
                "Expert tensor {} has {} elements, expected {}",
                tensor_name,
                data.len(),
                self.per_expert_len
            ));
        }
        if self.filled[expert_idx] {
            return Err(anyhow!(
                "Duplicate expert {} while packing {}",
                expert_idx,
                tensor_name
            ));
        }

        let offset = expert_idx * self.per_expert_len;
        self.data[offset..offset + self.per_expert_len].copy_from_slice(data);
        self.filled[expert_idx] = true;
        Ok(())
    }

    fn finish(self, tensor_name: &str) -> Result<Vec<Bf16>> {
        if let Some((idx, _)) = self
            .filled
            .iter()
            .enumerate()
            .find(|(_, filled)| !**filled)
        {
            return Err(anyhow!("Missing expert {} while packing {}", idx, tensor_name));
        }
        Ok(self.data)
    }
}

fn packed_expert_name(name: &str) -> Option<(String, usize)> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.len() != 8
        || parts[0] != "model"
        || parts[1] != "layers"
        || parts[3] != "mlp"
        || parts[4] != "experts"
        || parts[7] != "weight"
    {
        return None;
    }

    let expert_idx = parts[5].parse::<usize>().ok()?;
    let projection = parts[6];
    if !matches!(projection, "gate_proj" | "up_proj" | "down_proj") {
        return None;
    }

    Some((
        format!(
            "model.layers.{}.mlp.experts.{}.weight",
            parts[2], projection
        ),
        expert_idx,
    ))
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    match exponent {
        0 => {
            if fraction == 0 {
                f32::from_bits(sign)
            } else {
                let mut mantissa = fraction;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                f32::from_bits(sign | (((exponent + 127) as u32) << 23) | (mantissa << 13))
            }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => f32::from_bits(sign | (((exponent as u32) + 112) << 23) | (fraction << 13)),
    }
}

// use crate::init::config::Config;
// use crate::llama::model::Model;
// use crate::ptensor::tensor::Tensor;

/// 在指定目录中查找safetensors文件
fn find_safetensors_file<P: AsRef<Path>>(model_dir: P) -> Result<std::path::PathBuf> {
    let model_dir = model_dir.as_ref();

    // 常见的safetensors文件名模式
    let patterns = [
        "model.safetensors",
        "pytorch_model.safetensors",
        "model-00001-of-00001.safetensors",
    ];

    // 首先尝试单文件模式
    for pattern in &patterns {
        let file_path = model_dir.join(pattern);
        if file_path.exists() {
            return Ok(file_path);
        }
    }

    // 如果没找到单文件，查找分片文件
    let entries = std::fs::read_dir(model_dir)?;
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if file_name_str.starts_with("model-") && file_name_str.ends_with(".safetensors") {
            // 找到第一个分片文件，返回它
            // 注意：如果是多文件模式，可能需要更复杂的逻辑来处理所有分片
            return Ok(entry.path());
        }
    }

    Err(anyhow!("No safetensors file found in the model directory"))
}

/// 用于处理多文件safetensors模型的加载器
pub struct SafeTensorsLoader {
    model_files: Vec<String>,
    // config_path: String,
}

impl SafeTensorsLoader {
    /// 创建多文件safetensors加载器
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        /* 
        let config_file = model_dir.join("config.json");

        if !config_file.exists() {
            return Err(anyhow!("config.json not found in model directory"));
        }*/

        // 查找所有safetensors文件
        let mut model_files = Vec::new();
        let entries = std::fs::read_dir(model_dir)?;

        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.ends_with(".safetensors") {
                println!("Found safetensors file: {}", entry.path().to_string_lossy().to_string());
                model_files.push(entry.path().to_string_lossy().to_string());
            }
        }

        if model_files.is_empty() {
            return Err(anyhow!("No safetensors files found in model directory"));
        }

        // 排序确保正确的加载顺序
        model_files.sort();

        Ok(SafeTensorsLoader {
            model_files,
            // config_path: config_file.to_string_lossy().to_string(),
        })
    }

    /// 加载所有权重文件
    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        let mut all_weights = HashMap::new();
        
        for model_file in &self.model_files {
            let file = File::open(model_file)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, tensor_view) in safetensors.tensors() {
                let data: Vec<f16> = match tensor_view.dtype() {
                    Dtype::F16 => {
                        let raw_data = tensor_view.data();
                        let f16_data: Vec<f16> = raw_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bytes = [chunk[0], chunk[1]];
                                f16::from_le_bytes(bytes)
                            })
                            .collect();
                        f16_data
                    }
                    Dtype::F32 => {
                        let raw_data = tensor_view.data();
                        let f32_data: Vec<f32> = raw_data
                            .chunks_exact(4)
                            .map(|chunk| {
                                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                f32::from_le_bytes(bytes)
                            })
                            .collect();
                        f32_data.iter().map(|&x| x as f16).collect()
                    }
                    Dtype::BF16 => {
                        // Convert BF16 weights to f16 for the legacy f16 path.
                        let raw_data = tensor_view.data();
                        raw_data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                Bf16::from_bits(bits).to_f32() as f16
                            })
                            .collect()
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported tensor dtype: {:?}",
                            tensor_view.dtype()
                        ));
                    }
                };

                all_weights.insert(name.to_string(), data);
            }
            // break;
        }

        Ok(all_weights)
    }

    pub fn load_all_weights_bf16(&self) -> Result<HashMap<String, Vec<Bf16>>> {
        self.load_all_weights_bf16_packed_moe(0)
    }

    pub fn load_all_weights_bf16_packed_moe(
        &self,
        num_experts: usize,
    ) -> Result<HashMap<String, Vec<Bf16>>> {
        let mut all_weights = HashMap::new();
        let mut packed_experts: HashMap<String, PackedExpertTensor> = HashMap::new();

        for model_file in &self.model_files {
            let file = File::open(model_file)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)?;

            for (name, tensor_view) in safetensors.tensors() {
                let raw_data = tensor_view.data();
                let data: Vec<Bf16> = match tensor_view.dtype() {
                    Dtype::BF16 => raw_data
                        .chunks_exact(2)
                        .map(|chunk| Bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                        .collect(),
                    Dtype::F16 => raw_data
                        .chunks_exact(2)
                        .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                        .map(Bf16::from_f32)
                        .collect(),
                    Dtype::F32 => raw_data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .map(Bf16::from_f32)
                        .collect(),
                    _ => {
                        return Err(anyhow!(
                            "Unsupported tensor dtype: {:?}",
                            tensor_view.dtype()
                        ));
                    }
                };

                if num_experts > 0 {
                    if let Some((packed_name, expert_idx)) = packed_expert_name(&name) {
                        let entry = packed_experts
                            .entry(packed_name.clone())
                            .or_insert_with(|| PackedExpertTensor::new(num_experts, data.len()));
                        entry.insert(expert_idx, &data, &name)?;
                        continue;
                    }
                }

                all_weights.insert(name.to_string(), data);
            }
        }

        for (name, packed) in packed_experts {
            let data = packed.finish(&name)?;
            all_weights.insert(name, data);
        }

        Ok(all_weights)
    }

    /*
    /// 加载配置
    pub fn load_config(&self) -> Result<Config> {
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    } */
}

/// 便民函数：从目录加载Llama3模型

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::cache::Cache;
    use safetensors::tensor::TensorView;
    use safetensors::{serialize_to_file, Dtype};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ellm-safetensors-test-{nonce}"))
    }

    #[test]
    fn test_load_bf16_safetensors_into_cache() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        let values = [1.0f32, -2.5, 3.25, 4.5]
            .into_iter()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::BF16, vec![2, 2], &bytes).unwrap();
        let tensors = vec![("model.embed_tokens.weight", view)];
        serialize_to_file(tensors, &None, &path).unwrap();

        let loader = SafeTensorsLoader::new(&dir).unwrap();
        let weights = loader.load_all_weights_bf16().unwrap();
        assert_eq!(weights["model.embed_tokens.weight"], values);

        let mut cache = Cache::new(weights);
        let ptr = cache.get("model.embed_tokens.weight", 4);
        for i in 0..values.len() {
            assert_eq!(unsafe { *ptr.add(i) }, values[i]);
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_pack_qwen_moe_expert_tensors() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        let expert0 = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let expert1 = [5.0f32, 6.0, 7.0, 8.0]
            .into_iter()
            .map(Bf16::from_f32)
            .collect::<Vec<_>>();
        let bytes0 = expert0
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let bytes1 = expert1
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let view0 = TensorView::new(Dtype::BF16, vec![2, 2], &bytes0).unwrap();
        let view1 = TensorView::new(Dtype::BF16, vec![2, 2], &bytes1).unwrap();
        let tensors = vec![
            ("model.layers.0.mlp.experts.0.gate_proj.weight", view0),
            ("model.layers.0.mlp.experts.1.gate_proj.weight", view1),
        ];
        serialize_to_file(tensors, &None, &path).unwrap();

        let loader = SafeTensorsLoader::new(&dir).unwrap();
        let weights = loader.load_all_weights_bf16_packed_moe(2).unwrap();
        let packed_name = "model.layers.0.mlp.experts.gate_proj.weight";
        let mut expected = expert0;
        expected.extend(expert1);

        assert_eq!(weights[packed_name], expected);
        assert!(!weights.contains_key("model.layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(!weights.contains_key("model.layers.0.mlp.experts.1.gate_proj.weight"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_safetensors() {
        // 这里可以添加测试代码

        let torch_file = String::from("D:/llama-3-chinese-8b-instruct-v3");
        let loader = SafeTensorsLoader::new(&torch_file).unwrap();
        loader.load_all_weights_f16().unwrap();
    }

    #[test]
    /// 使用SafeTensorsModelLoader的详细示例
    pub fn detailed_loading_example()  {
        let model_dir = "D:/llama-3-chinese-8b-instruct-v3";

        // 方法2：使用详细的加载器
        let loader = SafeTensorsLoader::new(model_dir).unwrap();

        // 分别加载配置和权重
        // let config = loader.load_config()?;
        let weights = loader.load_all_weights_f16().unwrap();

        // 验证关键层的存在
        let expected_layers = [
            "model.embed_tokens.weight",
            // "model.norm.weight",
            // "lm_head.weight",
        ];

        for layer_name in &expected_layers {
            if weights.contains_key(*layer_name) {
                println!("✓ Found layer: {}", layer_name);
            } else {
                println!("✗ Missing layer: {}", layer_name);
            }
        }

        // 检查transformer层
        for i in 0..2 {
            let layer_prefix = format!("model.layers.{}", i);
            let attention_layers = [
                format!("{}.self_attn.q_proj.weight", layer_prefix),
                format!("{}.self_attn.k_proj.weight", layer_prefix),
                format!("{}.self_attn.v_proj.weight", layer_prefix),
                format!("{}.self_attn.o_proj.weight", layer_prefix),
                format!("{}.mlp.gate_proj.weight", layer_prefix),
                format!("{}.mlp.up_proj.weight", layer_prefix),
                format!("{}.mlp.down_proj.weight", layer_prefix),
                format!("{}.input_layernorm.weight", layer_prefix),
                format!("{}.post_attention_layernorm.weight", layer_prefix),
            ];

            let mut layer_complete = true;
            for layer_name in &attention_layers {
                if !weights.contains_key(layer_name) {
                    layer_complete = false;
                    break;
                }
            }

            if layer_complete {
                println!("✓ Layer {} complete", i);
            } else {
                println!("✗ Layer {} incomplete", i);
            }
        }

        println!("Model loading verification completed!");

        // Ok(())
    }
}
