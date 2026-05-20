use crate::bfloat16::Bf16;
use crate::qwen3_moe::config::Config;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug)]
struct LinearAttentionState {
    conv_state: Vec<f32>,
    recurrent_state: Vec<f32>,
}

#[derive(Debug)]
struct FullAttentionState {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct LayerState {
    linear: Option<LinearAttentionState>,
    full: Option<FullAttentionState>,
}

pub struct Qwen35CpuModel {
    config: Config,
    weights: HashMap<String, Vec<Bf16>>,
    layers: Vec<LayerState>,
    max_context: usize,
    eos_token_ids: Vec<usize>,
}

pub fn supports_config(config: &Config) -> bool {
    config
        .layer_types
        .iter()
        .any(|layer_type| layer_type == "linear_attention")
        || config.model_type.contains("qwen3_5")
}

impl Qwen35CpuModel {
    pub fn new(
        config: Config,
        weights: HashMap<String, Vec<Bf16>>,
        max_context: usize,
    ) -> Result<Self, String> {
        let eos_token_id = config.eos_token_id;
        Self::with_eos_token_ids(config, weights, max_context, vec![eos_token_id])
    }

    pub fn with_eos_token_ids(
        config: Config,
        weights: HashMap<String, Vec<Bf16>>,
        max_context: usize,
        mut eos_token_ids: Vec<usize>,
    ) -> Result<Self, String> {
        if max_context == 0 {
            return Err("max_context must be > 0".to_string());
        }
        if eos_token_ids.is_empty() {
            eos_token_ids.push(config.eos_token_id);
        }
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer_type = config
                .layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");
            let linear = if layer_type == "linear_attention" {
                Some(LinearAttentionState {
                    conv_state: vec![0.0; linear_conv_dim(&config) * config.linear_conv_kernel_dim],
                    recurrent_state: vec![
                        0.0;
                        config.linear_num_value_heads
                            * config.linear_key_head_dim
                            * config.linear_value_head_dim
                    ],
                })
            } else {
                None
            };
            let full = if layer_type == "full_attention" {
                Some(FullAttentionState {
                    keys: vec![0.0; max_context * config.num_key_value_heads * config.head_dim],
                    values: vec![0.0; max_context * config.num_key_value_heads * config.head_dim],
                })
            } else {
                None
            };
            layers.push(LayerState { linear, full });
        }

        Ok(Self {
            config,
            weights,
            layers,
            max_context,
            eos_token_ids,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
    ) -> Vec<usize> {
        assert!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        self.reset_state();
        let mut all_tokens = Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        let mut next_token = 0usize;

        for &token in prompt_tokens {
            next_token = self.forward_token_top1(token, all_tokens.len());
            all_tokens.push(token);
        }

        for _ in 0..max_new_tokens {
            if all_tokens.len() >= self.max_context {
                break;
            }
            all_tokens.push(next_token);
            if self.is_eos(next_token) {
                break;
            }
            next_token = self.forward_token_top1(next_token, all_tokens.len() - 1);
        }

        all_tokens[prompt_tokens.len()..].to_vec()
    }

    fn is_eos(&self, token: usize) -> bool {
        self.eos_token_ids.binary_search(&token).is_ok()
    }

    pub fn is_eos_token(&self, token: usize) -> bool {
        self.is_eos(token)
    }

    pub fn reset_state(&mut self) {
        for layer in self.layers.iter_mut() {
            if let Some(linear) = layer.linear.as_mut() {
                linear.conv_state.fill(0.0);
                linear.recurrent_state.fill(0.0);
            }
            if let Some(full) = layer.full.as_mut() {
                full.keys.fill(0.0);
                full.values.fill(0.0);
            }
        }
    }

    pub fn forward_token_top1(&mut self, token: usize, position: usize) -> usize {
        assert!(
            position < self.max_context,
            "position {} exceeds max_context {}",
            position,
            self.max_context
        );
        assert!(
            token < self.config.vocab_size,
            "token {} exceeds vocab_size {}",
            token,
            self.config.vocab_size
        );

        let mut hidden = embedding(&self.weights, token, self.config.hidden_size);
        let config = &self.config;
        let weights = &self.weights;

        for layer_idx in 0..config.num_hidden_layers {
            let residual = hidden.clone();
            let input_norm_weight = weight(
                weights,
                &format!("model.layers.{}.input_layernorm.weight", layer_idx),
            );
            let norm_hidden = rms_norm_offset(&hidden, input_norm_weight, config.rms_norm_eps);

            let layer_type = config
                .layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");
            let mixer_output = match layer_type {
                "linear_attention" => {
                    let state = self.layers[layer_idx]
                        .linear
                        .as_mut()
                        .expect("linear layer state missing");
                    linear_attention_step(config, weights, layer_idx, &norm_hidden, state)
                }
                "full_attention" => {
                    let state = self.layers[layer_idx]
                        .full
                        .as_mut()
                        .expect("full attention layer state missing");
                    full_attention_step(config, weights, layer_idx, position, &norm_hidden, state)
                }
                other => panic!("unsupported Qwen3.5 layer type: {}", other),
            };

            add_assign(&mut hidden, &residual, &mixer_output);

            let residual = hidden.clone();
            let post_norm_weight = weight(
                weights,
                &format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            );
            let norm_hidden = rms_norm_offset(&hidden, post_norm_weight, config.rms_norm_eps);
            let mlp_output = sparse_moe_step(config, weights, layer_idx, &norm_hidden);
            add_assign(&mut hidden, &residual, &mlp_output);
        }

        let norm_weight = weight(weights, "model.norm.weight");
        let hidden = rms_norm_offset(&hidden, norm_weight, config.rms_norm_eps);
        top1_matvec_bf16(
            &hidden,
            weight(weights, "lm_head.weight"),
            config.vocab_size,
            config.hidden_size,
        )
        .0
    }
}

fn weight<'a>(weights: &'a HashMap<String, Vec<Bf16>>, name: &str) -> &'a [Bf16] {
    weights
        .get(name)
        .unwrap_or_else(|| panic!("missing weight {}", name))
        .as_slice()
}

fn embedding(weights: &HashMap<String, Vec<Bf16>>, token: usize, hidden_size: usize) -> Vec<f32> {
    let embed = weight(weights, "model.embed_tokens.weight");
    let base = token * hidden_size;
    embed[base..base + hidden_size]
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

fn add_assign(output: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(output.len(), a.len());
    assert_eq!(output.len(), b.len());
    for idx in 0..output.len() {
        output[idx] = a[idx] + b[idx];
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn rms_norm_offset(input: &[f32], weight: &[Bf16], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), weight.len());
    let mean_sq = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = (mean_sq + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * inv_rms * (1.0 + weight.to_f32()))
        .collect()
}

fn rms_norm_heads_offset(
    input: &[f32],
    weight: &[Bf16],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), num_heads * head_dim);
    assert_eq!(weight.len(), head_dim);
    let mut output = vec![0.0; input.len()];
    for head in 0..num_heads {
        let base = head * head_dim;
        let mean_sq = input[base..base + head_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / head_dim as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for dim in 0..head_dim {
            output[base + dim] = input[base + dim] * inv_rms * (1.0 + weight[dim].to_f32());
        }
    }
    output
}

fn matvec_bf16(input: &[f32], weight: &[Bf16], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(input.len(), cols);
    assert_eq!(weight.len(), rows * cols);
    let mut output = vec![0.0; rows];
    for row in 0..rows {
        let weight_row = &weight[row * cols..(row + 1) * cols];
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += input[col] * weight_row[col].to_f32();
        }
        output[row] = acc;
    }
    output
}

fn top1_matvec_bf16(input: &[f32], weight: &[Bf16], rows: usize, cols: usize) -> (usize, f32) {
    assert_eq!(input.len(), cols);
    assert_eq!(weight.len(), rows * cols);
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for row in 0..rows {
        let weight_row = &weight[row * cols..(row + 1) * cols];
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += input[col] * weight_row[col].to_f32();
        }
        if acc > best_value {
            best_value = acc;
            best_idx = row;
        }
    }
    (best_idx, best_value)
}

fn linear_key_dim(config: &Config) -> usize {
    config.linear_num_key_heads * config.linear_key_head_dim
}

fn linear_value_dim(config: &Config) -> usize {
    config.linear_num_value_heads * config.linear_value_head_dim
}

fn linear_conv_dim(config: &Config) -> usize {
    linear_key_dim(config) * 2 + linear_value_dim(config)
}

fn l2_normalize_in_place(values: &mut [f32]) {
    let sum_sq = values.iter().map(|value| value * value).sum::<f32>();
    let inv_norm = (sum_sq + 1e-6).sqrt().recip();
    for value in values {
        *value *= inv_norm;
    }
}

fn causal_conv1d_step(
    current: &[f32],
    conv_weight: &[Bf16],
    state: &mut LinearAttentionState,
    config: &Config,
) -> Vec<f32> {
    let conv_dim = linear_conv_dim(config);
    let kernel = config.linear_conv_kernel_dim;
    assert_eq!(current.len(), conv_dim);
    assert_eq!(conv_weight.len(), conv_dim * kernel);

    let mut output = vec![0.0; conv_dim];
    for channel in 0..conv_dim {
        let base = channel * kernel;
        for pos in 0..kernel - 1 {
            state.conv_state[base + pos] = state.conv_state[base + pos + 1];
        }
        state.conv_state[base + kernel - 1] = current[channel];

        let mut acc = 0.0f32;
        for tap in 0..kernel {
            acc += state.conv_state[base + tap] * conv_weight[channel * kernel + tap].to_f32();
        }
        output[channel] = silu(acc);
    }
    output
}

fn linear_attention_step(
    config: &Config,
    weights: &HashMap<String, Vec<Bf16>>,
    layer_idx: usize,
    hidden: &[f32],
    state: &mut LinearAttentionState,
) -> Vec<f32> {
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);
    let conv_dim = linear_conv_dim(config);
    let key_dim = linear_key_dim(config);
    let value_dim = linear_value_dim(config);

    let mixed_qkv = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.in_proj_qkv.weight", prefix)),
        conv_dim,
        config.hidden_size,
    );
    let z = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.in_proj_z.weight", prefix)),
        value_dim,
        config.hidden_size,
    );
    let beta = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.in_proj_b.weight", prefix)),
        config.linear_num_value_heads,
        config.hidden_size,
    )
    .into_iter()
    .map(sigmoid)
    .collect::<Vec<_>>();
    let a = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.in_proj_a.weight", prefix)),
        config.linear_num_value_heads,
        config.hidden_size,
    );

    let mixed_qkv = causal_conv1d_step(
        &mixed_qkv,
        weight(weights, &format!("{}.conv1d.weight", prefix)),
        state,
        config,
    );
    let query = &mixed_qkv[..key_dim];
    let key = &mixed_qkv[key_dim..key_dim * 2];
    let value = &mixed_qkv[key_dim * 2..key_dim * 2 + value_dim];

    let a_log = weight(weights, &format!("{}.A_log", prefix));
    let dt_bias = weight(weights, &format!("{}.dt_bias", prefix));
    let g = a
        .iter()
        .zip(a_log)
        .zip(dt_bias)
        .map(|((a_value, a_log), dt_bias)| {
            -a_log.to_f32().exp() * softplus(a_value + dt_bias.to_f32())
        })
        .collect::<Vec<_>>();

    let core = recurrent_gated_delta_step(
        query,
        key,
        value,
        &g,
        &beta,
        &mut state.recurrent_state,
        config,
    );
    let normed = gated_rms_norm(
        &core,
        &z,
        weight(weights, &format!("{}.norm.weight", prefix)),
        config.rms_norm_eps,
        config.linear_value_head_dim,
    );

    matvec_bf16(
        &normed,
        weight(weights, &format!("{}.out_proj.weight", prefix)),
        config.hidden_size,
        value_dim,
    )
}

fn recurrent_gated_delta_step(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    recurrent_state: &mut [f32],
    config: &Config,
) -> Vec<f32> {
    let repeat = config.linear_num_value_heads / config.linear_num_key_heads;
    assert_eq!(
        config.linear_num_value_heads % config.linear_num_key_heads,
        0
    );

    let mut output = vec![0.0; linear_value_dim(config)];
    let scale = (config.linear_key_head_dim as f32).sqrt().recip();

    for v_head in 0..config.linear_num_value_heads {
        let k_head = v_head / repeat;
        let q_base = k_head * config.linear_key_head_dim;
        let k_base = k_head * config.linear_key_head_dim;
        let v_base = v_head * config.linear_value_head_dim;

        let mut q = query[q_base..q_base + config.linear_key_head_dim].to_vec();
        let mut k = key[k_base..k_base + config.linear_key_head_dim].to_vec();
        l2_normalize_in_place(&mut q);
        l2_normalize_in_place(&mut k);
        for value in &mut q {
            *value *= scale;
        }

        let state_base = v_head * config.linear_key_head_dim * config.linear_value_head_dim;
        for state_value in &mut recurrent_state
            [state_base..state_base + config.linear_key_head_dim * config.linear_value_head_dim]
        {
            *state_value *= g[v_head].exp();
        }

        let mut kv_mem = vec![0.0; config.linear_value_head_dim];
        for k_idx in 0..config.linear_key_head_dim {
            let state_row = state_base + k_idx * config.linear_value_head_dim;
            for v_idx in 0..config.linear_value_head_dim {
                kv_mem[v_idx] += recurrent_state[state_row + v_idx] * k[k_idx];
            }
        }

        let mut delta = vec![0.0; config.linear_value_head_dim];
        for v_idx in 0..config.linear_value_head_dim {
            delta[v_idx] = (value[v_base + v_idx] - kv_mem[v_idx]) * beta[v_head];
        }

        for k_idx in 0..config.linear_key_head_dim {
            let state_row = state_base + k_idx * config.linear_value_head_dim;
            for v_idx in 0..config.linear_value_head_dim {
                recurrent_state[state_row + v_idx] += k[k_idx] * delta[v_idx];
            }
        }

        for v_idx in 0..config.linear_value_head_dim {
            let mut acc = 0.0f32;
            for k_idx in 0..config.linear_key_head_dim {
                acc += recurrent_state[state_base + k_idx * config.linear_value_head_dim + v_idx]
                    * q[k_idx];
            }
            output[v_base + v_idx] = acc;
        }
    }

    output
}

fn gated_rms_norm(
    input: &[f32],
    gate: &[f32],
    weight: &[Bf16],
    eps: f32,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), gate.len());
    assert_eq!(input.len() % head_dim, 0);
    assert_eq!(weight.len(), head_dim);

    let mut output = vec![0.0; input.len()];
    for head in 0..input.len() / head_dim {
        let base = head * head_dim;
        let mean_sq = input[base..base + head_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / head_dim as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for dim in 0..head_dim {
            output[base + dim] =
                input[base + dim] * inv_rms * weight[dim].to_f32() * silu(gate[base + dim]);
        }
    }
    output
}

fn full_attention_step(
    config: &Config,
    weights: &HashMap<String, Vec<Bf16>>,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    state: &mut FullAttentionState,
) -> Vec<f32> {
    let prefix = format!("model.layers.{}.self_attn", layer_idx);
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;

    let q_and_gate = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.q_proj.weight", prefix)),
        q_dim * 2,
        config.hidden_size,
    );
    let mut query = rms_norm_heads_offset(
        &q_and_gate[..q_dim],
        weight(weights, &format!("{}.q_norm.weight", prefix)),
        config.num_attention_heads,
        config.head_dim,
        config.rms_norm_eps,
    );
    let gate = &q_and_gate[q_dim..q_dim * 2];

    let mut key = rms_norm_heads_offset(
        &matvec_bf16(
            hidden,
            weight(weights, &format!("{}.k_proj.weight", prefix)),
            kv_dim,
            config.hidden_size,
        ),
        weight(weights, &format!("{}.k_norm.weight", prefix)),
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps,
    );
    let value = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.v_proj.weight", prefix)),
        kv_dim,
        config.hidden_size,
    );

    apply_rope(
        &mut query,
        config.num_attention_heads,
        config.head_dim,
        config,
        position,
    );
    apply_rope(
        &mut key,
        config.num_key_value_heads,
        config.head_dim,
        config,
        position,
    );

    let cache_offset = position * kv_dim;
    state.keys[cache_offset..cache_offset + kv_dim].copy_from_slice(&key);
    state.values[cache_offset..cache_offset + kv_dim].copy_from_slice(&value);

    let mut attn_output = vec![0.0; q_dim];
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;
    let scaling = (config.head_dim as f32).sqrt().recip();

    for q_head in 0..config.num_attention_heads {
        let kv_head = q_head / kv_repeat;
        let q_base = q_head * config.head_dim;
        let mut scores = vec![0.0; position + 1];
        for pos in 0..=position {
            let k_base = pos * kv_dim + kv_head * config.head_dim;
            let mut score = 0.0f32;
            for dim in 0..config.head_dim {
                score += query[q_base + dim] * state.keys[k_base + dim];
            }
            scores[pos] = score * scaling;
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            sum += *score;
        }

        for pos in 0..=position {
            let prob = scores[pos] / sum;
            let v_base = pos * kv_dim + kv_head * config.head_dim;
            for dim in 0..config.head_dim {
                attn_output[q_base + dim] += prob * state.values[v_base + dim];
            }
        }
    }

    for idx in 0..attn_output.len() {
        attn_output[idx] *= sigmoid(gate[idx]);
    }

    matvec_bf16(
        &attn_output,
        weight(weights, &format!("{}.o_proj.weight", prefix)),
        config.hidden_size,
        q_dim,
    )
}

fn apply_rope(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    config: &Config,
    position: usize,
) {
    let rotary_dim = ((head_dim as f32) * config.partial_rotary_factor) as usize;
    if rotary_dim == 0 {
        return;
    }
    assert_eq!(rotary_dim % 2, 0);
    let half = rotary_dim / 2;
    let theta = if config.rope_theta == 0 {
        10000.0
    } else {
        config.rope_theta as f32
    };

    for pair in 0..half {
        let inv_freq = theta.powf(-((2 * pair) as f32) / rotary_dim as f32);
        let angle = position as f32 * inv_freq;
        let cos = angle.cos();
        let sin = angle.sin();
        for head in 0..num_heads {
            let base = head * head_dim;
            let x1 = values[base + pair];
            let x2 = values[base + half + pair];
            values[base + pair] = x1 * cos - x2 * sin;
            values[base + half + pair] = x2 * cos + x1 * sin;
        }
    }
}

fn sparse_moe_step(
    config: &Config,
    weights: &HashMap<String, Vec<Bf16>>,
    layer_idx: usize,
    hidden: &[f32],
) -> Vec<f32> {
    let prefix = format!("model.layers.{}.mlp", layer_idx);
    let gate_logits = matvec_bf16(
        hidden,
        weight(weights, &format!("{}.gate.weight", prefix)),
        config.num_experts,
        config.hidden_size,
    );
    let routes = normalized_topk_routes(&gate_logits, config.num_experts_per_tok);
    let mut output = vec![0.0; config.hidden_size];

    let gate_weight = weight(weights, &format!("{}.experts.gate_proj.weight", prefix));
    let up_weight = weight(weights, &format!("{}.experts.up_proj.weight", prefix));
    let down_weight = weight(weights, &format!("{}.experts.down_proj.weight", prefix));

    for (expert_idx, route_weight) in routes {
        let expert_gate = expert_matvec(
            hidden,
            gate_weight,
            expert_idx,
            config.moe_intermediate_size,
            config.hidden_size,
        );
        let expert_up = expert_matvec(
            hidden,
            up_weight,
            expert_idx,
            config.moe_intermediate_size,
            config.hidden_size,
        );
        let mut activated = vec![0.0; config.moe_intermediate_size];
        for idx in 0..activated.len() {
            activated[idx] = silu(expert_gate[idx]) * expert_up[idx];
        }
        let down = expert_matvec(
            &activated,
            down_weight,
            expert_idx,
            config.hidden_size,
            config.moe_intermediate_size,
        );
        for idx in 0..output.len() {
            output[idx] += down[idx] * route_weight;
        }
    }

    if config.shared_experts_intermediate_size > 0 {
        let shared_gate = matvec_bf16(
            hidden,
            weight(weights, &format!("{}.shared_expert_gate.weight", prefix)),
            1,
            config.hidden_size,
        )[0];
        let shared_gate_proj = matvec_bf16(
            hidden,
            weight(
                weights,
                &format!("{}.shared_expert.gate_proj.weight", prefix),
            ),
            config.shared_experts_intermediate_size,
            config.hidden_size,
        );
        let shared_up = matvec_bf16(
            hidden,
            weight(weights, &format!("{}.shared_expert.up_proj.weight", prefix)),
            config.shared_experts_intermediate_size,
            config.hidden_size,
        );
        let mut shared_activated = vec![0.0; config.shared_experts_intermediate_size];
        for idx in 0..shared_activated.len() {
            shared_activated[idx] = silu(shared_gate_proj[idx]) * shared_up[idx];
        }
        let shared_down = matvec_bf16(
            &shared_activated,
            weight(
                weights,
                &format!("{}.shared_expert.down_proj.weight", prefix),
            ),
            config.hidden_size,
            config.shared_experts_intermediate_size,
        );
        let gate = sigmoid(shared_gate);
        for idx in 0..output.len() {
            output[idx] += shared_down[idx] * gate;
        }
    }

    output
}

fn expert_matvec(
    input: &[f32],
    weight: &[Bf16],
    expert_idx: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let expert_stride = rows * cols;
    let base = expert_idx * expert_stride;
    matvec_bf16(input, &weight[base..base + expert_stride], rows, cols)
}

fn normalized_topk_routes(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|&lhs, &rhs| {
        logits[rhs]
            .partial_cmp(&logits[lhs])
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.cmp(&rhs))
    });
    indices.truncate(top_k);

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let full_sum = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .sum::<f32>();

    let mut routes = indices
        .into_iter()
        .map(|idx| (idx, (logits[idx] - max_logit).exp() / full_sum))
        .collect::<Vec<_>>();
    let top_sum = routes.iter().map(|(_, value)| *value).sum::<f32>();
    for (_, value) in &mut routes {
        *value /= top_sum;
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_vec(values: &[f32]) -> Vec<Bf16> {
        values.iter().copied().map(Bf16::from_f32).collect()
    }

    fn tiny_config(layer_type: &str) -> Config {
        Config {
            architectures: vec!["Qwen3_5MoeForConditionalGeneration".to_string()],
            attention_dropout: 0.0,
            attention_bias: false,
            attn_output_gate: true,
            decoder_sparse_step: 0,
            eos_token_id: 3,
            head_dim: 1,
            hidden_act: "silu".to_string(),
            hidden_size: 2,
            initializer_range: 0.0,
            intermediate_size: 0,
            max_position_embeddings: 8,
            max_window_layers: 0,
            mlp_only_layers: vec![],
            model_type: "qwen3_5_moe".to_string(),
            moe_intermediate_size: 1,
            norm_topk_prob: true,
            num_attention_heads: 1,
            num_experts: 2,
            num_experts_per_tok: 1,
            num_hidden_layers: 1,
            num_key_value_heads: 1,
            output_router_logits: false,
            qkv_bias: false,
            rms_norm_eps: 1e-6,
            rope_scaling: None,
            rope_theta: 10000,
            router_aux_loss_coef: 0.0,
            shared_experts_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            sliding_window: None,
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".to_string(),
            transformers_version: String::new(),
            use_cache: true,
            use_qk_norm: true,
            use_sliding_window: false,
            vocab_size: 4,
            layer_types: vec![layer_type.to_string()],
            linear_conv_kernel_dim: 1,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            partial_rotary_factor: 0.0,
        }
    }

    fn common_weights() -> HashMap<String, Vec<Bf16>> {
        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5]),
        );
        weights.insert("model.norm.weight".to_string(), bf16_vec(&[0.0, 0.0]));
        weights.insert(
            "lm_head.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            bf16_vec(&[0.0, 0.0]),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            bf16_vec(&[0.0, 0.0]),
        );
        weights.insert(
            "model.layers.0.mlp.gate.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.mlp.experts.gate_proj.weight".to_string(),
            bf16_vec(&[0.5, 0.0, 0.0, 0.5]),
        );
        weights.insert(
            "model.layers.0.mlp.experts.up_proj.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.mlp.experts.down_proj.weight".to_string(),
            bf16_vec(&[0.5, 0.5, 0.5, 0.5]),
        );
        weights.insert(
            "model.layers.0.mlp.shared_expert_gate.weight".to_string(),
            bf16_vec(&[0.25, 0.25]),
        );
        weights.insert(
            "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
            bf16_vec(&[0.25, 0.25]),
        );
        weights.insert(
            "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
            bf16_vec(&[0.5, 0.5]),
        );
        weights.insert(
            "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
            bf16_vec(&[0.5, 0.5]),
        );
        weights
    }

    #[test]
    fn test_reference_cpu_linear_layer_runs() {
        let mut weights = common_weights();
        weights.insert(
            "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]),
        );
        weights.insert(
            "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            bf16_vec(&[1.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            bf16_vec(&[0.25, 0.25]),
        );
        weights.insert(
            "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            bf16_vec(&[0.5, 0.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.conv1d.weight".to_string(),
            bf16_vec(&[1.0, 1.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.A_log".to_string(),
            bf16_vec(&[0.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.dt_bias".to_string(),
            bf16_vec(&[0.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.norm.weight".to_string(),
            bf16_vec(&[1.0]),
        );
        weights.insert(
            "model.layers.0.linear_attn.out_proj.weight".to_string(),
            bf16_vec(&[1.0, 0.5]),
        );

        let mut model = Qwen35CpuModel::new(tiny_config("linear_attention"), weights, 4).unwrap();
        let token = model.forward_token_top1(0, 0);
        assert!(token < 4);
    }

    #[test]
    fn test_reference_cpu_full_attention_layer_runs() {
        let mut weights = common_weights();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            bf16_vec(&[1.0, 0.0, 0.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            bf16_vec(&[1.0, 0.0]),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            bf16_vec(&[0.0, 1.0]),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            bf16_vec(&[1.0, 0.5]),
        );
        weights.insert(
            "model.layers.0.self_attn.q_norm.weight".to_string(),
            bf16_vec(&[0.0]),
        );
        weights.insert(
            "model.layers.0.self_attn.k_norm.weight".to_string(),
            bf16_vec(&[0.0]),
        );

        let mut model = Qwen35CpuModel::new(tiny_config("full_attention"), weights, 4).unwrap();
        let token = model.forward_token_top1(1, 0);
        assert!(token < 4);
    }

    #[test]
    fn test_normalized_topk_routes_sum_to_one() {
        let routes = normalized_topk_routes(&[1.0, 3.0, 2.0, -1.0], 2);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].0, 1);
        assert_eq!(routes[1].0, 2);
        let sum = routes.iter().map(|(_, value)| value).sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reference_cpu_accepts_multiple_eos_tokens() {
        let model = Qwen35CpuModel::with_eos_token_ids(
            tiny_config("linear_attention"),
            common_weights(),
            4,
            vec![3, 2, 3],
        )
        .unwrap();
        assert!(model.is_eos(2));
        assert!(model.is_eos(3));
        assert!(!model.is_eos(1));
    }
}
