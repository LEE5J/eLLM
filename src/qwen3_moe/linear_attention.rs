#[derive(Debug, Clone)]
pub struct GatedDeltaNetStepConfig {
    pub hidden_size: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNetStepConfig {
    #[inline]
    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }

    #[inline]
    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }

    #[inline]
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

#[derive(Debug, Clone)]
pub struct GatedDeltaNetWeights<'a> {
    pub in_proj_qkv: &'a [f32],
    pub in_proj_z: &'a [f32],
    pub in_proj_b: &'a [f32],
    pub in_proj_a: &'a [f32],
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub out_proj: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct GatedDeltaNetState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl GatedDeltaNetState {
    pub fn new(batch_size: usize, config: &GatedDeltaNetStepConfig) -> Self {
        Self {
            conv_state: vec![0.0; batch_size * config.conv_dim() * config.conv_kernel_size],
            recurrent_state: vec![
                0.0;
                batch_size
                    * config.num_value_heads
                    * config.key_head_dim
                    * config.value_head_dim
            ],
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn matvec_rows(input: &[f32], weight_rows: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(input.len(), cols);
    assert_eq!(weight_rows.len(), rows * cols);

    let mut output = vec![0.0; rows];
    for row in 0..rows {
        let weight = &weight_rows[row * cols..(row + 1) * cols];
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += input[col] * weight[col];
        }
        output[row] = acc;
    }
    output
}

fn l2_normalize_in_place(values: &mut [f32]) {
    let sum_sq = values.iter().map(|v| v * v).sum::<f32>();
    let inv_norm = (sum_sq + 1e-6).sqrt().recip();
    for value in values {
        *value *= inv_norm;
    }
}

fn causal_conv1d_update_silu(
    current: &[f32],
    conv_state: &mut [f32],
    conv_weight: &[f32],
    config: &GatedDeltaNetStepConfig,
    batch_idx: usize,
) -> Vec<f32> {
    let conv_dim = config.conv_dim();
    let kernel = config.conv_kernel_size;
    assert_eq!(current.len(), conv_dim);
    assert_eq!(conv_weight.len(), conv_dim * kernel);

    let mut output = vec![0.0; conv_dim];
    let state_base = batch_idx * conv_dim * kernel;
    for channel in 0..conv_dim {
        let base = state_base + channel * kernel;
        for pos in 0..kernel - 1 {
            conv_state[base + pos] = conv_state[base + pos + 1];
        }
        conv_state[base + kernel - 1] = current[channel];

        let mut acc = 0.0f32;
        for tap in 0..kernel {
            acc += conv_state[base + tap] * conv_weight[channel * kernel + tap];
        }
        output[channel] = silu(acc);
    }
    output
}

fn recurrent_gated_delta_step(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    recurrent_state: &mut [f32],
    config: &GatedDeltaNetStepConfig,
    batch_idx: usize,
) -> Vec<f32> {
    let key_dim = config.key_dim();
    let value_dim = config.value_dim();
    assert_eq!(query.len(), key_dim);
    assert_eq!(key.len(), key_dim);
    assert_eq!(value.len(), value_dim);
    assert_eq!(g.len(), config.num_value_heads);
    assert_eq!(beta.len(), config.num_value_heads);

    let repeat = config.num_value_heads / config.num_key_heads;
    assert_eq!(config.num_value_heads % config.num_key_heads, 0);

    let mut output = vec![0.0; value_dim];
    let state_batch_stride = config.num_value_heads * config.key_head_dim * config.value_head_dim;
    let state_batch_base = batch_idx * state_batch_stride;
    let scale = (config.key_head_dim as f32).sqrt().recip();

    for v_head in 0..config.num_value_heads {
        let k_head = v_head / repeat;
        let q_base = k_head * config.key_head_dim;
        let k_base = k_head * config.key_head_dim;
        let v_base = v_head * config.value_head_dim;

        let mut q = query[q_base..q_base + config.key_head_dim].to_vec();
        let mut k = key[k_base..k_base + config.key_head_dim].to_vec();
        l2_normalize_in_place(&mut q);
        l2_normalize_in_place(&mut k);
        for value in &mut q {
            *value *= scale;
        }

        let decay = g[v_head].exp();
        let beta = beta[v_head];
        let state_base = state_batch_base + v_head * config.key_head_dim * config.value_head_dim;

        for state_value in &mut recurrent_state
            [state_base..state_base + config.key_head_dim * config.value_head_dim]
        {
            *state_value *= decay;
        }

        let mut kv_mem = vec![0.0; config.value_head_dim];
        for k_idx in 0..config.key_head_dim {
            let state_row = state_base + k_idx * config.value_head_dim;
            for v_idx in 0..config.value_head_dim {
                kv_mem[v_idx] += recurrent_state[state_row + v_idx] * k[k_idx];
            }
        }

        let mut delta = vec![0.0; config.value_head_dim];
        for v_idx in 0..config.value_head_dim {
            delta[v_idx] = (value[v_base + v_idx] - kv_mem[v_idx]) * beta;
        }

        for k_idx in 0..config.key_head_dim {
            let state_row = state_base + k_idx * config.value_head_dim;
            for v_idx in 0..config.value_head_dim {
                recurrent_state[state_row + v_idx] += k[k_idx] * delta[v_idx];
            }
        }

        for v_idx in 0..config.value_head_dim {
            let mut acc = 0.0f32;
            for k_idx in 0..config.key_head_dim {
                acc +=
                    recurrent_state[state_base + k_idx * config.value_head_dim + v_idx] * q[k_idx];
            }
            output[v_base + v_idx] = acc;
        }
    }

    output
}

fn gated_rms_norm(
    input: &[f32],
    gate: &[f32],
    weight: &[f32],
    eps: f32,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), gate.len());
    assert_eq!(input.len() % head_dim, 0);
    assert_eq!(weight.len(), head_dim);

    let mut output = vec![0.0; input.len()];
    for head in 0..input.len() / head_dim {
        let base = head * head_dim;
        let mut variance = 0.0f32;
        for idx in 0..head_dim {
            let value = input[base + idx];
            variance += value * value;
        }
        let inv_rms = (variance / head_dim as f32 + eps).sqrt().recip();
        for idx in 0..head_dim {
            output[base + idx] = input[base + idx] * inv_rms * weight[idx] * silu(gate[base + idx]);
        }
    }
    output
}

pub fn gated_delta_net_decode_step_f32(
    hidden_states: &[f32],
    weights: &GatedDeltaNetWeights<'_>,
    state: &mut GatedDeltaNetState,
    config: &GatedDeltaNetStepConfig,
    batch_size: usize,
) -> Vec<f32> {
    assert_eq!(hidden_states.len(), batch_size * config.hidden_size);
    assert_eq!(
        weights.in_proj_qkv.len(),
        config.conv_dim() * config.hidden_size
    );
    assert_eq!(
        weights.in_proj_z.len(),
        config.value_dim() * config.hidden_size
    );
    assert_eq!(
        weights.in_proj_b.len(),
        config.num_value_heads * config.hidden_size
    );
    assert_eq!(
        weights.in_proj_a.len(),
        config.num_value_heads * config.hidden_size
    );
    assert_eq!(weights.a_log.len(), config.num_value_heads);
    assert_eq!(weights.dt_bias.len(), config.num_value_heads);
    assert_eq!(
        weights.out_proj.len(),
        config.hidden_size * config.value_dim()
    );

    let mut output = vec![0.0; batch_size * config.hidden_size];
    for batch_idx in 0..batch_size {
        let hidden =
            &hidden_states[batch_idx * config.hidden_size..(batch_idx + 1) * config.hidden_size];
        let mixed_qkv = matvec_rows(
            hidden,
            weights.in_proj_qkv,
            config.conv_dim(),
            config.hidden_size,
        );
        let z = matvec_rows(
            hidden,
            weights.in_proj_z,
            config.value_dim(),
            config.hidden_size,
        );
        let b = matvec_rows(
            hidden,
            weights.in_proj_b,
            config.num_value_heads,
            config.hidden_size,
        )
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
        let a = matvec_rows(
            hidden,
            weights.in_proj_a,
            config.num_value_heads,
            config.hidden_size,
        );

        let mixed_qkv = causal_conv1d_update_silu(
            &mixed_qkv,
            &mut state.conv_state,
            weights.conv1d,
            config,
            batch_idx,
        );
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let query = &mixed_qkv[..key_dim];
        let key = &mixed_qkv[key_dim..key_dim * 2];
        let value = &mixed_qkv[key_dim * 2..key_dim * 2 + value_dim];

        let g = a
            .iter()
            .zip(weights.a_log)
            .zip(weights.dt_bias)
            .map(|((a_value, a_log), dt_bias)| -a_log.exp() * softplus(a_value + dt_bias))
            .collect::<Vec<_>>();

        let core = recurrent_gated_delta_step(
            query,
            key,
            value,
            &g,
            &b,
            &mut state.recurrent_state,
            config,
            batch_idx,
        );
        let normed = gated_rms_norm(
            &core,
            &z,
            weights.norm,
            config.rms_norm_eps,
            config.value_head_dim,
        );
        let projected = matvec_rows(
            &normed,
            weights.out_proj,
            config.hidden_size,
            config.value_dim(),
        );
        output[batch_idx * config.hidden_size..(batch_idx + 1) * config.hidden_size]
            .copy_from_slice(&projected);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {}: actual={} expected={}",
                idx,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_causal_conv1d_update_matches_silu_depthwise_step() {
        let config = GatedDeltaNetStepConfig {
            hidden_size: 2,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        };
        let mut state = vec![0.0; config.conv_dim() * config.conv_kernel_size];
        let current = vec![1.0, 2.0, 3.0];
        let weight = vec![
            0.1, 0.2, 0.3, 0.4, //
            0.2, 0.3, 0.4, 0.5, //
            0.3, 0.4, 0.5, 0.6,
        ];
        let out = causal_conv1d_update_silu(&current, &mut state, &weight, &config, 0);

        assert_close(&out, &[silu(0.4), silu(1.0), silu(1.8)], 1e-6);
        assert_close(&state[0..4], &[0.0, 0.0, 0.0, 1.0], 1e-6);
        assert_close(&state[4..8], &[0.0, 0.0, 0.0, 2.0], 1e-6);
        assert_close(&state[8..12], &[0.0, 0.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_gated_delta_decode_step_runs_recurrent_update() {
        let config = GatedDeltaNetStepConfig {
            hidden_size: 2,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            conv_kernel_size: 1,
            rms_norm_eps: 1e-6,
        };
        let hidden = vec![1.0, -2.0];
        let in_proj_qkv = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            0.5, -0.5,
        ];
        let in_proj_z = vec![1.0, 1.0];
        let in_proj_b = vec![0.25, -0.25];
        let in_proj_a = vec![0.5, 0.25];
        let conv1d = vec![1.0, 1.0, 1.0];
        let a_log = vec![0.0];
        let dt_bias = vec![0.0];
        let norm = vec![1.0];
        let out_proj = vec![2.0, -1.0];
        let weights = GatedDeltaNetWeights {
            in_proj_qkv: &in_proj_qkv,
            in_proj_z: &in_proj_z,
            in_proj_b: &in_proj_b,
            in_proj_a: &in_proj_a,
            conv1d: &conv1d,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm: &norm,
            out_proj: &out_proj,
        };
        let mut state = GatedDeltaNetState::new(1, &config);
        let output = gated_delta_net_decode_step_f32(&hidden, &weights, &mut state, &config, 1);

        assert_eq!(output.len(), 2);
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert_ne!(output, vec![0.0, 0.0]);
        assert!(state.recurrent_state[0].is_finite());
        assert_ne!(state.recurrent_state[0], 0.0);
    }
}
