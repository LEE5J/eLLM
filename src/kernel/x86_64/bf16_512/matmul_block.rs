#![allow(non_snake_case)]

use std::arch::x86_64::{
    __m512, __m512bh, __m512i, _mm512_dpbf16_ps, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_set1_epi32, _mm512_storeu_ps,
};
use std::mem;

use crate::bfloat16::Bf16;
use crate::init::matmul_params::MatMulParams;

#[inline(always)]
unsafe fn load_c16(c: *const Bf16) -> __m512 {
    let mut values = [0.0f32; 16];
    for lane in 0..16 {
        values[lane] = (*c.add(lane)).to_f32();
    }
    _mm512_loadu_ps(values.as_ptr())
}

#[inline(always)]
unsafe fn store_c16(acc: __m512, c: *mut Bf16, a: *const Bf16, b_panel: *const Bf16, k: usize, kc: usize, col: usize) {
    let mut values = [0.0f32; 16];
    _mm512_storeu_ps(values.as_mut_ptr(), acc);

    if k < kc {
        let a_tail = (*a.add(k)).to_f32();
        let b_row = b_panel.add(k * 32 + col);
        for lane in 0..16 {
            values[lane] += a_tail * (*b_row.add(lane)).to_f32();
        }
    }

    for lane in 0..16 {
        *c.add(lane) = Bf16::from_f32(values[lane]);
    }
}

#[inline(always)]
unsafe fn splat_pair(a0: Bf16, a1: Bf16) -> __m512bh {
    let pair = (a0.to_bits() as u32) | ((a1.to_bits() as u32) << 16);
    mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(pair as i32))
}

#[inline(always)]
unsafe fn load_b_pair16(b_panel: *const Bf16, k: usize, col: usize) -> __m512bh {
    let mut packed = [0u16; 32];
    let b0 = b_panel.add(k * 32 + col);
    let b1 = b_panel.add((k + 1) * 32 + col);

    for lane in 0..16 {
        packed[lane * 2] = (*b0.add(lane)).to_bits();
        packed[lane * 2 + 1] = (*b1.add(lane)).to_bits();
    }

    let raw = _mm512_loadu_si512(packed.as_ptr() as *const __m512i);
    mem::transmute::<__m512i, __m512bh>(raw)
}

/// AVX512 BF16 3x32 micro-kernel.
///
/// A is MR x Kc row-major, B is a packed Kc x 32 panel, and C is MR x 32.
/// Accumulation uses `_mm512_dpbf16_ps`; results are rounded back to Bf16.
#[target_feature(enable = "avx512bf16")]
pub unsafe fn matmul_block(
    a: *const Bf16,
    b_panel: *const Bf16,
    c: *mut Bf16,
    param: &MatMulParams,
) {
    debug_assert_eq!(param.a_row_step_micro, 3);
    debug_assert_eq!(param.b_row_step_micro, 32);

    let lda = param.a_row_step_macro;
    let ldc = param.b_row_step_macro;
    let kc = param.column_step_macro;

    for col in [0usize, 16usize] {
        let c0 = c.add(col);
        let c1 = c.add(ldc + col);
        let c2 = c.add(2 * ldc + col);

        let mut acc0 = load_c16(c0);
        let mut acc1 = load_c16(c1);
        let mut acc2 = load_c16(c2);

        let mut k = 0usize;
        while k + 1 < kc {
            let bvec = load_b_pair16(b_panel, k, col);
            let a0 = splat_pair(*a.add(k), *a.add(k + 1));
            let a1 = splat_pair(*a.add(lda + k), *a.add(lda + k + 1));
            let a2 = splat_pair(*a.add(2 * lda + k), *a.add(2 * lda + k + 1));

            acc0 = _mm512_dpbf16_ps(acc0, a0, bvec);
            acc1 = _mm512_dpbf16_ps(acc1, a1, bvec);
            acc2 = _mm512_dpbf16_ps(acc2, a2, bvec);

            k += 2;
        }

        store_c16(acc0, c0, a, b_panel, k, kc, col);
        store_c16(acc1, c1, a.add(lda), b_panel, k, kc, col);
        store_c16(acc2, c2, a.add(2 * lda), b_panel, k, kc, col);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bf16_matmul_block_uses_avx512bf16() {
        if !std::arch::is_x86_feature_detected!("avx512bf16") {
            eprintln!("skip: avx512bf16 not detected");
            return;
        }

        let params = MatMulParams {
            a_row_step_macro: 64,
            b_row_step_macro: 32,
            column_step_macro: 64,
            a_row_step_micro: 3,
            b_row_step_micro: 32,
        };

        let mut a = vec![Bf16::ZERO; 3 * 64];
        let mut b = vec![Bf16::ZERO; 64 * 32];
        let mut c = vec![Bf16::ZERO; 3 * 32];

        for i in 0..3 {
            for k in 0..64 {
                a[i * 64 + k] = Bf16::from_f32(((i + k) % 7) as f32 * 0.25);
            }
        }
        for k in 0..64 {
            for j in 0..32 {
                b[k * 32 + j] = Bf16::from_f32(((k + j) % 5) as f32 * 0.125);
            }
        }

        unsafe {
            matmul_block(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), &params);
        }

        for i in 0..3 {
            for j in 0..32 {
                let mut expected = 0.0f32;
                for k in 0..64 {
                    expected += a[i * 64 + k].to_f32() * b[k * 32 + j].to_f32();
                }
                assert_abs_diff_eq!(c[i * 32 + j].to_f32(), expected, epsilon = 0.25);
            }
        }
    }
}
