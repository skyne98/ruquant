//! GPTQ onto the W4A4 grid (per-row symmetric int4 codes in [-8, 7]), and W4A4 block packing.
//!
//! Same algorithm as accuracy/eval_w4a4.py's gptq_int4: per-row scale from the clip search, act order
//! (columns by decreasing Hessian diagonal), 1% dampening, blocks of 128 columns, the inverse Hessian's
//! upper Cholesky factor for the error feedback.

use faer::linalg::solvers::{DenseSolveCore, Llt};
use faer::{Mat, Side};
use rayon::prelude::*;

/// Per-row scale for codes in [-8, 7]: clip * amax / 7 with the clip ratio (1.0 down to 0.7 in 13 steps)
/// that minimizes the row's squared rounding error. Matches quantize_row_w4a4_ref in our ggml.
pub fn int4_scale(row: &[f32]) -> f32 {
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return 0.0;
    }
    let (mut best_s, mut best_err) = (0f32, f64::INFINITY);
    for r in 0..13 {
        let s = amax * (1.0 - 0.025 * r as f32) / 7.0;
        let err: f64 = row
            .iter()
            .map(|&x| {
                let e = ((x / s).round_ties_even().clamp(-8.0, 7.0) * s) as f64 - x as f64;
                e * e
            })
            .sum();
        if err < best_err {
            best_err = err;
            best_s = s;
        }
    }
    best_s
}

fn code(x: f32, s: f32) -> i8 {
    if s == 0.0 { 0 } else { (x / s).round_ties_even().clamp(-8.0, 7.0) as i8 }
}

/// GPTQ of `w` (n rows x k, row-major) against `h` (k x k, sum of x x^T over the calibration inputs), on the
/// clip-search scales, then `rounds` times with learned scales: each row's scale refit to its codes (least squares
/// under the dampened Hessian: s = c^T H w / c^T H c) and GPTQ again. Returns the codes (n x k) and per-row scales.
pub fn gptq_learned(w: &[f32], h: &[f32], k: usize, rounds: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
    let mut scales: Vec<f32> = w.par_chunks(k).map(int4_scale).collect();
    let per_block = |s: &[f32]| s.iter().flat_map(|&v| std::iter::repeat(v).take(k / QK)).collect::<Vec<f32>>();
    let mut codes = gptq_scaled(w, h, k, &per_block(&scales), QK)?;
    for _ in 0..rounds {
        scales = refit_scales(w, h, k, &codes, &scales);
        codes = gptq_scaled(w, h, k, &per_block(&scales), QK)?;
    }
    Ok((codes, scales))
}

/// GPTQ with a scale per group of `gs` inputs (256, or 64 with --wgroups 64), set before GPTQ from the group's clip
/// search (static groups, so act order stays): each an integer multiple k (1..15) of the row's unit v = the largest
/// group scale / 15, which the kernels fold exactly in int32. Returns the codes, the units (per row) and the
/// multipliers (per row per group).
pub fn gptq_groups(w: &[f32], h: &[f32], k: usize, gs: usize) -> Result<(Vec<i8>, Vec<f32>, Vec<u8>), String> {
    let nb = k / gs;
    let bs: Vec<f32> = w.par_chunks(gs).map(int4_scale).collect();
    let units: Vec<f32> = bs.chunks(nb).map(|r| r.iter().fold(0f32, |m, &v| m.max(v)) / 15.0).collect();
    let mults: Vec<u8> = bs
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let v = units[i / nb];
            if v > 0.0 { (s / v).round_ties_even().clamp(1.0, 15.0) as u8 } else { 1 }
        })
        .collect();
    let scales: Vec<f32> = mults.iter().enumerate().map(|(i, &m)| m as f32 * units[i / nb]).collect();
    Ok((gptq_scaled(w, h, k, &scales, gs)?, units, mults))
}

/// Per row, the scale s minimizing (w - s c)^T H (w - s c) for the codes c, with GPTQ's dampening; a row whose
/// fit is degenerate keeps its old scale.
fn refit_scales(w: &[f32], h: &[f32], k: usize, codes: &[i8], old: &[f32]) -> Vec<f32> {
    let n = w.len() / k;
    let mean_diag = (0..k).map(|i| h[i * k + i] as f64).sum::<f64>() / k as f64;
    let damp = (0.01 * mean_diag).max(1e-8) as f32;
    let hd: Vec<f32> = (0..k * k).map(|x| h[x] + if x / k == x % k { damp } else { 0.0 }).collect();
    let c: Vec<f32> = codes.par_iter().map(|&v| v as f32).collect();
    let mut ch = vec![0f32; n * k];
    unsafe {
        gemm::gemm(
            n, k, k,
            ch.as_mut_ptr(), 1, k as isize, false,
            c.as_ptr(), 1, k as isize,
            hd.as_ptr(), 1, k as isize,
            0.0f32, 1.0f32, false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }
    (0..n)
        .into_par_iter()
        .map(|r| {
            let (chr, cr, wr) = (&ch[r * k..(r + 1) * k], &c[r * k..(r + 1) * k], &w[r * k..(r + 1) * k]);
            let num: f64 = chr.iter().zip(wr).map(|(&a, &b)| a as f64 * b as f64).sum();
            let den: f64 = chr.iter().zip(cr).map(|(&a, &b)| a as f64 * b as f64).sum();
            let s = (num / den) as f32;
            if den > 0.0 && s.is_finite() && s > 0.0 { s } else { old[r] }
        })
        .collect()
}

/// GPTQ onto codes times the given scales, per row per group of `gs` inputs; returns the codes.
fn gptq_scaled(w: &[f32], h: &[f32], k: usize, scales: &[f32], gs: usize) -> Result<Vec<i8>, String> {
    let n = w.len() / k;
    let nb = k / gs;

    // act order + dead inputs + dampening
    let diag: Vec<f32> = (0..k).map(|i| h[i * k + i]).collect();
    let mut perm: Vec<usize> = (0..k).collect();
    perm.sort_by(|&a, &b| diag[b].partial_cmp(&diag[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mean_diag = diag.iter().map(|&d| d as f64).sum::<f64>() / k as f64;
    let damp = (0.01 * mean_diag).max(1e-8) as f32;
    let hp = Mat::<f32>::from_fn(k, k, |i, j| {
        let (pi, pj) = (perm[i], perm[j]);
        if diag[pi] == 0.0 || diag[pj] == 0.0 {
            return if i == j { 1.0 } else { 0.0 };
        }
        h[pi * k + pj] + if i == j { damp } else { 0.0 }
    });
    // U = upper Cholesky factor of H^-1
    let inv = Llt::new(hp.as_ref(), Side::Lower).map_err(|e| format!("Hessian not positive definite: {e:?}"))?.inverse();
    let l2 = Llt::new(inv.as_ref(), Side::Lower).map_err(|e| format!("inverse Hessian not positive definite: {e:?}"))?;
    let lf = l2.L();
    let u = Mat::<f32>::from_fn(k, k, |i, j| if j >= i { lf[(j, i)] } else { 0.0 });

    // permuted weights (dead inputs zeroed), worked on in place
    let mut wp: Vec<f32> = (0..n * k)
        .into_par_iter()
        .map(|x| {
            let (r, j) = (x / k, x % k);
            if diag[perm[j]] == 0.0 { 0.0 } else { w[r * k + perm[j]] }
        })
        .collect();
    let mut qp = vec![0i8; n * k];

    const B: usize = 128;
    let mut err = vec![0f32; n * B];
    for i1 in (0..k).step_by(B) {
        let i2 = (i1 + B).min(k);
        let b = i2 - i1;
        // within the block, rows are independent
        wp.par_chunks_mut(k)
            .zip(qp.par_chunks_mut(k))
            .zip(err.par_chunks_mut(B))
            .zip(scales.par_chunks(nb))
            .for_each(|(((row, qrow), erow), srow)| {
                for i in 0..b {
                    let x = row[i1 + i];
                    let s = srow[perm[i1 + i] / gs];
                    let c = code(x, s);
                    qrow[i1 + i] = c;
                    let e = (x - c as f32 * s) / u[(i1 + i, i1 + i)];
                    for j in i..b {
                        row[i1 + j] -= e * u[(i1 + i, i1 + j)];
                    }
                    erow[i] = e;
                }
            });
        // trailing columns: W[:, i2:] -= Err[:, :b] . U[i1:i2, i2:]
        if i2 < k {
            let rest = k - i2;
            let ublk = Mat::<f32>::from_fn(b, rest, |i, j| u[(i1 + i, i2 + j)]);
            unsafe {
                gemm::gemm(
                    n, rest, b,
                    wp.as_mut_ptr().add(i2), 1, k as isize, true,
                    err.as_ptr(), 1, B as isize,
                    ublk.as_ptr(), ublk.col_stride(), ublk.row_stride(),
                    1.0f32, -1.0f32, false, false, false,
                    gemm::Parallelism::Rayon(0),
                );
            }
        }
    }

    // undo the permutation
    let mut codes = vec![0i8; n * k];
    codes.par_chunks_mut(k).zip(qp.par_chunks(k)).for_each(|(dst, src)| {
        for (j, &c) in src.iter().enumerate() {
            dst[perm[j]] = c;
        }
    });
    Ok(codes)
}

const QK: usize = 256;

/// W4A4 / W4A4H blocks (ggml-common.h block_w4a4) with weight groups: per 256 codes a header (row unit, 1 << 30 |
/// the block's multiplier, row length, 0) and 32 words with element 8w+j in bits [4j+3:4j].
pub fn pack_groups(codes: &[i8], units: &[f32], mults: &[u8], k: usize) -> Vec<u8> {
    let mut out = pack(codes, units, k);
    let per = mults.len() / (out.len() / 144); // multipliers per block: 1 (groups of 256) or 4 (groups of 64)
    out.par_chunks_mut(144).zip(mults.par_chunks(per)).for_each(|(blk, m)| {
        // sum = 1 << 30 | k (RQ_W_MULT), or with groups of 64 1 << 30 | 1 << 29 | k0 | k1 << 4 | k2 << 8 | k3 << 12
        let v = if per == 1 {
            1i32 << 30 | m[0] as i32
        } else {
            m.iter().enumerate().fold(3i32 << 29, |a, (g, &k)| a | (k as i32) << (4 * g))
        };
        blk[4..8].copy_from_slice(&v.to_le_bytes());
    });
    out
}

/// W4A4 / W4A4H blocks (ggml-common.h block_w4a4): per 256 codes a header (row scale, row code sum,
/// row length, 0) and 32 words with element 8w+j in bits [4j+3:4j].
pub fn pack(codes: &[i8], scales: &[f32], k: usize) -> Vec<u8> {
    let row_bytes = k / QK * 144;
    let mut out = vec![0u8; scales.len() * row_bytes];
    out.par_chunks_mut(row_bytes).zip(codes.par_chunks(k)).zip(scales.par_iter()).for_each(|((dst, c), &s)| {
        let sum: i32 = c.iter().map(|&v| v as i32).sum();
        for (b, blk) in dst.chunks_mut(144).enumerate() {
            blk[0..4].copy_from_slice(&s.to_le_bytes());
            blk[4..8].copy_from_slice(&sum.to_le_bytes());
            blk[8..12].copy_from_slice(&(k as i32).to_le_bytes());
            blk[12..16].copy_from_slice(&0i32.to_le_bytes());
            for w in 0..QK / 8 {
                let mut word = 0u32;
                for j in 0..8 {
                    word |= ((c[b * QK + 8 * w + j] as u32) & 0xF) << (4 * j);
                }
                blk[16 + 4 * w..20 + 4 * w].copy_from_slice(&word.to_le_bytes());
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }

    #[test]
    fn gptq_beats_rounding_and_packs() {
        let (n, k, m) = (64usize, 512usize, 2048usize);
        let mut s = 0x9E3779B97F4A7C15u64;
        let w: Vec<f32> = (0..n * k).map(|_| rng(&mut s) * 0.02).collect();
        // inputs with uneven channel scales, as after a norm
        let x: Vec<f32> = (0..m * k).map(|i| rng(&mut s) * (0.1 + 3.0 * (i % k) as f32 / k as f32)).collect();
        let mut h = vec![0f32; k * k];
        for r in 0..m {
            let xr = &x[r * k..(r + 1) * k];
            for i in 0..k {
                for j in 0..k {
                    h[i * k + j] += xr[i] * xr[j];
                }
            }
        }
        let out_err = |q: &dyn Fn(usize, usize) -> f32| -> f64 {
            let (mut num, mut den) = (0f64, 0f64);
            for r in (0..m).step_by(16) {
                for o in 0..n {
                    let (mut a, mut b) = (0f64, 0f64);
                    for i in 0..k {
                        a += (x[r * k + i] * w[o * k + i]) as f64;
                        b += (x[r * k + i] * q(o, i)) as f64;
                    }
                    num += (a - b) * (a - b);
                    den += a * a;
                }
            }
            (num / den).sqrt()
        };
        let scales: Vec<f32> = w.chunks(k).map(int4_scale).collect();
        let rtn = out_err(&|o, i| code(w[o * k + i], scales[o]) as f32 * scales[o]);
        let (codes, gs) = gptq_learned(&w, &h, k, 0).unwrap();
        assert_eq!(gs, scales);
        assert!(codes.iter().all(|&c| (-8..=7).contains(&c)));
        let g = out_err(&|o, i| codes[o * k + i] as f32 * gs[o]);
        println!("output rel. error: rounding {rtn:.4}, gptq {g:.4}");
        assert!(g < rtn * 0.95, "gptq {g} not better than rounding {rtn}");

        let (lc, ls) = gptq_learned(&w, &h, k, 2).unwrap();
        let l = out_err(&|o, i| lc[o * k + i] as f32 * ls[o]);
        println!("output rel. error: gptq with learned scales {l:.4}");
        assert!(l < g * 1.01, "learned scales {l} worse than gptq {g}"); // random weights: little to gain

        // weight groups: blocks of different scale, as GPTQ's static groups with integer multipliers
        let wg: Vec<f32> = w.iter().enumerate().map(|(i, &v)| v * [0.25f32, 1.0, 4.0][(i % k / 256 + i / k) % 3]).collect();
        let row_err = |c: &[i8], s: &dyn Fn(usize, usize) -> f32| -> f64 {
            let (mut num, mut den) = (0f64, 0f64);
            for o in 0..n {
                for i in 0..k {
                    let d = (wg[o * k + i] - c[o * k + i] as f32 * s(o, i)) as f64;
                    num += d * d;
                    den += (wg[o * k + i] as f64).powi(2);
                }
            }
            (num / den).sqrt()
        };
        let (rc, rs) = gptq_learned(&wg, &h, k, 0).unwrap();
        let (gc, gu, gm) = gptq_groups(&wg, &h, k).unwrap();
        let (er, eg) = (row_err(&rc, &|o, _| rs[o]), row_err(&gc, &|o, i| gm[o * (k / 256) + i / 256] as f32 * gu[o]));
        println!("weight rel. error, blocks scaled x1/4..x4: per row {er:.4}, weight groups {eg:.4}");
        assert!(eg < er * 0.8, "weight groups {eg} not better than per row {er}");
        assert!(gm.iter().all(|&m| (1..=15).contains(&m)) && gm.contains(&15));
        let gp = pack_groups(&gc, &gu, &gm, k);
        let blk = &gp[3 * (k / 256) * 144 + 144..][..144]; // row 3, block 1
        assert_eq!(f32::from_le_bytes(blk[0..4].try_into().unwrap()), gu[3]);
        assert_eq!(i32::from_le_bytes(blk[4..8].try_into().unwrap()), 1 << 30 | gm[3 * (k / 256) + 1] as i32);
        // ggml's dequantizer reads the multipliers the same way
        let to_float = unsafe { (*crate::sys::ggml_get_type_traits(crate::gguf::W4A4)).to_float }.unwrap();
        let mut deq = vec![0f32; n * k];
        unsafe { to_float(gp.as_ptr() as *const _, deq.as_mut_ptr(), (n * k) as i64) };
        for i in (0..n * k).step_by(97) {
            let (o, j) = (i / k, i % k);
            assert_eq!(deq[i], gc[i] as f32 * (gm[o * (k / 256) + j / 256] as f32 * gu[o]), "element {i}");
        }

        let packed = pack(&codes, &gs, k);
        assert_eq!(packed.len(), n * (k / 256) * 144);
        for o in [0usize, 17, 63] {
            let blk = &packed[o * (k / 256) * 144 + 144..][..144]; // second block of the row
            assert_eq!(f32::from_le_bytes(blk[0..4].try_into().unwrap()), gs[o]);
            let sum: i32 = codes[o * k..(o + 1) * k].iter().map(|&c| c as i32).sum();
            assert_eq!(i32::from_le_bytes(blk[4..8].try_into().unwrap()), sum);
            for e in [0usize, 7, 8, 255] {
                let word = u32::from_le_bytes(blk[16 + 4 * (e / 8)..20 + 4 * (e / 8)].try_into().unwrap());
                let c = (((word >> (4 * (e % 8))) & 0xF) << 28) as i32 >> 28;
                assert_eq!(c as i8, codes[o * k + 256 + e]);
            }
        }
    }
}

/// GPTAQ's target (asymmetric calibration): w (n x k rows) moved to w + w (C - H) (H + damp)^-1, the least-squares
/// weight that maps the quantized model's inputs X~ (H = X~ X~^T, from ruquant hessian --fp) to the FP model's outputs
/// W X (C = X X~^T), dampened in the correction only (w C (H + damp)^-1 shrank w even where X~ = X). GPTQ under H then
/// rounds it, which minimizes |W X - Q X~| instead of |W X~ - Q X~|.
pub fn gptaq_shift(w: &mut [f32], h: &[f32], c: &[f32], k: usize) -> Result<(), String> {
    let diag: Vec<f32> = (0..k).map(|i| h[i * k + i]).collect();
    let mean_diag = diag.iter().map(|&d| d as f64).sum::<f64>() / k as f64;
    let damp = (0.01 * mean_diag).max(1e-8) as f32;
    let hd = Mat::<f32>::from_fn(k, k, |i, j| {
        if diag[i] == 0.0 || diag[j] == 0.0 {
            return if i == j { 1.0 } else { 0.0 };
        }
        h[i * k + j] + if i == j { damp } else { 0.0 }
    });
    let inv = Llt::new(hd.as_ref(), Side::Lower).map_err(|e| format!("Hessian not positive definite: {e:?}"))?.inverse();
    // M = I + (C - H) Hd^-1 (C row-major: c[i k + j] = sum_t x_i x~_j); dead inputs pass through
    let cr: Vec<f32> = (0..k * k)
        .into_par_iter()
        .map(|x| {
            let j = x % k;
            if diag[j] == 0.0 { 0.0 } else { c[x] - h[x] }
        })
        .collect();
    let ir: Vec<f32> = (0..k * k).into_par_iter().map(|x| inv[(x / k, x % k)]).collect();
    let mut mr = vec![0f32; k * k];
    unsafe {
        gemm::gemm(
            k, k, k,
            mr.as_mut_ptr(), 1, k as isize, false,
            cr.as_ptr(), 1, k as isize,
            ir.as_ptr(), 1, k as isize,
            0.0f32, 1.0f32, false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }
    // RQ_GPTAQ_ALPHA: the share of the correction applied (1 = the full least-squares target)
    let alpha: f32 = std::env::var("RQ_GPTAQ_ALPHA").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    mr.par_iter_mut().for_each(|v| *v *= alpha);
    for i in 0..k {
        mr[i * k + i] += 1.0;
    }
    let n = w.len() / k;
    let mut out = vec![0f32; w.len()];
    unsafe {
        gemm::gemm(
            n, k, k,
            out.as_mut_ptr(), 1, k as isize, false,
            w.as_ptr(), 1, k as isize,
            mr.as_ptr(), 1, k as isize,
            0.0f32, 1.0f32, false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }
    w.copy_from_slice(&out);
    Ok(())
}
