//! `ruquant convert IN.gguf OUT.gguf`: a Qwen3.5-family GGUF (any ggml types, e.g. unsloth's UD-Q8_K_XL)
//! into a rotated W4A4 GGUF that our llama.cpp fork runs as is.
//!
//! The residual stream is rotated by R1 = diag(random signs) H, H the ruquant Hadamard (ggml_rq_hadamard),
//! with the RMSNorm scales folded into the weights that read it:
//!   token_embd             E R1
//!   attn_q/k/v, attn_qkv, attn_gate, ssm_alpha/beta    W diag(attn_norm) R1       -> W4A4
//!   ffn_gate/up            W diag(post_attention_norm) R1                          -> W4A4
//!   attn_output, ssm_out, ffn_down                     R1^T W H_in                 -> W4A4H (x H_in online)
//!   output                 W diag(output_norm) R1     (kept in its own type)
//!   the three norms        ones
//! The attention output gate and DeltaNet's z-gate sit in front of attn_output / ssm_out, so their input
//! rotation stays online (W4A4H), as for ffn_down; ggml's W4A4H quantizer applies H_in to the rows it
//! stores. Weights are rounded per row by ggml's W4A4 reference quantizer (clip search). Everything else,
//! including the unused MTP layer, is copied unchanged.
//!
//! `--ost DIR` takes R1 and OSTQuant's smoothing scales from accuracy/ostquant_qwen.py: DIR/r1.f32 is R1, and
//! per layer s_in / s_post divide the norm output (norm weight 1/s, readers W diag(norm) R1 diag(s)), s_ud
//! divides ffn_up's rows and multiplies ffn_down's columns, s_ov divides attn_v's rows and multiplies
//! attn_output's columns (each KV head's scale over its query heads).

use crate::gguf::{self, Gguf, TensorInfo, Writer};
use crate::gptq;
use crate::sys;
use rayon::prelude::*;
use std::time::Instant;

unsafe extern "C" {
    /// From ggml-quants.c (our fork): x <- x H, or x H^T with transpose.
    fn ggml_rq_hadamard(x: *mut f32, n: i64, transpose: bool);
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Copy,
    Ones,
    Embed,
    Head,
    ReadIn(usize),
    ReadPost(usize),
    Write(usize),
}

const READ_IN: [&str; 7] = ["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"];
const READ_POST: [&str; 2] = ["ffn_gate", "ffn_up"];
const WRITE: [&str; 3] = ["attn_output", "ssm_out", "ffn_down"];

fn role(name: &str, n_main: usize) -> Role {
    match name {
        "token_embd.weight" => return Role::Embed,
        "output.weight" => return Role::Head,
        "output_norm.weight" => return Role::Ones,
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else { return Role::Copy };
    let Some((il, tail)) = rest.split_once('.') else { return Role::Copy };
    let Ok(il) = il.parse::<usize>() else { return Role::Copy };
    if il >= n_main {
        return Role::Copy; // MTP layer: llama.cpp does not run it yet
    }
    let Some(base) = tail.strip_suffix(".weight") else { return Role::Copy };
    if base == "attn_norm" || base == "post_attention_norm" {
        Role::Ones
    } else if READ_IN.contains(&base) {
        Role::ReadIn(il)
    } else if READ_POST.contains(&base) {
        Role::ReadPost(il)
    } else if WRITE.contains(&base) {
        Role::Write(il)
    } else {
        Role::Copy
    }
}

/// splitmix64, for reproducible signs
fn signs(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            if (z ^ (z >> 31)) & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

fn hadamard(x: &mut [f32]) {
    unsafe { ggml_rq_hadamard(x.as_mut_ptr(), x.len() as i64, false) }
}

/// x <- x diag(H_n, H_n, ...): the online rotation of a W4A4H row in blocks of n (n = the row length: one H)
fn hadamard_blocks(x: &mut [f32], n: usize) {
    x.chunks_mut(n).for_each(hadamard);
}

/// The Hadamard length a W4A4H tensor's kernels read, in every block header (pad[0], bytes 8..12 of 144)
fn set_rot_len(data: &mut [u8], n: usize) {
    data.chunks_exact_mut(144).for_each(|b| b[8..12].copy_from_slice(&(n as i32).to_le_bytes()));
}

/// A Hessian of x H (ruquant hessian rotates W4A4H inputs by the whole-row H) in the basis of the block rotation
/// x diag(H_n, ...): B^T S B with B = H^T diag(H_n, ...), as rows of S B, a transpose, and rows again (S symmetric).
fn rebase_hessian(h: &[f32], k: usize, n: usize) -> Vec<f32> {
    let rows = |m: &mut [f32]| {
        m.par_chunks_mut(k).for_each(|r| {
            unsafe { ggml_rq_hadamard(r.as_mut_ptr(), k as i64, true) };
            hadamard_blocks(r, n);
        })
    };
    let mut t = h.to_vec();
    rows(&mut t);
    let mut u = vec![0f32; k * k];
    u.par_chunks_mut(k).enumerate().for_each(|(j, col)| col.iter_mut().enumerate().for_each(|(i, v)| *v = t[i * k + j]));
    rows(&mut u);
    u
}

/// --rot-blocks: W4A4H rows rotated in `blocks` Hadamard blocks of k / blocks, so a tensor-parallel split of K
/// (-sm tensor, up to `blocks` ways) keeps whole blocks on each device
fn rot_len(k: usize, blocks: usize, name: &str) -> Result<usize, String> {
    let n = k / blocks;
    if k % blocks != 0 || n % 256 != 0 {
        return Err(format!("{name}: {k} inputs don't split into {blocks} Hadamard blocks of whole 256-blocks"));
    }
    Ok(n)
}

/// Every row of `t` as f32, through ggml's own dequantizer for its type.
fn dequantize(g: &Gguf, t: &TensorInfo) -> Result<Vec<f32>, String> {
    let k = t.dims[0] as usize;
    let rows = t.n_elements() as usize / k;
    let data = g.data(t)?;
    if t.ty == gguf::F32 {
        return Ok(data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
    }
    let row_bytes = data.len() / rows;
    let mut out = vec![0f32; rows * k];
    let to_float = unsafe { (*sys::ggml_get_type_traits(t.ty)).to_float }
        .ok_or(format!("{}: ggml has no dequantizer for type {}", t.name, t.ty))?;
    out.par_chunks_mut(k * 64).enumerate().for_each(|(c, dst)| {
        let r0 = c * 64;
        let n = dst.len() / k;
        let src = &data[r0 * row_bytes..(r0 + n) * row_bytes];
        unsafe { to_float(src.as_ptr() as *const _, dst.as_mut_ptr(), (n * k) as i64) };
    });
    Ok(out)
}

/// Rows of f32 into `ty` with ggml_quantize_chunk (W4A4 / W4A4H use the reference clip search).
fn quantize(x: &[f32], k: usize, ty: u32) -> Vec<u8> {
    let rows = x.len() / k;
    let row_size = unsafe { sys::ggml_row_size(ty, k as i64) };
    let mut out = vec![0u8; rows * row_size];
    out.par_chunks_mut(row_size * 16).enumerate().for_each(|(c, dst)| {
        let r0 = c * 16;
        let n = dst.len() / row_size;
        unsafe {
            sys::ggml_quantize_chunk(ty, x[r0 * k..].as_ptr(), dst.as_mut_ptr() as *mut _, 0, n as i64, k as i64, std::ptr::null())
        };
    });
    out
}

pub fn main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut seed = 1u64;
    let mut rotate = true;
    let mut keep: Vec<String> = Vec::new();
    let mut hdir: Option<String> = None;
    let mut refit = 0usize; // --refit: rounds of learned GPTQ scales
    let mut wgroups = 0usize; // --wgroups [64]: GPTQ with a scale multiplier per block of 256 (or per 64 inputs)
    let mut rot_blocks = 1usize; // --rot-blocks: block-diagonal online Hadamard on the W4A4H tensors
    let mut klt = false;
    let mut r1_file: Option<String> = None;
    let mut types_from: Option<String> = None;
    let mut ost: Option<String> = None;
    let mut head_w4a4 = false;
    let mut sets: Vec<(String, u32)> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => seed = it.next().and_then(|s| s.parse().ok()).ok_or("--seed needs a number")?,
            "--no-rotate" => rotate = false,
            "--head" => match it.next().map(String::as_str) {
                Some("w4a4") => head_w4a4 = true,
                Some("keep") => head_w4a4 = false,
                _ => return Err("--head is w4a4 or keep".into()),
            },
            "--set" => {
                let v = it.next().ok_or("--set needs NAME=TYPE")?;
                let (name, ty) = v.split_once('=').ok_or("--set needs NAME=TYPE")?;
                let ty = match ty {
                    "q8_0" => gguf::Q8_0,
                    "w4a4" => gguf::W4A4,
                    "w4a4h" => gguf::W4A4H,
                    _ => return Err(format!("--set: type {ty} is q8_0, w4a4 or w4a4h")),
                };
                sets.push((name.to_string(), ty));
            }
            "--hessians" => hdir = Some(it.next().ok_or("--hessians needs a directory")?.clone()),
            "--refit" => refit = it.next().and_then(|v| v.parse().ok()).ok_or("--refit needs a count")?,
            "--wgroups" => {
                wgroups = 256;
                if it.clone().next().is_some_and(|v| v == "64") {
                    it.next();
                    wgroups = 64;
                }
            }
            "--rot-blocks" => rot_blocks = it.next().and_then(|v| v.parse().ok()).filter(|&b| b >= 1).ok_or("--rot-blocks needs a count")?,
            "--r1-file" => r1_file = Some(it.next().ok_or("--r1-file needs a path")?.clone()),
            "--types-from" => types_from = Some(it.next().ok_or("--types-from needs a GGUF")?.clone()),
            "--ost" => {
                let d = it.next().ok_or("--ost needs a directory")?.clone();
                r1_file = Some(format!("{d}/r1.f32"));
                ost = Some(d);
            }
            "--r1" => match it.next().map(String::as_str) {
                Some("klt") => klt = true,
                Some("hadamard") => klt = false,
                _ => return Err("--r1 is klt or hadamard".into()),
            },
            "--keep" => {
                let list = it.next().ok_or("--keep needs a comma list")?;
                keep.extend(list.split(',').filter(|k| !k.is_empty()).map(|k| k.to_string()));
            }
            _ => pos.push(a.clone()),
        }
    }
    let [input, output] = &pos[..] else {
        return Err("usage: ruquant convert IN.gguf OUT.gguf [--seed N] [--no-rotate] [--keep NAME,...] [--hessians DIR] [--r1 hadamard|klt]\n  \
                    --r1 klt: R1 = V diag(signs) H, V the eigenvectors of the readers' weight covariance (OSTQuant's start)\n  \
                    --r1-file: R1 read from a raw f32 n x n file (y = x R1), e.g. from accuracy/eval_w4a4.py --learn-r1\n  \
                    --ost DIR: R1 and smoothing scales from accuracy/ostquant_qwen.py\n  \
                    --head w4a4: the lm_head (rotated, its input is the R1 stream) as W4A4 too, rounded\n  \
                    --set NAME=TYPE: the tensor NAME (output.weight) or every layer's NAME (ssm_out) gets TYPE (q8_0,\n  \
                    w4a4, w4a4h; w4a4h only for\n  \
                    writers), over --types-from, e.g. --set ssm_out=q8_0\n  \
                    --types-from: another quant of the same model (e.g. unsloth's Q4_0) picks the types: its 4-bit linears\n  \
                    (q4_0, q4_1) become W4A4 / W4A4H, everything else gets that file's type, re-quantized from IN after rotation\n  \
                    --keep: linear tensors whose name contains one of these stay Q8_0 (still rotated), e.g. ffn_down,blk.0.\n  \
                    --hessians: GPTQ instead of rounding wherever DIR (from ruquant hessian) has the input's Hessian\n  \
                    --refit N: GPTQ with learned scales, N rounds of a least-squares scale refit + GPTQ again\n  \
                    --wgroups [64]: GPTQ with a scale per 256 inputs (or per 64: needs activation groups of 64), an\n  \
                    integer multiple (1..15) of the row's unit\n  \
                    --rot-blocks B: W4A4H inputs rotated in B Hadamard blocks, not one, so -sm tensor up to B ways can split\n  \
                    them (the Hessians are rebased exactly)".into());
    };
    let t0 = Instant::now();
    let g = Gguf::open(input)?;
    let arch = g.string("general.architecture").ok_or("no general.architecture")?;
    let n_layer = g.u32(&format!("{arch}.block_count")).ok_or("no block_count")? as usize;
    let n_mtp = g.u32(&format!("{arch}.nextn_predict_layers")).unwrap_or(0) as usize;
    let n_main = n_layer - n_mtp;
    let n_embd = g.u32(&format!("{arch}.embedding_length")).ok_or("no embedding_length")? as usize;
    println!("{input}: {arch}, {n_main} layers (+{n_mtp} MTP, copied), hidden {n_embd}, {} tensors", g.tensors.len());

    let norm = |name: &str| -> Result<Vec<f32>, String> {
        let t = g.tensor(name).ok_or(format!("missing {name}"))?;
        dequantize(&g, t)
    };
    // OSTQuant smoothing scale `kind` of layer il, if DIR has it
    let ost_scale = |il: usize, kind: &str| -> Result<Option<Vec<f32>>, String> {
        let Some(d) = &ost else { return Ok(None) };
        let p = format!("{d}/blk.{il}.{kind}.f32");
        if !std::path::Path::new(&p).exists() {
            return Ok(None);
        }
        let raw = std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?;
        Ok(Some(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    };
    // R1 = [V] diag(signs) H on every row of a tensor (rows of length n_embd)
    let d = signs(n_embd, seed);
    let dense_only = r1_file.is_some(); // the file is the whole R1: no signs, no Hadamard after it
    let v = if let Some(f) = &r1_file {
        let raw = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        if raw.len() != n_embd * n_embd * 4 {
            return Err(format!("{f}: {} bytes, expected {n_embd} x {n_embd} f32", raw.len()));
        }
        Some(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<f32>>())
    } else if rotate && klt {
        Some(klt_basis(&g, n_main, n_embd, &norm)?)
    } else {
        None
    };
    let r1 = |x: &mut [f32]| {
        if !rotate {
            return;
        }
        if let Some(v) = &v {
            let rows = x.len() / n_embd;
            let mut y = vec![0f32; x.len()];
            unsafe {
                gemm::gemm(
                    rows, n_embd, n_embd,
                    y.as_mut_ptr(), 1, n_embd as isize, false,
                    x.as_ptr(), 1, n_embd as isize,
                    v.as_ptr(), 1, n_embd as isize,
                    0.0f32, 1.0f32, false, false, false,
                    gemm::Parallelism::Rayon(0),
                );
            }
            x.copy_from_slice(&y);
            if dense_only {
                return;
            }
        }
        x.par_chunks_mut(n_embd).for_each(|r| {
            r.iter_mut().zip(&d).for_each(|(v, s)| *v *= s);
            hadamard(r);
        });
    };

    // Tied embeddings (the small Qwen3.5s): the head reads the final norm's output, whose scale has to go into the
    // head's rows, so the shared matrix is untied: output.weight = E diag(output_norm) R1 from token_embd's data,
    // Q8_0 unless --set names a type. llama.cpp uses output.weight whenever the file has one.
    let mut src: Vec<TensorInfo> = g.tensors.clone();
    let untie = g.tensor("output.weight").is_none() && rotate;
    if untie {
        let e = g.tensor("token_embd.weight").ok_or("no token_embd.weight")?;
        src.push(TensorInfo { name: "output.weight".into(), dims: e.dims.clone(), ty: e.ty, offset: e.offset });
    }
    // output layout: same tensors, same order, new types
    let map = match &types_from {
        Some(p) => Some(Gguf::open(p)?),
        None => None,
    };
    let mut infos = Vec::new();
    for t in &src {
        let kept = keep.iter().any(|k| t.name.contains(k.as_str()));
        let synthetic = untie && t.name == "output.weight";
        let mapped = map
            .as_ref()
            .filter(|_| !synthetic)
            .map(|m| m.tensor(&t.name).map(|mt| mt.ty).ok_or(format!("{}: not in --types-from", t.name)));
        let mapped = mapped.transpose()?.or(if synthetic { Some(gguf::Q8_0) } else { None });
        let four_bit = matches!(mapped, Some(2 | 3)); // q4_0, q4_1
        let linear = matches!(role(&t.name, n_main), Role::ReadIn(_) | Role::ReadPost(_) | Role::Write(_));
        // NAME is a whole tensor name (output.weight) or a layer tensor's base name (ssm_out)
        let set = sets
            .iter()
            .find(|(n, _)| t.name == *n || t.name.split('.').nth(2) == Some(n.as_str()))
            .map(|&(_, ty)| ty);
        let ty = match role(&t.name, n_main) {
            _ if linear && set.is_some() => {
                let ty = set.unwrap();
                if ty == gguf::W4A4H && !matches!(role(&t.name, n_main), Role::Write(_)) {
                    return Err(format!("--set: {} reads the residual stream, it can't be w4a4h", t.name));
                }
                ty
            }
            _ if linear && mapped.is_some() && !four_bit => mapped.unwrap(),
            Role::ReadIn(_) | Role::ReadPost(_) | Role::Write(_) if kept => gguf::Q8_0,
            Role::ReadIn(_) | Role::ReadPost(_) => gguf::W4A4,
            Role::Write(_) => if rotate { gguf::W4A4H } else { gguf::W4A4 },
            Role::Ones => gguf::F32,
            Role::Head if head_w4a4 && rotate => gguf::W4A4,
            Role::Head if set.is_some_and(|ty| ty != gguf::W4A4H) => set.unwrap(),
            Role::Embed | Role::Head => mapped.unwrap_or(t.ty),
            _ => t.ty,
        };
        infos.push(TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty, offset: 0 });
    }
    let mut kvs: Vec<gguf::Kv> = g.kvs.iter().map(|k| gguf::Kv { key: k.key.clone(), raw: k.raw.clone() }).collect();
    kvs.push(gguf::kv_string(
        "ruquant.w4a4",
        &if rotate {
            format!(
                "R1 = {}; online H ({rot_blocks} block(s) per row) on attn_output, ssm_out, ffn_down; kept Q8_0: {keep:?}; GPTQ: {}",
                match &r1_file {
                    Some(f) => format!("dense from {f}{}", if ost.is_some() { ", OSTQuant smoothing" } else { "" }),
                    None => format!("{}diag(splitmix64 signs, seed {seed}) H_{n_embd}", if klt { "V_klt " } else { "" }),
                },
                hdir.is_some()
            )
        } else {
            "no rotation".to_string()
        },
    ));
    let mut w = Writer::create(output, &kvs, &infos, g.alignment())?;

    let (mut in_bytes, mut out_bytes) = (0usize, 0usize);
    for (t, info) in src.iter().zip(&infos) {
        let role = role(&t.name, n_main);
        let base = t.name.split('.').nth(2).unwrap_or("");
        let k = t.dims[0] as usize;
        let data: Vec<u8> = match role {
            Role::Copy => g.data(t)?.to_vec(),
            Role::Ones => {
                let kind = match base {
                    "attn_norm" => Some("s_in"),
                    "post_attention_norm" => Some("s_post"),
                    _ => None,
                };
                let s = match (kind, il_of(&t.name)) {
                    (Some(kind), Some(il)) => ost_scale(il, kind)?,
                    _ => None,
                };
                (0..t.n_elements() as usize)
                    .map(|j| s.as_ref().map_or(1.0, |s| 1.0 / s[j]))
                    .flat_map(|v: f32| v.to_le_bytes())
                    .collect()
            }
            _ => {
                let mut x = dequantize(&g, t)?;
                match role {
                    Role::Embed => r1(&mut x),
                    Role::Head | Role::ReadIn(_) | Role::ReadPost(_) => {
                        let s = match role {
                            Role::Head => norm("output_norm.weight")?,
                            Role::ReadIn(il) => norm(&format!("blk.{il}.attn_norm.weight"))?,
                            Role::ReadPost(il) => norm(&format!("blk.{il}.post_attention_norm.weight"))?,
                            _ => unreachable!(),
                        };
                        x.par_chunks_mut(k).for_each(|r| r.iter_mut().zip(&s).for_each(|(v, s)| *v *= s));
                        r1(&mut x);
                        if let Role::ReadIn(il) | Role::ReadPost(il) = role {
                            let kind = if matches!(role, Role::ReadIn(_)) { "s_in" } else { "s_post" };
                            if let Some(s) = ost_scale(il, kind)? {
                                x.par_chunks_mut(k).for_each(|r| r.iter_mut().zip(&s).for_each(|(v, s)| *v *= s));
                            }
                            let rows_kind = match base {
                                "ffn_up" => Some("s_ud"),
                                "attn_v" => Some("s_ov"),
                                _ => None,
                            };
                            if let Some(s) = match rows_kind { Some(kind) => ost_scale(il, kind)?, None => None } {
                                if s.len() != x.len() / k {
                                    return Err(format!("{}: {} rows, {} scales", t.name, x.len() / k, s.len()));
                                }
                                x.par_chunks_mut(k).zip(s.par_iter()).for_each(|(r, s)| r.iter_mut().for_each(|v| *v /= s));
                            }
                        }
                    }
                    Role::Write(il) if rotate => {
                        let cols_kind = match base {
                            "ffn_down" => Some("s_ud"),
                            "attn_output" => Some("s_ov"),
                            _ => None,
                        };
                        if let Some(s) = match cols_kind { Some(kind) => ost_scale(il, kind)?, None => None } {
                            // attn_output's columns are per query head: query head h reads KV head h / groups
                            let hd = g.tensor(&format!("blk.{il}.attn_k_norm.weight")).map_or(k, |t| t.dims[0] as usize);
                            let groups = k / s.len();
                            let s: Vec<f32> = (0..k).map(|c| s[(c / hd / groups) * hd + c % hd]).collect();
                            x.par_chunks_mut(k).for_each(|r| r.iter_mut().zip(&s).for_each(|(v, s)| *v *= s));
                        }
                        // R1^T W through the transpose; the W4A4H quantizer then stores Q((R1^T W) H_in)
                        let n = x.len() / k;
                        let mut tr = vec![0f32; x.len()];
                        tr.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
                            for (i, v) in col.iter_mut().enumerate() {
                                *v = x[i * k + j];
                            }
                        });
                        r1(&mut tr);
                        x.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                            for (j, v) in row.iter_mut().enumerate() {
                                *v = tr[j * n + i];
                            }
                        });
                    }
                    _ => {}
                }
                let hess = hdir.as_deref().and_then(|d| hessian_path(d, &t.name, role));
                let rot = if info.ty == gguf::W4A4H { rot_len(k, rot_blocks, &t.name)? } else { k };
                let mut q = match hess {
                    Some(path) if info.ty == gguf::W4A4 || info.ty == gguf::W4A4H => {
                        if info.ty == gguf::W4A4H {
                            x.par_chunks_mut(k).for_each(|r| hadamard_blocks(r, rot)); // the stored rows are W H
                        }
                        let raw = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
                        if raw.len() != k * k * 4 {
                            return Err(format!("{path}: {} bytes, expected a {k} x {k} f32 Hessian", raw.len()));
                        }
                        let mut h: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                        if info.ty == gguf::W4A4H && rot != k {
                            h = rebase_hessian(&h, k, rot);
                        }
                        // GPTAQ: with the FP cross term (ruquant hessian --fp) next to H, fit W C H^-1 first
                        let cpath = path.replace(".f32", ".c.f32");
                        if std::path::Path::new(&cpath).exists() {
                            let raw = std::fs::read(&cpath).map_err(|e| format!("{cpath}: {e}"))?;
                            let mut c: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                            if info.ty == gguf::W4A4H && rot != k {
                                // rebase_hessian(A) is B^T A^T B: transposed in, so C comes out as B^T C B
                                let ct: Vec<f32> = (0..k * k).into_par_iter().map(|x| c[(x % k) * k + x / k]).collect();
                                c = rebase_hessian(&ct, k, rot);
                            }
                            gptq::gptaq_shift(&mut x, &h, &c, k).map_err(|e| format!("{}: {e}", t.name))?;
                        }
                        if wgroups > 0 {
                            let (codes, units, mults) = gptq::gptq_groups(&x, &h, k, wgroups).map_err(|e| format!("{}: {e}", t.name))?;
                            gptq::pack_groups(&codes, &units, &mults, k)
                        } else {
                            let (codes, scales) = gptq::gptq_learned(&x, &h, k, refit).map_err(|e| format!("{}: {e}", t.name))?;
                            gptq::pack(&codes, &scales, k)
                        }
                    }
                    _ if info.ty == gguf::W4A4H && rot != k => {
                        x.par_chunks_mut(k).for_each(|r| hadamard_blocks(r, rot));
                        quantize(&x, k, gguf::W4A4) // same blocks; the rotation is ours
                    }
                    _ => quantize(&x, k, info.ty),
                };
                if info.ty == gguf::W4A4H {
                    set_rot_len(&mut q, rot);
                }
                q
            }
        };
        if data.len() != info.n_bytes()? {
            return Err(format!("{}: produced {} bytes, expected {}", t.name, data.len(), info.n_bytes()?));
        }
        in_bytes += t.n_bytes()?;
        out_bytes += data.len();
        w.tensor(&data)?;
        if !matches!(role, Role::Copy | Role::Ones) {
            println!("  {:<34} {:>6} -> {:<6} {:6.1} s", t.name, type_name(t.ty), type_name(info.ty), t0.elapsed().as_secs_f32());
        }
    }
    w.finish()?;
    println!(
        "{output}: {:.2} GB of tensors (from {:.2} GB) in {:.0} s",
        out_bytes as f64 / 1e9,
        in_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

/// OSTQuant's KLT basis: eigenvectors (columns, row-major n x n) of the covariance of every reader row
/// (norm scales folded in), over all main layers.
fn klt_basis(g: &Gguf, n_main: usize, n: usize, norm: &dyn Fn(&str) -> Result<Vec<f32>, String>) -> Result<Vec<f32>, String> {
    let t0 = Instant::now();
    let mut s = vec![0f32; n * n];
    let mut mean = vec![0f64; n];
    let mut rows = 0usize;
    for t in &g.tensors {
        let r = role(&t.name, n_main);
        let scale = match r {
            Role::ReadIn(il) => norm(&format!("blk.{il}.attn_norm.weight"))?,
            Role::ReadPost(il) => norm(&format!("blk.{il}.post_attention_norm.weight"))?,
            _ => continue,
        };
        let mut x = dequantize(g, t)?;
        x.par_chunks_mut(n).for_each(|r| r.iter_mut().zip(&scale).for_each(|(v, s)| *v *= s));
        let m = x.len() / n;
        for r in x.chunks(n) {
            mean.iter_mut().zip(r).for_each(|(a, &b)| *a += b as f64);
        }
        rows += m;
        unsafe {
            gemm::gemm(
                n, n, m,
                s.as_mut_ptr(), 1, n as isize, true,
                x.as_ptr(), n as isize, 1,
                x.as_ptr(), 1, n as isize,
                1.0f32, 1.0f32, false, false, false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }
    mean.iter_mut().for_each(|m| *m /= rows as f64);
    let cov = faer::Mat::<f64>::from_fn(n, n, |i, j| (s[i * n + j] as f64 - rows as f64 * mean[i] * mean[j]) / (rows - 1) as f64);
    let eig = cov.self_adjoint_eigen(faer::Side::Lower).map_err(|e| format!("KLT eigendecomposition failed: {e:?}"))?;
    let u = eig.U();
    println!("  KLT basis from {rows} reader rows in {:.0} s", t0.elapsed().as_secs_f32());
    Ok((0..n * n).map(|x| u[(x / n, x % n)] as f32).collect())
}

fn il_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// The Hessian ruquant hessian wrote for this weight's input, if any.
fn hessian_path(dir: &str, name: &str, role: Role) -> Option<String> {
    let base = name.split('.').nth(2)?;
    let (il, kind) = match role {
        Role::ReadIn(il) => (il, "in"),
        Role::ReadPost(il) => (il, "post"),
        Role::Write(il) if base == "ffn_down" => (il, "down"),
        Role::Write(il) => (il, "o"),
        _ => return None,
    };
    let path = format!("{dir}/blk.{il}.{kind}.f32");
    std::path::Path::new(&path).exists().then_some(path)
}

fn type_name(t: u32) -> String {
    let p = unsafe { sys::ggml_type_name(t) };
    if p.is_null() {
        return t.to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// `ruquant drafter DRAFT.gguf TARGET_SRC.gguf OUT.gguf [--ost DIR | --seed N]`: a DFlash / DSpark drafter
/// for a target converted by `ruquant convert` with the same R1.
///
/// The drafter reads the target's residual stream at a few layers (rotated: h R1) through fc, and borrows
/// the target's token embedding and lm_head, which are rotated (E R1, W diag(norm) R1) in our target. So
/// every 5120-wide input chunk of fc gets R1 folded in (W R1, as a reader), and the drafter gets its own
/// unrotated token_embd and output, copied as they are from TARGET_SRC (the unrotated model the target was
/// converted from, e.g. unsloth's Q4_0). The drafter's own stream stays unrotated.
pub fn drafter_main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut seed = 1u64;
    let mut ost: Option<String> = None;
    let mut head_w4a4h = false;
    let mut lin_w4a4h = false;
    let mut fc_w4a4 = false;
    let mut rot_blocks = 1usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => seed = it.next().and_then(|s| s.parse().ok()).ok_or("--seed needs a number")?,
            "--ost" => ost = Some(it.next().ok_or("--ost needs a directory")?.clone()),
            "--rot-blocks" => rot_blocks = it.next().and_then(|v| v.parse().ok()).filter(|&b| b >= 1).ok_or("--rot-blocks needs a count")?,
            "--fc" => match it.next().map(String::as_str) {
                Some("w4a4") => fc_w4a4 = true,
                Some("keep") => fc_w4a4 = false,
                _ => return Err("--fc is w4a4 or keep".into()),
            },
            "--linears" => match it.next().map(String::as_str) {
                Some("w4a4h") => lin_w4a4h = true,
                Some("keep") => lin_w4a4h = false,
                _ => return Err("--linears is w4a4h or keep".into()),
            },
            "--head" => match it.next().map(String::as_str) {
                Some("w4a4h") => head_w4a4h = true,
                Some("keep") => head_w4a4h = false,
                _ => return Err("--head is w4a4h or keep".into()),
            },
            _ => pos.push(a.clone()),
        }
    }
    let [draft, src, output] = &pos[..] else {
        return Err("usage: ruquant drafter DRAFT.gguf TARGET_SRC.gguf OUT.gguf [--ost DIR | --seed N] [--head w4a4h]\n  \
                    --head w4a4h: the drafter's own lm_head as W4A4H (online Hadamard on its unrotated input)\n  \
                    --linears w4a4h: the drafter layers' linears as W4A4H too\n  \
                    --rot-blocks B: attn_output / ffn_down rotated in B Hadamard blocks (-sm tensor up to B ways)\n  \
                    --fc w4a4: fc as W4A4, no online Hadamard (its 25600-wide input does not factor; the input is the
                    target's rotated residual stream)".into());
    };
    let t0 = Instant::now();
    let g = Gguf::open(draft)?;
    let s = Gguf::open(src)?;
    let arch = g.string("general.architecture").ok_or("no general.architecture")?;
    let n_embd = g.u32(&format!("{arch}.embedding_length")).ok_or("no embedding_length")? as usize;
    let v: Option<Vec<f32>> = match &ost {
        Some(d) => {
            let f = format!("{d}/r1.f32");
            let raw = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
            if raw.len() != n_embd * n_embd * 4 {
                return Err(format!("{f}: {} bytes, expected {n_embd} x {n_embd} f32", raw.len()));
            }
            Some(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        }
        None => None,
    };
    let d = signs(n_embd, seed);
    // rows of n_embd: x <- x R1
    let r1 = |x: &mut [f32]| {
        if let Some(v) = &v {
            let rows = x.len() / n_embd;
            let mut y = vec![0f32; x.len()];
            unsafe {
                gemm::gemm(
                    rows, n_embd, n_embd,
                    y.as_mut_ptr(), 1, n_embd as isize, false,
                    x.as_ptr(), 1, n_embd as isize,
                    v.as_ptr(), 1, n_embd as isize,
                    0.0f32, 1.0f32, false, false, false,
                    gemm::Parallelism::Rayon(0),
                );
            }
            x.copy_from_slice(&y);
        } else {
            x.par_chunks_mut(n_embd).for_each(|r| {
                r.iter_mut().zip(&d).for_each(|(v, s)| *v *= s);
                hadamard(r);
            });
        }
    };

    // the layers' linears, W4A4H with --linears w4a4h (rows a multiple of 64, inputs the online Hadamard can factor)
    let lin = |t: &TensorInfo| -> bool {
        const NAMES: [&str; 9] = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "attn_conv_proj", "ffn_conv_proj"];
        lin_w4a4h
            && t.name.starts_with("blk.")
            && t.name.split('.').nth(2).is_some_and(|b| NAMES.contains(&b))
            && t.dims.len() == 2
            && t.dims[1] % 64 == 0
            && matches!(t.dims[0], 4096 | 5120 | 6144 | 17408)
    };
    let mut infos: Vec<TensorInfo> = g
        .tensors
        .iter()
        .map(|t| {
            let ty = if lin(t) {
                gguf::W4A4H
            } else if t.name == "fc.weight" && fc_w4a4 {
                gguf::W4A4
            } else {
                t.ty
            };
            TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty, offset: 0 }
        })
        .collect();
    let mut extra = Vec::new();
    for name in ["token_embd.weight", "output.weight"] {
        if g.tensor(name).is_some() {
            return Err(format!("{draft} already has {name}"));
        }
        let t = s.tensor(name).ok_or(format!("{src}: no {name}"))?;
        let ty = if name == "output.weight" && head_w4a4h { gguf::W4A4H } else { t.ty };
        infos.push(TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty, offset: 0 });
        extra.push(t);
    }
    let mut kvs: Vec<gguf::Kv> = g.kvs.iter().map(|k| gguf::Kv { key: k.key.clone(), raw: k.raw.clone() }).collect();
    kvs.push(gguf::kv_string(
        "ruquant.drafter",
        &format!("fc inputs x R1 ({}); token_embd, output unrotated from {src}",
                 if ost.is_some() { "OSTQuant" } else { "signs + Hadamard" }),
    ));
    let mut w = Writer::create(output, &kvs, &infos, g.alignment())?;
    for t in &g.tensors {
        if t.name == "fc.weight" {
            let k = t.dims[0] as usize;
            if k % n_embd != 0 {
                return Err(format!("fc.weight: input {k} is not a multiple of {n_embd}"));
            }
            let mut x = dequantize(&g, t)?;
            r1(&mut x); // every row is k / n_embd chunks of n_embd, each a target hidden state
            w.tensor(&quantize(&x, k, if fc_w4a4 { gguf::W4A4 } else { t.ty }))?;
            println!("  fc.weight {:?}: {} chunks x R1  {:.1} s", t.dims, k / n_embd, t0.elapsed().as_secs_f32());
        } else if lin(t) {
            let k = t.dims[0] as usize;
            let base = t.name.split('.').nth(2).unwrap_or("");
            // the row-parallel ones under -sm tensor (split along K) get the block rotation of --rot-blocks
            let rot = if matches!(base, "attn_output" | "ffn_down") { rot_len(k, rot_blocks, &t.name)? } else { k };
            let mut x = dequantize(&g, t)?;
            let q = if rot == k {
                quantize(&x, k, gguf::W4A4H)
            } else {
                x.par_chunks_mut(k).for_each(|r| hadamard_blocks(r, rot));
                let mut q = quantize(&x, k, gguf::W4A4);
                set_rot_len(&mut q, rot);
                q
            };
            w.tensor(&q)?;
        } else {
            w.tensor(g.data(t)?)?;
        }
    }
    for t in extra {
        if t.name == "output.weight" && head_w4a4h {
            let k = t.dims[0] as usize;
            w.tensor(&quantize(&dequantize(&s, t)?, k, gguf::W4A4H))?;
            println!("  + {} {} -> w4a4h from {src}", t.name, type_name(t.ty));
        } else {
            w.tensor(s.data(t)?)?;
            println!("  + {} {} from {src}", t.name, type_name(t.ty));
        }
    }
    w.finish()?;
    println!("{output}: done in {:.0} s", t0.elapsed().as_secs_f32());
    Ok(())
}

/// `ruquant head CONVERTED.gguf SRC.gguf OUT.gguf [--ost DIR | --seed N] [--type w4a4|q8_0]`: CONVERTED (from
/// `ruquant convert` with the same R1) with its lm_head rebuilt from SRC's output.weight diag(output_norm) R1 in
/// another type (W4A4 by default, through ggml's reference quantizers). Everything else is copied.
pub fn head_main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut seed = 1u64;
    let mut ost: Option<String> = None;
    let mut head_ty = gguf::W4A4;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => seed = it.next().and_then(|s| s.parse().ok()).ok_or("--seed needs a number")?,
            "--ost" => ost = Some(it.next().ok_or("--ost needs a directory")?.clone()),
            "--type" => match it.next().map(String::as_str) {
                Some("w4a4") => head_ty = gguf::W4A4,
                Some("q8_0") => head_ty = gguf::Q8_0,
                _ => return Err("--type is w4a4 or q8_0".into()),
            },
            _ => pos.push(a.clone()),
        }
    }
    let [conv, src, output] = &pos[..] else {
        return Err("usage: ruquant head CONVERTED.gguf SRC.gguf OUT.gguf [--ost DIR | --seed N] [--type w4a4|q8_0]".into());
    };
    let t0 = Instant::now();
    let g = Gguf::open(conv)?;
    let s = Gguf::open(src)?;
    let arch = g.string("general.architecture").ok_or("no general.architecture")?;
    let n_embd = g.u32(&format!("{arch}.embedding_length")).ok_or("no embedding_length")? as usize;
    let v: Option<Vec<f32>> = match &ost {
        Some(d) => {
            let f = format!("{d}/r1.f32");
            let raw = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
            Some(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        }
        None => None,
    };
    let d = signs(n_embd, seed);
    let infos: Vec<TensorInfo> = g
        .tensors
        .iter()
        .map(|t| TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty: if t.name == "output.weight" { head_ty } else { t.ty }, offset: 0 })
        .collect();
    let kvs: Vec<gguf::Kv> = g.kvs.iter().map(|k| gguf::Kv { key: k.key.clone(), raw: k.raw.clone() }).collect();
    let mut w = Writer::create(output, &kvs, &infos, g.alignment())?;
    for t in &g.tensors {
        if t.name != "output.weight" {
            w.tensor(g.data(t)?)?;
            continue;
        }
        let st = s.tensor("output.weight").ok_or(format!("{src}: no output.weight"))?;
        let norm = dequantize(&s, s.tensor("output_norm.weight").ok_or(format!("{src}: no output_norm.weight"))?)?;
        let mut x = dequantize(&s, st)?;
        x.par_chunks_mut(n_embd).for_each(|r| r.iter_mut().zip(&norm).for_each(|(v, s)| *v *= s));
        if let Some(v) = &v {
            let rows = x.len() / n_embd;
            let mut y = vec![0f32; x.len()];
            unsafe {
                gemm::gemm(
                    rows, n_embd, n_embd,
                    y.as_mut_ptr(), 1, n_embd as isize, false,
                    x.as_ptr(), 1, n_embd as isize,
                    v.as_ptr(), 1, n_embd as isize,
                    0.0f32, 1.0f32, false, false, false,
                    gemm::Parallelism::Rayon(0),
                );
            }
            x = y;
        } else {
            x.par_chunks_mut(n_embd).for_each(|r| {
                r.iter_mut().zip(&d).for_each(|(v, s)| *v *= s);
                hadamard(r);
            });
        }
        w.tensor(&quantize(&x, n_embd, head_ty))?;
        println!("  output.weight {} -> {} from {src}  {:.1} s", type_name(st.ty), type_name(head_ty), t0.elapsed().as_secs_f32());
    }
    w.finish()?;
    println!("{output}: done in {:.0} s", t0.elapsed().as_secs_f32());
    Ok(())
}

/// `ruquant setclip MODEL.gguf C [CH]`: in place, give every W4A4 tensor activation clip C and every W4A4H tensor
/// CH (default C), in its block headers (pad[1]), where ggml reads it unless RQ_ACT_CLIP / RQ_ACT_CLIP_H are set.
/// 0 clears it (back to 1, or the environment).
pub fn setclip_main(args: &[String]) -> Result<(), String> {
    let (path, c, ch) = match args {
        [p, c] => (p, c.parse::<f32>().map_err(|e| e.to_string())?, None),
        [p, c, ch] => (p, c.parse::<f32>().map_err(|e| e.to_string())?, Some(ch.parse::<f32>().map_err(|e| e.to_string())?)),
        _ => return Err("usage: ruquant setclip MODEL.gguf C [CH]".into()),
    };
    let ch = ch.unwrap_or(c);
    let g = Gguf::open(path)?;
    let spans: Vec<(usize, usize, f32)> = g
        .tensors
        .iter()
        .filter(|t| t.ty == gguf::W4A4 || t.ty == gguf::W4A4H)
        .map(|t| Ok((g.data_start + t.offset as usize, t.n_bytes()?, if t.ty == gguf::W4A4H { ch } else { c })))
        .collect::<Result<_, String>>()?;
    drop(g);
    let f = std::fs::OpenOptions::new().read(true).write(true).open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut map = unsafe { memmap2::MmapMut::map_mut(&f) }.map_err(|e| format!("{path}: {e}"))?;
    let (mut n4, mut n4h) = (0, 0);
    for &(start, len, clip) in &spans {
        for blk in map[start..start + len].chunks_exact_mut(144) {
            blk[12..16].copy_from_slice(&clip.to_le_bytes()); // block_w4a4.pad[1]
        }
        if clip == ch && clip != c { n4h += 1 } else { n4 += 1 }
    }
    map.flush().map_err(|e| format!("{path}: {e}"))?;
    println!("{path}: activation clip {c} on W4A4, {ch} on W4A4H ({} tensors)", n4 + n4h);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // X^T X of rotated rows, the way ruquant hessian accumulates it
    fn gram(x: &[f32], k: usize) -> Vec<f64> {
        let mut h = vec![0f64; k * k];
        for r in x.chunks(k) {
            for i in 0..k {
                for j in 0..k {
                    h[i * k + j] += r[i] as f64 * r[j] as f64;
                }
            }
        }
        h
    }

    #[test]
    fn rebased_hessian_is_the_block_rotations() {
        let (k, n, m) = (1536, 768, 40); // 1536 = 12 x 128, 768 = 12 x 64
        let mut s = 7u64;
        let x: Vec<f32> = (0..m * k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * if (s >> 20) % 97 == 0 { 30.0 } else { 1.0 }
            })
            .collect();
        let mut full = x.clone();
        full.chunks_mut(k).for_each(hadamard);
        let mut blk = x.clone();
        blk.chunks_mut(k).for_each(|r| hadamard_blocks(r, n));
        let h_full: Vec<f32> = gram(&full, k).iter().map(|&v| v as f32).collect();
        let want = gram(&blk, k);
        let got = rebase_hessian(&h_full, k, n);
        let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs()));
        let err = got.iter().zip(&want).fold(0f64, |a, (&g, &w)| a.max((g as f64 - w).abs()));
        assert!(err < 1e-5 * scale, "max error {err} of {scale}");
    }
}

/// `ruquant a16 IN.gguf OUT.gguf [--only NAME,...] [--type f16|q8_0]`: the W4A4 / W4A4H tensors (those whose name
/// contains one of --only, default all) dequantized as they are, the online Hadamard undone, and stored in a plain
/// type, so they run with unquantized activations: the model's own int4 weights at A16. An ablation for how much of
/// the error is activation quantization.
pub fn a16_main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut only: Vec<String> = Vec::new();
    let mut ty = gguf::F16;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--only" => only.extend(it.next().ok_or("--only needs a comma list")?.split(',').filter(|s| !s.is_empty()).map(String::from)),
            "--type" => match it.next().map(String::as_str) {
                Some("f16") => ty = gguf::F16,
                Some("q8_0") => ty = gguf::Q8_0,
                _ => return Err("--type is f16 or q8_0".into()),
            },
            _ => pos.push(a.clone()),
        }
    }
    let [input, output] = &pos[..] else {
        return Err("usage: ruquant a16 IN.gguf OUT.gguf [--only NAME,...] [--type f16|q8_0]".into());
    };
    let t0 = Instant::now();
    let g = Gguf::open(input)?;
    let pick = |t: &TensorInfo| (t.ty == gguf::W4A4 || t.ty == gguf::W4A4H) && (only.is_empty() || only.iter().any(|o| t.name.contains(o.as_str())));
    let infos: Vec<TensorInfo> = g
        .tensors
        .iter()
        .map(|t| TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty: if pick(t) { ty } else { t.ty }, offset: 0 })
        .collect();
    let kvs: Vec<gguf::Kv> = g.kvs.iter().map(|k| gguf::Kv { key: k.key.clone(), raw: k.raw.clone() }).collect();
    let mut w = Writer::create(output, &kvs, &infos, g.alignment())?;
    let mut n = 0;
    for t in &g.tensors {
        if !pick(t) {
            w.tensor(g.data(t)?)?;
            continue;
        }
        let x = dequantize(&g, t)?; // W4A4H: back in the unrotated basis
        w.tensor(&quantize(&x, t.dims[0] as usize, ty))?;
        n += 1;
    }
    w.finish()?;
    println!("{output}: {n} tensors at A16 ({}) in {:.0} s", type_name(ty), t0.elapsed().as_secs_f32());
    Ok(())
}

/// `ruquant graft BASE.gguf DONOR.gguf OUT.gguf --only NAME,...`: BASE with the tensors whose name contains one of
/// --only taken from DONOR (same tensor names and shapes, any type). With a donor converted with every linear kept at
/// Q8_0 (same rotation), this is BASE with those classes free of quantization error, in seconds instead of a GPTQ run.
pub fn graft_main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut only: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--only" => only.extend(it.next().ok_or("--only needs a comma list")?.split(',').filter(|s| !s.is_empty()).map(String::from)),
            _ => pos.push(a.clone()),
        }
    }
    let [base, donor, output] = &pos[..] else {
        return Err("usage: ruquant graft BASE.gguf DONOR.gguf OUT.gguf --only NAME,...".into());
    };
    if only.is_empty() {
        return Err("graft: --only names the tensors to take from DONOR".into());
    }
    let t0 = Instant::now();
    let g = Gguf::open(base)?;
    let d = Gguf::open(donor)?;
    let pick = |t: &TensorInfo| only.iter().any(|o| t.name.contains(o.as_str()));
    let mut infos = Vec::new();
    for t in &g.tensors {
        let src = if pick(t) { d.tensor(&t.name).ok_or(format!("{donor}: no {}", t.name))? } else { t };
        if src.dims != t.dims {
            return Err(format!("{}: shape {:?} in {donor}, {:?} in {base}", t.name, src.dims, t.dims));
        }
        infos.push(TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty: src.ty, offset: 0 });
    }
    let kvs: Vec<gguf::Kv> = g.kvs.iter().map(|k| gguf::Kv { key: k.key.clone(), raw: k.raw.clone() }).collect();
    let mut w = Writer::create(output, &kvs, &infos, g.alignment())?;
    let mut n = 0;
    for t in &g.tensors {
        if pick(t) {
            w.tensor(d.data(d.tensor(&t.name).unwrap())?)?;
            n += 1;
        } else {
            w.tensor(g.data(t)?)?;
        }
    }
    w.finish()?;
    println!("{output}: {n} tensors from {donor} in {:.0} s", t0.elapsed().as_secs_f32());
    Ok(())
}
