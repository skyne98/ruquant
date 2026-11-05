//! `ruquant hessian MODEL.gguf TEXT OUTDIR [--seqs 16] [--ctx 2048] [--per-pass 16]`: GPTQ Hessians
//! H = sum x x^T of every distinct linear input, from MODEL run on TEXT by libllama.
//!
//! MODEL is the rotated W4A4 GGUF, so each layer's inputs carry the quantization error of the layers
//! before it (GPTQ's sequential mode). Four inputs per layer: `in` (attn_norm output: attn_q/k/v or
//! DeltaNet's attn_qkv/attn_gate/ssm_alpha/ssm_beta), `post` (ffn_gate/up), `o` (attn_output or ssm_out)
//! and `down` (ffn_down); the last two get the online Hadamard first, as the W4A4H weights see them.
//! A layer's Hessians take 1.6 GB (ffn_down's input is 17408 wide), so layers go in passes of
//! --per-pass over the same calibration windows. Output: OUTDIR/blk.N.KIND.f32, K x K f32.
//! --act-clip C [--act-clip-h CH]: accumulate the inputs as the W4A4 kernels see them, after the per-token int4
//! quantization (clip C; CH for the Hadamard inputs, default C), so GPTQ fits the weights to the activations the
//! runtime actually multiplies.
//! --fp FPMODEL: GPTAQ (asymmetric calibration). FPMODEL is the same rotation unquantized (convert --keep every
//! linear); each window runs through both, and next to H = X~ X~^T of MODEL's inputs X~ (quantized as the runtime's
//! activation groups do) it writes C = X X~^T with X the FP model's inputs at the same tokens, as
//! OUTDIR/blk.N.KIND.c.f32. convert then fits each weight to W C H^-1 (the least-squares weight for the quantized
//! inputs and the FP outputs) before GPTQ, so GPTQ's codes reproduce W X, not W X~.

use crate::sys;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::io::Write;
use std::time::Instant;

unsafe extern "C" {
    fn ggml_rq_hadamard(x: *mut f32, n: i64, transpose: bool);
    fn ggml_rq_act_group() -> i32;
}

/// The input kind a weight reads, for the one weight per input whose matmul we observe.
fn kind_of(base: &str) -> Option<(&'static str, bool)> {
    Some(match base {
        "attn_q" | "attn_qkv" => ("in", false),
        "ffn_gate" => ("post", false),
        "attn_output" | "ssm_out" => ("o", true),
        "ffn_down" => ("down", true),
        _ => return None,
    })
}

fn parse(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (il, tail) = rest.split_once('.')?;
    Some((il.parse().ok()?, tail.strip_suffix(".weight")?))
}

/// The runtime's per-token activation quantizer (rq_quantize_act_kernel), dequantized in place.
fn fake_quant(row: &mut [f32], clip: f32) {
    let (mut mn, mut mx) = (0f32, 0f32);
    for &v in row.iter() {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    mn *= clip;
    mx *= clip;
    if mn == 0.0 && mx == 0.0 {
        mn = -1.0;
        mx = 1.0;
    }
    let scale = (mx - mn) / 15.0;
    let inv = 1.0 / scale;
    let z = (-mn * inv).round_ties_even().clamp(0.0, 15.0);
    for v in row.iter_mut() {
        let q = ((*v * inv).round_ties_even() + z).clamp(0.0, 15.0);
        *v = (q - z) * scale;
    }
}

/// The runtime's grouped activation quantizer (rq_quantize_row_a / rq_quantize_act_kernel), dequantized in place:
/// per group of `g` a zero point and an integer multiple (1..255) of the token's unit.
fn fake_quant_groups(row: &mut [f32], clip: f32, g: usize) {
    let ranges: Vec<(f32, f32)> = row
        .chunks(g)
        .map(|c| {
            let (mn, mx) = c.iter().fold((0f32, 0f32), |(a, b), &v| (a.min(v), b.max(v)));
            (mn * clip, mx * clip)
        })
        .collect();
    let mut r = ranges.iter().fold(0f32, |m, &(lo, hi)| m.max(hi - lo));
    if r == 0.0 {
        r = 2.0;
    }
    let u = r / (15.0 * 255.0);
    for (c, &(lo, hi)) in row.chunks_mut(g).zip(&ranges) {
        let m = ((hi - lo) * 255.0 / r).ceil().clamp(1.0, 255.0);
        let inv = 1.0 / (m * u);
        let z = (-lo * inv).round_ties_even().clamp(0.0, 15.0);
        for v in c.iter_mut() {
            let q = ((*v * inv).round_ties_even() + z).clamp(0.0, 15.0);
            *v = (q - z) * m * u;
        }
    }
}

struct State {
    act_clip: Option<(f32, f32)>,
    layers: std::ops::Range<usize>,
    h: HashMap<(usize, &'static str), (usize, Vec<f32>)>, // (layer, kind) -> (K, H)
    buf: Vec<f32>,
    // GPTAQ: the FP model's inputs of this window (captured first), then C += X^T X~ while the quantized model runs
    capture: bool,
    group: usize,
    x: HashMap<(usize, &'static str), Vec<f32>>,
    c: HashMap<(usize, &'static str), Vec<f32>>,
}

unsafe extern "C" fn observe(t: *mut sys::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    let st = unsafe { &mut *(ud as *mut State) };
    let t = unsafe { &*t };
    if t.op != sys::GGML_OP_MUL_MAT {
        return !ask;
    }
    let src0 = unsafe { &*t.src[0] };
    let name = unsafe { CStr::from_ptr(src0.name.as_ptr()) }.to_string_lossy();
    let Some((il, base)) = parse(&name) else { return !ask };
    let Some((kind, had)) = kind_of(base) else { return !ask };
    if !st.layers.contains(&il) {
        return !ask;
    }
    if ask {
        return true;
    }
    let x = unsafe { &*t.src[1] };
    let (k, m) = (x.ne[0] as usize, (x.ne[1] * x.ne[2] * x.ne[3]) as usize);
    assert!(x.type_ == sys::GGML_TYPE_F32 && x.nb[1] as usize == k * 4, "{name}: input not contiguous f32");
    st.buf.resize(k * m, 0.0);
    unsafe { sys::ggml_backend_tensor_get(t.src[1], st.buf.as_mut_ptr() as *mut c_void, 0, k * m * 4) };
    if had {
        st.buf.par_chunks_mut(k).for_each(|row| unsafe { ggml_rq_hadamard(row.as_mut_ptr(), k as i64, false) });
    }
    if st.capture {
        st.x.insert((il, kind), st.buf.clone());
        return true;
    }
    if let Some((c, ch)) = st.act_clip {
        let clip = if had { ch } else { c };
        let g = st.group;
        if g > 0 {
            st.buf.par_chunks_mut(k).for_each(|row| fake_quant_groups(row, clip, g));
        } else {
            st.buf.par_chunks_mut(k).for_each(|row| fake_quant(row, clip));
        }
    }
    if let Some(xf) = st.x.get(&(il, kind)) {
        assert!(xf.len() == st.buf.len(), "{name}: FP and quantized inputs differ in shape");
        let c = st.c.entry((il, kind)).or_insert_with(|| vec![0f32; k * k]);
        // C += X^T X~ (both row-major m x k)
        unsafe {
            gemm::gemm(
                k, k, m,
                c.as_mut_ptr(), 1, k as isize, true,
                xf.as_ptr(), k as isize, 1,
                st.buf.as_ptr(), 1, k as isize,
                1.0f32, 1.0f32, false, false, false,
                gemm::Parallelism::Rayon(0),
            );
        }
    }
    let (_, h) = st.h.entry((il, kind)).or_insert_with(|| (k, vec![0f32; k * k]));
    // H += X^T X, X row-major m x k
    unsafe {
        gemm::gemm(
            k, k, m,
            h.as_mut_ptr(), 1, k as isize, true,
            st.buf.as_ptr(), k as isize, 1,
            st.buf.as_ptr(), 1, k as isize,
            1.0f32, 1.0f32, false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }
    true
}

pub fn main(args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let (mut seqs, mut ctx, mut per_pass) = (16usize, 2048usize, 16usize);
    let (mut act_clip, mut act_clip_h): (Option<f32>, Option<f32>) = (None, None);
    let mut fp_path: Option<String> = None;
    let mut only: Option<(usize, usize)> = None;
    let mut no_act_quant = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |what: &str| -> Result<usize, String> {
            it.next().and_then(|s| s.parse().ok()).ok_or(format!("{what} needs a number"))
        };
        match a.as_str() {
            "--seqs" => seqs = num("--seqs")?,
            "--ctx" => ctx = num("--ctx")?,
            "--per-pass" => per_pass = num("--per-pass")?,
            "--act-clip" => act_clip = Some(it.next().and_then(|s| s.parse().ok()).ok_or("--act-clip needs a number")?),
            "--act-clip-h" => act_clip_h = Some(it.next().and_then(|s| s.parse().ok()).ok_or("--act-clip-h needs a number")?),
            "--fp" => fp_path = Some(it.next().ok_or("--fp needs a model")?.clone()),
            "--no-act-quant" => no_act_quant = true,
            "--layers" => {
                let v = it.next().ok_or("--layers needs A..B")?;
                let (a, b) = v.split_once("..").ok_or("--layers needs A..B")?;
                only = Some((a.parse().map_err(|_| "--layers: bad A")?, b.parse().map_err(|_| "--layers: bad B")?));
            }
            _ => pos.push(a.clone()),
        }
    }
    let [model_path, text_path, out_dir] = &pos[..] else {
        return Err("usage: ruquant hessian MODEL.gguf TEXT OUTDIR [--seqs 16] [--ctx 2048] [--per-pass 16] [--act-clip C [--act-clip-h CH]] [--fp FPMODEL] [--layers A..B]".into());
    };
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let text = std::fs::read_to_string(text_path).map_err(|e| format!("{text_path}: {e}"))?;

    unsafe { sys::llama_backend_init() };
    let mut mp = unsafe { sys::llama_model_default_params() };
    mp.n_gpu_layers = 999;
    let cpath = CString::new(model_path.as_str()).unwrap();
    let model = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), mp) };
    if model.is_null() {
        return Err(format!("{model_path}: failed to load"));
    }
    // GPTAQ: the inputs the quantized model multiplies are its activations after the runtime's grouped quantizer
    if fp_path.is_some() && act_clip.is_none() && !no_act_quant {
        act_clip = Some(1.0);
    }
    let group = if fp_path.is_some() { (unsafe { ggml_rq_act_group() }) as usize } else { 0 };
    let fp_model = match &fp_path {
        Some(p) => {
            let c = CString::new(p.as_str()).unwrap();
            let m = unsafe { sys::llama_model_load_from_file(c.as_ptr(), mp) };
            if m.is_null() {
                return Err(format!("{p}: failed to load"));
            }
            Some(m)
        }
        None => None,
    };
    let vocab = unsafe { sys::llama_model_get_vocab(model) };
    let n_layer = unsafe { sys::llama_model_n_layer(model) } as usize;

    // tokenize the whole text once, then take `seqs` evenly spread windows of `ctx` tokens
    let ctext = CString::new(text.as_bytes()).map_err(|_| "text has a NUL byte")?;
    let mut tokens = vec![0i32; text.len() + 16];
    let n = unsafe {
        sys::llama_tokenize(vocab, ctext.as_ptr(), text.len() as i32, tokens.as_mut_ptr(), tokens.len() as i32, false, false)
    };
    if n < (seqs * ctx) as i32 {
        return Err(format!("{text_path}: {n} tokens, need {}", seqs * ctx));
    }
    tokens.truncate(n as usize);
    let stride = (tokens.len() - ctx) / seqs;
    println!("{model_path}: {n_layer} layers; {text_path}: {n} tokens, {seqs} windows of {ctx}");

    let t0 = Instant::now();
    let (lo, hi) = only.unwrap_or((0, n_layer));
    for first in (lo..hi.min(n_layer)).step_by(per_pass) {
        let layers = first..(first + per_pass).min(hi.min(n_layer));
        let mut st = State {
            act_clip: act_clip.map(|c| (c, act_clip_h.unwrap_or(c))),
            layers: layers.clone(),
            h: HashMap::new(),
            buf: Vec::new(),
            capture: false,
            group,
            x: HashMap::new(),
            c: HashMap::new(),
        };
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = ctx as u32;
        cp.n_batch = ctx as u32;
        if fp_model.is_some() {
            cp.n_ubatch = ctx as u32; // one graph per window: --fp pairs each FP input with the same tokens' quantized one
        }
        cp.cb_eval = Some(observe);
        cp.cb_eval_user_data = &mut st as *mut State as *mut c_void;
        let lctx = unsafe { sys::llama_init_from_model(model, cp) };
        if lctx.is_null() {
            return Err("failed to create a llama context".into());
        }
        let fctx = match fp_model {
            Some(m) => {
                let c = unsafe { sys::llama_init_from_model(m, cp) };
                if c.is_null() {
                    return Err("failed to create the FP model's llama context".into());
                }
                Some(c)
            }
            None => None,
        };
        // every position is an output: with llama_batch_get_one only the last token is, and llama.cpp drops every
        // other row before the last layer's FFN, so its ffn Hessians came from one token per window
        let decode = |c: *mut sys::llama_context, win: &mut [i32]| -> Result<(), String> {
            unsafe {
                sys::llama_memory_clear(sys::llama_get_memory(c), true);
                let b = sys::llama_batch_init(ctx as i32, 0, 1);
                for (i, &t) in win.iter().enumerate() {
                    *b.token.add(i) = t;
                    *b.pos.add(i) = i as i32;
                    *b.n_seq_id.add(i) = 1;
                    **b.seq_id.add(i) = 0;
                    *b.logits.add(i) = 1;
                }
                let mut b = b;
                b.n_tokens = ctx as i32;
                let rc = sys::llama_decode(c, b);
                sys::llama_batch_free(b);
                if rc != 0 {
                    return Err(format!("llama_decode failed ({rc})"));
                }
            }
            Ok(())
        };
        for s in 0..seqs {
            let win = &mut tokens[s * stride..s * stride + ctx];
            if let Some(f) = fctx {
                st.capture = true;
                decode(f, win)?;
                st.capture = false;
            }
            decode(lctx, win)?;
            st.x.clear();
            print!("\r  layers {:2}..{:2}: window {}/{seqs}  {:5.0} s", layers.start, layers.end, s + 1, t0.elapsed().as_secs_f32());
            std::io::stdout().flush().ok();
        }
        unsafe { sys::llama_free(lctx) };
        if let Some(f) = fctx {
            unsafe { sys::llama_free(f) };
        }
        for ((il, kind), (k, h)) in &st.h {
            let path = format!("{out_dir}/blk.{il}.{kind}.f32");
            let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))?;
            let _ = k;
        }
        for ((il, kind), c) in &st.c {
            let path = format!("{out_dir}/blk.{il}.{kind}.c.f32");
            let bytes: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))?;
        }
        println!("  -> {} Hessians", st.h.len());
    }
    unsafe { sys::llama_model_free(model) };
    if let Some(m) = fp_model {
        unsafe { sys::llama_model_free(m) };
    }
    println!("done in {:.0} s", t0.elapsed().as_secs_f32());
    Ok(())
}
