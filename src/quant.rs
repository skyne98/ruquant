//! Does the int4 (4-bit weight) path actually work through this pipeline?
//!
//! Builds `out = Q4_0[K,N] × F32[K,M]` and checks the GPU result against the CPU backend.
//! Every backend runs a different q4_0 kernel: ggml-cpu its own vec_dot; ggml-cuda unpacks the
//! nibbles to int8 and uses dp4a (M=1) or int8 `mma` (M>1); ggml-vulkan dequantises to f32 and
//! does a float dot (its integer-dot path needs q8_1 activations, which this never feeds).
//! None of them uses int4×int4 hardware. Agreement means the 4-bit weights are handled right.

use crate::sys;

use std::os::raw::c_void;
use std::time::Instant;

pub struct Shape {
    pub k: i64,
    pub n: i64,
    pub m: i64,
}

impl Shape {
    fn label(&self) -> String {
        format!("K={} N={} M={}", self.k, self.n, self.m)
    }
    fn flops(&self) -> f64 {
        2.0 * (self.k * self.n * self.m) as f64
    }
    fn weight_bytes(&self) -> f64 {
        // q4_0: 32 weights + one f16 scale per 32 elements
        (self.k * self.n) as f64 * 18.0 / 32.0
    }
}

/// Deterministic inputs, so every backend quantises and multiplies the same numbers.
fn pattern(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = ((i * 7 + salt * 13) % 64) as f32 - 32.0;
            v * 0.03125
        })
        .collect()
}

struct Run {
    ms: f64,
    out: Vec<f32>,
}

unsafe fn q4_matmul(dev: sys::ggml_backend_dev_t, s: &Shape) -> Result<Run, String> {
    let backend = unsafe { sys::ggml_backend_dev_init(dev, std::ptr::null()) };
    if backend.is_null() {
        return Err("dev_init returned null".into());
    }

    let mut params: sys::ggml_init_params = unsafe { std::mem::zeroed() };
    params.mem_size =
        unsafe { sys::ggml_tensor_overhead() } * 8 + unsafe { sys::ggml_graph_overhead() };
    params.no_alloc = true;
    let ctx = unsafe { sys::ggml_init(params) };

    // weights: 4-bit, activations: f32 (the decode/prefill shape ggml supports everywhere)
    let w = unsafe { sys::ggml_new_tensor_2d(ctx, sys::GGML_TYPE_Q4_0, s.k, s.n) };
    let x = unsafe { sys::ggml_new_tensor_2d(ctx, sys::GGML_TYPE_F32, s.k, s.m) };
    let out = unsafe { sys::ggml_mul_mat(ctx, w, x) };
    let graph = unsafe { sys::ggml_new_graph(ctx) };
    unsafe { sys::ggml_build_forward_expand(graph, out) };

    let wbytes = unsafe { sys::ggml_nbytes(w) };
    let wf = pattern((s.k * s.n) as usize, 1);
    let mut wq = vec![0u8; wbytes];
    let written = unsafe {
        sys::ggml_quantize_chunk(
            sys::GGML_TYPE_Q4_0,
            wf.as_ptr(),
            wq.as_mut_ptr() as *mut c_void,
            0,
            s.n,
            s.k,
            std::ptr::null(),
        )
    };
    if written == 0 || written > wbytes {
        return Err(format!("quantize_chunk wrote {written} of {wbytes} bytes"));
    }

    let xf = pattern((s.k * s.m) as usize, 2);

    let buf = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
    if buf.is_null() {
        return Err("alloc_ctx_tensors returned null".into());
    }
    unsafe {
        sys::ggml_backend_tensor_set(w, wq.as_ptr() as *const c_void, 0, wbytes);
        sys::ggml_backend_tensor_set(x, xf.as_ptr() as *const c_void, 0, xf.len() * 4);
    }

    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        let status = unsafe { sys::ggml_backend_graph_compute(backend, graph) };
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        if (status as u32) != 0 {
            unsafe {
                sys::ggml_backend_buffer_free(buf);
                sys::ggml_free(ctx);
                sys::ggml_backend_free(backend);
            }
            return Err(format!("graph_compute failed on q4_0: {status:?}"));
        }
    }

    let mut result = vec![0f32; (s.n * s.m) as usize];
    unsafe {
        sys::ggml_backend_tensor_get(out, result.as_mut_ptr() as *mut c_void, 0, result.len() * 4);
        sys::ggml_backend_buffer_free(buf);
        sys::ggml_free(ctx);
        sys::ggml_backend_free(backend);
    }

    Ok(Run {
        ms: best,
        out: result,
    })
}

/// Max |gpu - cpu| relative to the CPU magnitude: the two backends run different q4_0
/// kernels, so exact equality is not expected — a large value means the quant path is broken.
fn rel_err(reference: &[f32], other: &[f32]) -> f32 {
    let scale = reference.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    reference
        .iter()
        .zip(other)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale
}

pub fn run(devices: &[sys::ggml_backend_dev_t]) -> Result<(), String> {
    let cpu = *devices
        .iter()
        .find(|d| sys::is_cpu(**d))
        .ok_or("no CPU backend")?;
    let gpus: Vec<_> = devices
        .iter()
        .copied()
        .filter(|d| !sys::is_cpu(*d))
        .collect();

    let shapes = [
        Shape {
            k: 4096,
            n: 1024,
            m: 1,
        }, // decode: GEMV
        Shape {
            k: 4096,
            n: 1024,
            m: 64,
        }, // small prefill
    ];

    for shape in &shapes {
        let reference = unsafe { q4_matmul(cpu, shape) }?;
        println!(
            "\nQ4_0 weights × f32 activations, {} (best of 5)",
            shape.label()
        );
        println!(
            "  {:<8} {:8.3} ms  {:6.1} GFLOP/s  {:6.1} GB/s  reference",
            sys::dev_name(cpu),
            reference.ms,
            sys::gflops(shape.flops(), reference.ms),
            shape.weight_bytes() / (reference.ms * 1e6),
        );

        for dev in &gpus {
            match unsafe { q4_matmul(*dev, shape) } {
                Ok(run) => {
                    let err = rel_err(&reference.out, &run.out);
                    println!(
                        "  {:<8} {:8.3} ms  {:6.1} GFLOP/s  {:6.1} GB/s  rel_err {:.1e}  {}",
                        sys::dev_name(*dev),
                        run.ms,
                        sys::gflops(shape.flops(), run.ms),
                        shape.weight_bytes() / (run.ms * 1e6),
                        err,
                        if err < 5e-2 { "ok" } else { "MISMATCH" }
                    );
                }
                Err(e) => println!("  {:<8} skipped: {e}", sys::dev_name(*dev)),
            }
        }
    }
    Ok(())
}
