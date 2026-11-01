//! Matmul smoke test: same f32 graph on every ggml device, checked against CPU.

use crate::sys::{self, ggml_backend_dev_t};
use std::os::raw::c_void;
use std::time::Instant;

const M: i64 = 256;
const N: i64 = 256;
const K: i64 = 256;

/// Deterministic operand pattern, so CPU and GPU see identical input.
fn operand(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7 + salt * 13) % 32) as f32 - 16.0) * 0.25)
        .collect()
}

/// Build and run the matmul graph on `dev`. Returns (best ms, output [N, M]).
unsafe fn matmul(dev: ggml_backend_dev_t) -> Result<(f64, Vec<f32>), String> {
    let backend = unsafe { sys::ggml_backend_dev_init(dev, std::ptr::null()) };
    if backend.is_null() {
        return Err("dev_init returned null".into());
    }

    let mut params: sys::ggml_init_params = unsafe { std::mem::zeroed() };
    params.mem_size =
        unsafe { sys::ggml_tensor_overhead() } * 8 + unsafe { sys::ggml_graph_overhead() };
    params.no_alloc = true;
    let ctx = unsafe { sys::ggml_init(params) };

    let a = unsafe { sys::ggml_new_tensor_2d(ctx, sys::GGML_TYPE_F32, K, M) };
    let b = unsafe { sys::ggml_new_tensor_2d(ctx, sys::GGML_TYPE_F32, K, N) };
    let c = unsafe { sys::ggml_mul_mat(ctx, a, b) };
    let graph = unsafe { sys::ggml_new_graph(ctx) };
    unsafe { sys::ggml_build_forward_expand(graph, c) };

    let buf = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
    if buf.is_null() {
        unsafe {
            sys::ggml_free(ctx);
            sys::ggml_backend_free(backend)
        };
        return Err("alloc_ctx_tensors returned null".into());
    }

    let (adata, bdata) = (operand((M * K) as usize, 1), operand((N * K) as usize, 2));
    unsafe {
        sys::ggml_backend_tensor_set(a, adata.as_ptr() as *const c_void, 0, adata.len() * 4);
        sys::ggml_backend_tensor_set(b, bdata.as_ptr() as *const c_void, 0, bdata.len() * 4);
    }

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let status = unsafe { sys::ggml_backend_graph_compute(backend, graph) };
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        if (status as u32) != 0 {
            unsafe {
                sys::ggml_backend_buffer_free(buf);
                sys::ggml_free(ctx);
                sys::ggml_backend_free(backend)
            };
            return Err(format!("graph_compute failed: {status:?}"));
        }
    }

    let mut out = vec![0f32; (M * N) as usize];
    unsafe {
        sys::ggml_backend_tensor_get(c, out.as_mut_ptr() as *mut c_void, 0, out.len() * 4);
        sys::ggml_backend_buffer_free(buf);
        sys::ggml_free(ctx);
        sys::ggml_backend_free(backend);
    }
    Ok((best, out))
}

pub fn run(devices: &[ggml_backend_dev_t]) -> Result<(), String> {
    let cpu = *devices
        .iter()
        .find(|d| sys::is_cpu(**d))
        .ok_or("no CPU backend for a reference result")?;
    let (cpu_ms, reference) = unsafe { matmul(cpu) }?;
    let flops = 2.0 * (M * N * K) as f64;

    println!("\n{M}x{N}x{K} f32 matmul, best of 3");
    println!(
        "  {:<8} {:9.3} ms  {:7.1} GFLOP/s  reference",
        sys::dev_name(cpu),
        cpu_ms,
        sys::gflops(flops, cpu_ms)
    );

    for dev in devices.iter().filter(|d| !sys::is_cpu(**d)) {
        let (ms, out) = unsafe { matmul(*dev) }?;
        let max_err = reference
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "  {:<8} {:9.3} ms  {:7.1} GFLOP/s  {:.2}x cpu  max_err {:.2e}  {}",
            sys::dev_name(*dev),
            ms,
            sys::gflops(flops, ms),
            cpu_ms / ms,
            max_err,
            if max_err < 1e-2 { "ok" } else { "MISMATCH" }
        );
    }
    Ok(())
}
