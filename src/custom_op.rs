//! Route A: our kernel as a first-class ggml node.
//!
//! `ggml_map_custom1` normally falls back to CPU; the patched ggml-cuda claims the op and runs
//! our launcher inside its own CUDA graph capture, so it replays with everything else.

use crate::kernels;
use crate::sys;

use std::os::raw::c_void;
use std::time::Instant;

const N: i64 = 8;
const ITERS: usize = 50;
const WARMUP: usize = 10;

pub fn run(dev: sys::ggml_backend_dev_t) -> Result<(), String> {
    let backend = unsafe { sys::ggml_backend_dev_init(dev, std::ptr::null()) };
    if backend.is_null() {
        return Err(format!("{}: dev_init returned null", sys::dev_name(dev)));
    }

    let mut params: sys::ggml_init_params = unsafe { std::mem::zeroed() };
    params.mem_size =
        unsafe { sys::ggml_tensor_overhead() } * 8 + unsafe { sys::ggml_graph_overhead() };
    params.no_alloc = true;
    let ctx = unsafe { sys::ggml_init(params) };

    // a --[our kernel: ×2]--> b --[ggml_scale: ×3]--> c
    let a = unsafe { sys::ggml_new_tensor_1d(ctx, sys::GGML_TYPE_F32, N) };
    let b = unsafe {
        sys::ggml_map_custom1(ctx, a, Some(kernels::double_rows), 1, std::ptr::null_mut())
    };
    let c = unsafe { sys::ggml_scale(ctx, b, 3.0) };

    let graph = unsafe { sys::ggml_new_graph(ctx) };
    unsafe { sys::ggml_build_forward_expand(graph, c) };

    let buf = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
    if buf.is_null() {
        return Err("alloc_ctx_tensors returned null".into());
    }

    let input: Vec<f32> = (0..N).map(|i| i as f32 + 0.5).collect();
    unsafe { sys::ggml_backend_tensor_set(a, input.as_ptr() as *const c_void, 0, input.len() * 4) };

    let mut times = Vec::with_capacity(ITERS - WARMUP);
    for i in 0..ITERS {
        let t0 = Instant::now();
        let status = unsafe { sys::ggml_backend_graph_compute(backend, graph) };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if (status as u32) != 0 {
            return Err(format!("graph_compute failed: {status:?}"));
        }
        if i >= WARMUP {
            times.push(ms);
        }
    }

    let mut out = vec![0f32; N as usize];
    unsafe {
        sys::ggml_backend_tensor_get(c, out.as_mut_ptr() as *mut c_void, 0, out.len() * 4);
        sys::ggml_backend_buffer_free(buf);
        sys::ggml_free(ctx);
        sys::ggml_backend_free(backend);
    }

    let bad = out
        .iter()
        .zip(&input)
        .filter(|(got, want)| (**got - **want * 6.0).abs() > 1e-5)
        .count();
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "\ncustom kernel in-graph on {}: best {:.1} us  median {:.1} us  {}",
        sys::dev_name(dev),
        times[0] * 1e3,
        times[times.len() / 2] * 1e3,
        if bad == 0 { "ok" } else { "MISMATCH" }
    );

    if bad == 0 {
        Ok(())
    } else {
        Err(format!("{bad} mismatched elements"))
    }
}
