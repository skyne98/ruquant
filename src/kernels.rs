//! Our device kernels, wired into ggml by hand — no C++ glue.
//!
//! `ggml_map_custom1` normally falls back to CPU. The patched ggml-cuda exposes
//! `ggml_cuda_set_custom1_hook`; once registered, it claims GGML_OP_MAP_CUSTOM1 and calls our
//! launcher from inside its own CUDA graph capture, so the kernel replays with the graph.

use crate::sys;

use std::os::raw::{c_int, c_void};

#[cfg(rq_cuda)]
unsafe extern "C" {
    /// From kernels/rq_scale.cu (nvcc, arch=native).
    fn rq_scale_launch(src: *const f32, dst: *mut f32, n: i64, mul: f32, stream: *mut c_void);
    /// From the ruquant hook in patches/llama.cpp-ruquant.patch (ggml-cuda.cu).
    fn ggml_cuda_set_custom1_hook(
        f: Option<unsafe extern "C" fn(*mut c_void, *mut sys::ggml_tensor)>,
    );
}

/// GPU path: dst = src * 2, launched on ggml-cuda's stream, therefore captured.
#[cfg(rq_cuda)]
unsafe extern "C" fn scale_twice(stream: *mut c_void, dst: *mut sys::ggml_tensor) {
    let src = unsafe { (*dst).src[0] };
    unsafe {
        rq_scale_launch(
            (*src).data as *const f32,
            (*dst).data as *mut f32,
            sys::ggml_nelements(dst),
            2.0,
            stream,
        )
    };
}

/// Hands the GPU path to ggml-cuda. A no-op where the crate was built without CUDA.
pub fn register() {
    #[cfg(rq_cuda)]
    unsafe {
        ggml_cuda_set_custom1_hook(Some(scale_twice))
    }
}

/// CPU path for the same op. Never called on the GPU route; it keeps the graph correct if it
/// is ever scheduled onto the CPU backend.
pub unsafe extern "C" fn double_rows(
    dst: *mut sys::ggml_tensor,
    a: *const sys::ggml_tensor,
    ith: c_int,
    nth: c_int,
    _userdata: *mut c_void,
) {
    let n = unsafe { sys::ggml_nelements(dst) };
    let src = unsafe { (*a).data } as *const f32;
    let out = unsafe { (*dst).data } as *mut f32;
    let mut i = ith as i64;
    while i < n {
        unsafe { *out.offset(i as isize) = *src.offset(i as isize) * 2.0 };
        i += nth as i64;
    }
}
