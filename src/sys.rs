//! Raw ggml bindings for our patched tree, plus the three lookups every caller needs.
#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CStr;

pub fn load_devices() -> Vec<ggml_backend_dev_t> {
    unsafe {
        ggml_backend_load_all();
        (0..ggml_backend_dev_count())
            .map(|i| ggml_backend_dev_get(i))
            .collect()
    }
}

pub fn dev_name(dev: ggml_backend_dev_t) -> String {
    unsafe {
        CStr::from_ptr(ggml_backend_dev_name(dev))
            .to_string_lossy()
            .into_owned()
    }
}

pub fn dev_desc(dev: ggml_backend_dev_t) -> String {
    unsafe {
        CStr::from_ptr(ggml_backend_dev_description(dev))
            .to_string_lossy()
            .into_owned()
    }
}

pub fn dev_memory(dev: ggml_backend_dev_t) -> (usize, usize) {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { ggml_backend_dev_memory(dev, &mut free, &mut total) };
    (free, total)
}

pub fn is_cpu(dev: ggml_backend_dev_t) -> bool {
    dev_name(dev).to_uppercase().contains("CPU")
}

pub fn is_cuda(dev: ggml_backend_dev_t) -> bool {
    dev_name(dev).starts_with("CUDA")
}

pub fn gflops(flops: f64, ms: f64) -> f64 {
    flops / (ms * 1e6)
}
