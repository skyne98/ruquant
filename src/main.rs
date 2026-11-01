//! ruquant: ggml + our kernels, one pipeline.
//!
//!   cargo run --release   (inside nix develop, after ./build-ggml.sh)

mod bench;
mod custom_op;
mod kernels;
mod quant;
mod sys;

use std::process::ExitCode;

fn main() -> ExitCode {
    let devices = sys::load_devices();
    println!("ggml devices: {}", devices.len());
    for (i, dev) in devices.iter().enumerate() {
        let (free, total) = sys::dev_memory(*dev);
        println!(
            "  [{i}] {:<8} {:<40} {} / {} MiB",
            sys::dev_name(*dev),
            sys::dev_desc(*dev),
            free >> 20,
            total >> 20
        );
    }

    match run(&devices) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\n{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(devices: &[sys::ggml_backend_dev_t]) -> Result<(), String> {
    // must happen before any graph runs: it is what makes ggml-cuda claim GGML_OP_MAP_CUSTOM1
    kernels::register();

    bench::run(devices)?;

    // the hook only exists in (our patched) ggml-cuda; any other backend can't run the node
    match devices.iter().find(|d| sys::is_cuda(**d)) {
        Some(cuda) => custom_op::run(*cuda)?,
        None => println!("\ncustom kernel: skipped, no CUDA device"),
    }

    quant::run(devices)?;
    Ok(())
}
