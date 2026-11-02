//! ruquant: ggml + our kernels, one pipeline.
//!
//!   ./run.sh                              device demo: matmuls, the in-graph kernel, Q4_0
//!   ./run.sh convert IN.gguf OUT.gguf     a GGUF into a rotated W4A4 GGUF (see convert.rs)
//!   ./run.sh hessian MODEL TEXT DIR       GPTQ Hessians for convert --hessians (see hessian.rs)
//!   ./run.sh drafter DRAFT SRC OUT        a DFlash / DSpark drafter for a converted target (see convert.rs)
//! (inside nix develop, after ./build-llama.sh)

mod bench;
mod convert;
mod custom_op;
mod gguf;
mod gptq;
mod hessian;
mod kernels;
mod quant;
mod sys;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let tool: Option<fn(&[String]) -> Result<(), String>> = match args.get(1).map(String::as_str) {
        Some("convert") => Some(convert::main),
        Some("hessian") => Some(hessian::main),
        Some("drafter") => Some(convert::drafter_main),
        Some("head") => Some(convert::head_main),
        Some("setclip") => Some(convert::setclip_main),
        _ => None,
    };
    if let Some(tool) = tool {
        return match tool(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{}: {e}", args[1]);
                ExitCode::FAILURE
            }
        };
    }

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
