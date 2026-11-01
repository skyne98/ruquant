//! Binds and builds everything C/C++ for ruquant:
//!   1. bindgen over OUR ggml headers (third_party/llama.cpp, patched, built into build-ggml/)
//!   2. our kernels: kernels/*.cu via nvcc (the hook that hands them to ggml is src/kernels.rs)
//!   3. kernels/compile_flags.txt so clangd resolves the same headers the compilers get
//!
//! Prerequisite: ./build-ggml.sh. This script never builds ggml itself.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bin = root.join("build-ggml/bin");
    if !bin.join("libggml.so").exists() {
        panic!(
            "build-ggml/bin/libggml.so not found — run ./build-ggml.sh first \
             (it builds our patched ggml tree; this crate only links it)"
        );
    }

    let ggml = root.join("third_party/llama.cpp/ggml");
    let include = ggml.join("include");
    let src = ggml.join("src");
    let cuda = env::var("CUDA_PATH").ok();

    emit_bindings(&root, &[&include, &src], cuda.as_deref());
    build_kernels(&root);
    write_clangd_flags(&root, &[&include, &src], cuda.as_deref());
    link_ggml(&root, &bin, cuda.as_deref());
}

fn emit_bindings(root: &Path, includes: &[&PathBuf], cuda: Option<&str>) {
    // bindings.h is a source file: writing it here would make it newer than this run and
    // force build.rs (bindgen + nvcc) to rerun on every cargo build.
    let header = root.join("bindings.h");

    let mut builder = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("ggml.*")
        .allowlist_type("ggml.*")
        .allowlist_var("GGML.*")
        .allowlist_var("ggml.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .layout_tests(false);

    for dir in includes {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }
    if let Some(cuda) = cuda {
        builder = builder.clang_arg(format!("-I{cuda}/include"));
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    builder
        .generate()
        .expect("bindgen failed (run inside nix develop, LIBCLANG_PATH must be set)")
        .write_to_file(out.join("bindings.rs"))
        .unwrap();
}

fn build_kernels(root: &Path) {
    // -arch=native: only this machine's GPU, matching ggml's CMAKE_CUDA_ARCHITECTURES=native
    cc::Build::new()
        .cuda(true)
        .file(root.join("kernels/rq_scale.cu"))
        .flag("-arch=native")
        .opt_level(2)
        .compile("rq_scale");
}

/// clangd gets no shell environment, so the same include paths the compilers receive are
/// written out literally. System dirs are queried from g++ once (libstdc++ is implicit to g++
/// and invisible to clang otherwise).
fn write_clangd_flags(root: &Path, includes: &[&PathBuf], cuda: Option<&str>) {
    let mut lines: Vec<String> = includes
        .iter()
        .map(|d| format!("-I{}", d.display()))
        .collect();
    if let Some(cuda) = cuda {
        lines.push(format!("-I{cuda}/include"));
    }
    lines.push("-std=c++17".into());

    let probe = Command::new("g++")
        .args(["-xc++", "-E", "-Wp,-v", "-"])
        .stdin(std::process::Stdio::null())
        .output();

    if let Ok(out) = probe {
        let text = String::from_utf8_lossy(&out.stderr);
        let mut in_list = false;
        for line in text.lines() {
            if line.starts_with("#include <...> search starts here:") {
                in_list = true;
            } else if line.starts_with("End of search list") {
                break;
            } else if in_list && !line.contains("framework") {
                lines.push(format!("-idirafter{}", line.trim()));
            }
        }
    }

    let _ = fs::write(
        root.join("kernels/compile_flags.txt"),
        lines.join("\n") + "\n",
    );
}

fn link_ggml(root: &Path, bin: &Path, cuda: Option<&str>) {
    println!("cargo:rustc-link-search=native={}", bin.display());
    for lib in ["ggml", "ggml-base", "ggml-cuda", "ggml-cpu", "ggml-vulkan"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bin.display());

    if let Some(cuda) = cuda {
        println!("cargo:rustc-link-search=native={cuda}/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib");
    }

    for f in ["bindings.h", "kernels/rq_scale.cu"] {
        println!("cargo:rerun-if-changed={}", root.join(f).display());
    }
}
