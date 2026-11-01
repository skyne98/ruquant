{
  description = "ruquant — ggml kernels on CPU / CUDA / Vulkan";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # cudatoolkit
      };

      # ggml-vulkan's build (find_package(Vulkan) + glslc) expects one SDK root.
      # nb: binaries live in shaderc's `bin` output, headers in the `dev` outputs.
      vulkanSdk = pkgs.symlinkJoin {
        name = "vulkan-sdk";
        paths = with pkgs; [
          vulkan-headers
          vulkan-loader
          (lib.getDev vulkan-loader)
          (lib.getBin shaderc)
          (lib.getDev shaderc)
          spirv-headers
        ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rustc
          cargo
          rustfmt
          clippy
          cmake
          ninja
          clang
          llvmPackages.libclang
          pkg-config
          python3

          cudaPackages.cudatoolkit

          vulkanSdk
          vulkan-tools
        ];

        env = {
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          VULKAN_SDK = "${vulkanSdk}";
          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
          CMAKE_PREFIX_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          # bindgen needs libclang and libc headers, neither on the default NixOS path
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.glibc.dev}/include";

          # dev build = this machine's GPU only.
          #  - target-cpu=native -> sys crate sets GGML_NATIVE=ON -> CMAKE_CUDA_ARCHITECTURES=native (sm_86)
          #  - -L: rustc resolves cargo:rustc-link-lib itself, so the CUDA lib dir must be an -L flag
          RUSTFLAGS = "-C target-cpu=native -L ${pkgs.cudaPackages.cudatoolkit}/lib";

          # ggml auto-detects sccache on PATH and makes it the launcher for nvcc too,
          # which dies with "cannot find binary path" on Nix wrapper scripts.
          GGML_CCACHE = "OFF";
        };

        shellHook = ''
          echo "ruquant devshell — build: cargo build --release, run: ./run.sh"
        '';
      };
    };
}
