#!/bin/bash
# Build script for the WebAssembly module

echo "Building STFT Audio Processor WASM module..."

# Navigate to the web directory
cd "$(dirname "$0")"

# Build the library with wasm features
echo "Building library with WASM features..."
cargo build --release -p stft_audio_lib --features wasm

# Build the WebAssembly module
echo "Building WASM module..."
wasm-pack build --release --target web --out-dir pkg

echo "WASM build completed successfully!"
echo "WASM files are in: web/pkg/"