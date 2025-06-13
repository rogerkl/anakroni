//! Anakroni Library
//!
//! A library for processing audio data using Short-Time Fourier Transform (STFT).
//! Provides functionality for loading, analyzing, and reconstructing audio data
//! using overlapping windowed FFT analysis.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

pub mod audio_io;
pub mod processor;
pub mod spectrogram;
pub mod stft;
pub mod temporal_fft; // Now a module directory
pub mod utils;
pub mod window;

pub use num_complex::Complex64;
pub use processor::STFTProcessor;
pub use rustfft; // Re-export rustfft for external use if needed

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the library
///
/// Sets up logging and other initialization for the library.
/// For WASM targets, this will set up browser-specific error handling.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn init() {
    #[cfg(feature = "wasm")]
    {
        console_error_panic_hook::set_once();
    }

    // Initialize logging
    #[cfg(all(not(target_arch = "wasm32"), feature = "env_logger"))]
    {
        env_logger::init();
    }
}

/// Result type for audio processing operations
pub type Result<T> = std::result::Result<T, String>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
        assert!(true);
    }
}
