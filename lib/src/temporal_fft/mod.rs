//! Temporal FFT analysis for tracking frequency bin evolution over time
//!
//! This module implements a second-level FFT analysis that takes the STFT frames
//! and performs an FFT on the temporal evolution of each frequency bin.
//! This creates a frequency-time-frequency representation.

pub mod analyzer;
pub mod core;
pub mod operations;
pub mod statistics;
pub mod synthesizer;

#[cfg(not(target_arch = "wasm32"))]
pub mod visualization;

// Re-export main types for convenience
pub use analyzer::TemporalFFTAnalyzer;
pub use core::{TemporalBinFFT, TemporalFFTAnalysis, TemporalFFTConfig};
pub use statistics::{PhaseStatistics, TemporalStatistics};
pub use synthesizer::TemporalFFTSynthesizer;

// Re-export operations for convenience
pub use operations::{
    CircularTemporalShift,
    DispersiveTemporalPhaseMultiply,
    DispersiveTemporalShift,
    TemporalBlur,     // Add this
    TemporalConvolve, // Add this
    TemporalHighpass,
    TemporalLowpass,
    TemporalOperation,
    // Phase operations
    TemporalPhaseMultiply,
    TemporalPhaseReversal,
    TemporalPhaseScrambling,
    TemporalPowerAmplitude,       // Add this
    TemporalPowerAmplitudePerBin, // Add this
    // Shift operations
    TemporalShift,
    // Stretch operations
    TemporalStretch,
    // Basic operations
    ZeroTemporalDC,
};
