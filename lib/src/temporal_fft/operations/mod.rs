//! Temporal FFT operations module

mod basic;
mod blur; // Add this
mod convolve; // Add this
mod cross_synthesis;
mod phase_multiply;
mod power; // Add this
mod power_bin; // Add this
mod shift;
mod stretch;
mod utils;

// Re-export operations
pub use basic::{TemporalHighpass, TemporalLowpass, ZeroTemporalDC};
pub use blur::TemporalBlur; // Add this
pub use convolve::TemporalConvolve; // Add this
pub use cross_synthesis::TemporalCrossSynthesize;
pub use phase_multiply::{
    DispersiveTemporalPhaseMultiply, TemporalPhaseMultiply, TemporalPhaseReversal,
    TemporalPhaseScrambling,
};
pub use power::TemporalPowerAmplitude; // Add this
pub use power_bin::TemporalPowerAmplitudePerBin; // Add this
pub use shift::{CircularTemporalShift, DispersiveTemporalShift, TemporalShift};
pub use stretch::TemporalStretch;

pub use utils::{enforce_hermitian_symmetry, process_all_temporal_spectra};

use crate::temporal_fft::TemporalFFTAnalysis;

/// Trait for temporal FFT operations
pub trait TemporalOperation {
    /// Apply the operation to the temporal FFT analysis
    fn apply(&self, analysis: &mut TemporalFFTAnalysis);

    /// Get the name of the operation
    fn name(&self) -> &'static str;

    /// Get a description of the operation and its parameters
    fn description(&self) -> String;
}

// Convenience methods on TemporalFFTAnalysis
impl TemporalFFTAnalysis {
    /// Apply a temporal operation using the trait interface
    pub fn apply_operation(&mut self, operation: &dyn TemporalOperation) {
        operation.apply(self);
    }

    // Direct convenience methods for common operations

    /// Zero out temporal DC component for all bins
    pub fn zero_temporal_dc(&mut self) {
        ZeroTemporalDC.apply(self);
    }

    /// Apply temporal highpass filter
    pub fn temporal_highpass(&mut self, cutoff_normalized: f64) {
        TemporalHighpass::new(cutoff_normalized).apply(self);
    }

    /// Apply temporal lowpass filter
    pub fn temporal_lowpass(&mut self, cutoff_normalized: f64) {
        TemporalLowpass::new(cutoff_normalized).apply(self);
    }

    /// Apply temporal shift
    pub fn shift(&mut self, shift_frames: i32) {
        TemporalShift::new(shift_frames).apply(self);
    }

    /// Apply circular temporal shift
    pub fn shift_circular(&mut self, shift_frames: i32) {
        CircularTemporalShift::new(shift_frames).apply(self);
    }

    /// Apply dispersive temporal shift
    pub fn shift_dispersive(&mut self, base_shift: i32, freq_factor: f64) {
        DispersiveTemporalShift::new(base_shift, freq_factor).apply(self);
    }

    /// Apply temporal stretch
    pub fn stretch(&mut self, stretch_factor: f64) {
        TemporalStretch::new(stretch_factor).apply(self);
    }

    /// Apply phase multiplication
    pub fn phase_multiply(&mut self, factor: f64) {
        TemporalPhaseMultiply::new(factor).apply(self);
    }

    /// Apply dispersive phase multiplication
    pub fn phase_multiply_dispersive(&mut self, base_factor: f64, freq_scaling: f64) {
        DispersiveTemporalPhaseMultiply::new(base_factor, freq_scaling).apply(self);
    }

    /// Apply phase reversal
    pub fn phase_reverse(&mut self) {
        TemporalPhaseReversal.apply(self);
    }

    /// Apply phase scrambling
    pub fn phase_scramble(&mut self, intensity: f64, seed: u64) {
        TemporalPhaseScrambling::new(intensity, seed).apply(self);
    }

    /// Apply temporal blur
    pub fn blur(&mut self, radius: usize, strength: f64) {
        TemporalBlur::new(radius, strength).apply(self);
    }

    /// Apply power amplitude scaling
    pub fn power_amplitude(&mut self, factor: f64) {
        TemporalPowerAmplitude::new(factor).apply(self);
    }

    /// Apply power amplitude scaling per bin
    pub fn power_amplitude_per_bin(&mut self, factor: f64) {
        TemporalPowerAmplitudePerBin::new(factor).apply(self);
    }

    /// Apply temporal cross-synthesis with another temporal FFT
    pub fn cross_synthesize(&mut self, phase_source: TemporalFFTAnalysis) {
        TemporalCrossSynthesize::new(phase_source).apply(self);
    }
}
