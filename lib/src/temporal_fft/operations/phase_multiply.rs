//! Phase multiplication operations

use super::{process_all_temporal_spectra, TemporalOperation};
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Basic phase multiplication
pub struct TemporalPhaseMultiply {
    factor: f64,
}

impl TemporalPhaseMultiply {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl TemporalOperation for TemporalPhaseMultiply {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        log::info!(
            "Applying temporal phase multiply with factor {} to {} bins across {} channels",
            self.factor,
            analysis.num_frequency_bins,
            analysis.num_channels
        );

        process_all_temporal_spectra(analysis, |_channel_idx, _bin_idx, temporal_spectrum| {
            apply_phase_multiply_to_spectrum(temporal_spectrum, self.factor);
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Phase Multiply"
    }

    fn description(&self) -> String {
        format!("Multiply all phases by factor {}", self.factor)
    }
}

/// Dispersive phase multiplication (frequency-dependent)
pub struct DispersiveTemporalPhaseMultiply {
    base_factor: f64,
    freq_scaling: f64,
}

impl DispersiveTemporalPhaseMultiply {
    pub fn new(base_factor: f64, freq_scaling: f64) -> Self {
        Self {
            base_factor,
            freq_scaling,
        }
    }
}

impl TemporalOperation for DispersiveTemporalPhaseMultiply {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        log::info!(
            "Applying dispersive temporal phase multiply with base factor {} and frequency scaling {}",
            self.base_factor,
            self.freq_scaling
        );

        let num_frequency_bins = analysis.num_frequency_bins;

        for bin_fft in analysis.bin_ffts.iter_mut() {
            // Calculate frequency-dependent factor
            let freq_ratio = bin_fft.bin_index as f64 / (num_frequency_bins as f64 - 1.0);
            let factor = self.base_factor + (self.freq_scaling * freq_ratio);

            apply_phase_multiply_to_spectrum(&mut bin_fft.temporal_spectrum, factor);

            log::debug!(
                "Applied dispersive phase multiply (factor: {:.3}) to channel {}, frequency bin {}",
                factor,
                bin_fft.channel_index,
                bin_fft.bin_index
            );
        }
    }

    fn name(&self) -> &'static str {
        "Dispersive Temporal Phase Multiply"
    }

    fn description(&self) -> String {
        format!(
            "Frequency-dependent phase multiply: base factor {}, scaling {}",
            self.base_factor, self.freq_scaling
        )
    }
}

/// Phase reversal (multiply by -1)
pub struct TemporalPhaseReversal;

impl TemporalOperation for TemporalPhaseReversal {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        TemporalPhaseMultiply::new(-1.0).apply(analysis);
    }

    fn name(&self) -> &'static str {
        "Temporal Phase Reversal"
    }

    fn description(&self) -> String {
        "Reverse all phases (multiply by -1.0)".to_string()
    }
}

/// Phase scrambling with pseudo-random pattern
pub struct TemporalPhaseScrambling {
    intensity: f64,
    seed: u64,
}

impl TemporalPhaseScrambling {
    pub fn new(intensity: f64, seed: u64) -> Self {
        Self { intensity, seed }
    }
}

impl TemporalOperation for TemporalPhaseScrambling {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        log::info!(
            "Applying temporal phase scrambling with intensity {} and seed {}",
            self.intensity,
            self.seed
        );

        // Simple LCG (Linear Congruential Generator) for reproducible pseudo-random values
        let mut rng_state = self.seed;

        for bin_fft in analysis.bin_ffts.iter_mut() {
            // Generate pseudo-random factor for this frequency bin
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_factor = ((rng_state >> 16) as f64) / 32768.0 - 1.0; // Range [-1, 1]

            let factor = 1.0 + self.intensity * random_factor;

            apply_phase_multiply_to_spectrum(&mut bin_fft.temporal_spectrum, factor);
        }
    }

    fn name(&self) -> &'static str {
        "Temporal Phase Scrambling"
    }

    fn description(&self) -> String {
        format!(
            "Pseudo-random phase scrambling with intensity {} and seed {}",
            self.intensity, self.seed
        )
    }
}

/// Core phase multiplication implementation for a single temporal spectrum
///
/// This is the low-level function that performs the actual phase multiplication
/// on a temporal spectrum, following the same approach as mammut-fft but adapted
/// for complex FFT with both positive and negative frequencies.
pub fn apply_phase_multiply_to_spectrum(temporal_spectrum: &mut Vec<Complex64>, factor: f64) {
    let fft_size = temporal_spectrum.len();
    let nyquist_bin = fft_size / 2;

    // Process each temporal frequency bin
    for temp_bin_idx in 0..fft_size {
        // Skip DC (bin 0) and Nyquist (bin N/2) components
        // These should remain real-valued for proper FFT structure
        if temp_bin_idx == 0 || temp_bin_idx == nyquist_bin {
            // Ensure DC and Nyquist components are real by zeroing imaginary part
            temporal_spectrum[temp_bin_idx] =
                Complex64::new(temporal_spectrum[temp_bin_idx].re, 0.0);
            continue;
        }

        // Convert to polar coordinates
        let (amplitude, phase) = temporal_spectrum[temp_bin_idx].to_polar();

        // Multiply phase by factor and normalize to [-π, π] range
        let new_phase = normalize_phase(phase * factor);

        // Convert back to rectangular coordinates
        temporal_spectrum[temp_bin_idx] = Complex64::from_polar(amplitude, new_phase);
    }
}

/// Utility function to normalize a phase value to the range [-π, π].
pub fn normalize_phase(phase: f64) -> f64 {
    // Normalize the phase to the range [-π, π]
    // First get it to [0, 2π) with the modulo
    let normalized_phase = ((phase % (2.0 * std::f64::consts::PI)) + (2.0 * std::f64::consts::PI))
        % (2.0 * std::f64::consts::PI);

    // Then shift values above π to the [-π, π] range
    if normalized_phase > std::f64::consts::PI {
        normalized_phase - 2.0 * std::f64::consts::PI
    } else {
        normalized_phase
    }
}
