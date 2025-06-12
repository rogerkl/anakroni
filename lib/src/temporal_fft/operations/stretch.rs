//! Temporal stretch operations

use super::{enforce_hermitian_symmetry, process_all_temporal_spectra, TemporalOperation};
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Temporal stretch operation
pub struct TemporalStretch {
    stretch_factor: f64,
}

impl TemporalStretch {
    pub fn new(stretch_factor: f64) -> Self {
        Self { stretch_factor }
    }
}

impl TemporalOperation for TemporalStretch {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;

        if (self.stretch_factor - 1.0).abs() < 1e-10 {
            // No stretch needed
            return;
        }

        log::info!(
            "Applying temporal stretch factor {} to all {} bins across {} channels",
            self.stretch_factor,
            analysis.num_frequency_bins,
            analysis.num_channels
        );

        process_all_temporal_spectra(analysis, |_channel_idx, _bin_idx, temporal_spectrum| {
            stretch_complex_spectrum(temporal_spectrum, self.stretch_factor, fft_size);
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Stretch"
    }

    fn description(&self) -> String {
        if self.stretch_factor > 1.0 {
            format!("Expand temporal frequencies by {}x", self.stretch_factor)
        } else {
            format!("Compress temporal frequencies to {}x", self.stretch_factor)
        }
    }
}

/// Internal function to stretch a complex spectrum
///
/// This function handles the core stretching logic:
/// 1. Uses logical frequency indices to handle positive/negative frequencies correctly
/// 2. Interpolates fractional bin positions
/// 3. Accumulates contributions from multiple source bins to the same destination
/// 4. Preserves DC component
/// 5. Skips out-of-range destinations
fn stretch_complex_spectrum(spectrum: &mut Vec<Complex64>, stretch_factor: f64, fft_size: usize) {
    if (stretch_factor - 1.0).abs() < 1e-10 {
        return;
    }

    // Create temporary storage for stretched data, initialized to zero
    let mut stretched = vec![Complex64::new(0.0, 0.0); fft_size];

    // Always preserve DC component at index 0
    stretched[0] = spectrum[0];

    // Process all non-DC frequency bins
    for src_idx in 1..fft_size {
        let logical_freq = array_index_to_logical_freq(src_idx, fft_size);

        if logical_freq == 0 {
            continue; // Skip DC (already handled)
        }

        // Calculate stretched logical frequency
        let stretched_logical_freq_f64 = logical_freq as f64 * stretch_factor;

        // Handle positive and negative frequencies separately
        if logical_freq > 0 {
            // Positive frequency
            distribute_stretched_bin(
                spectrum[src_idx],
                stretched_logical_freq_f64,
                &mut stretched,
                fft_size,
                false, // not negative
            );
        } else {
            // Negative frequency
            distribute_stretched_bin(
                spectrum[src_idx],
                stretched_logical_freq_f64,
                &mut stretched,
                fft_size,
                true, // is negative
            );
        }
    }

    // Ensure Hermitian symmetry for real-valued time domain signal
    enforce_hermitian_symmetry(&mut stretched);

    // Replace original spectrum with stretched data
    *spectrum = stretched;
}

/// Distribute a stretched bin's contribution to the destination spectrum
///
/// Handles fractional bin positions through linear interpolation
fn distribute_stretched_bin(
    source_value: Complex64,
    target_logical_freq_f64: f64,
    destination: &mut Vec<Complex64>,
    fft_size: usize,
    is_negative: bool,
) {
    // Calculate the two adjacent integer logical frequencies
    let lower_logical_freq = target_logical_freq_f64.floor() as i32;
    let upper_logical_freq = target_logical_freq_f64.ceil() as i32;

    // Calculate interpolation weights
    let upper_weight = target_logical_freq_f64 - lower_logical_freq as f64;
    let lower_weight = 1.0 - upper_weight;

    // Distribute to lower bin if it's valid and in range
    if let Some(lower_idx) = logical_freq_to_array_index(lower_logical_freq, fft_size) {
        if is_logical_freq_in_valid_range(lower_logical_freq, fft_size, is_negative) {
            destination[lower_idx] += source_value * lower_weight;
        }
    }

    // Distribute to upper bin if it's different from lower and valid
    if upper_logical_freq != lower_logical_freq {
        if let Some(upper_idx) = logical_freq_to_array_index(upper_logical_freq, fft_size) {
            if is_logical_freq_in_valid_range(upper_logical_freq, fft_size, is_negative) {
                destination[upper_idx] += source_value * upper_weight;
            }
        }
    }
}

/// Check if a logical frequency is within the valid range for the FFT size
fn is_logical_freq_in_valid_range(logical_freq: i32, fft_size: usize, is_negative: bool) -> bool {
    if logical_freq == 0 {
        return true; // DC is always valid
    }

    let max_positive = (fft_size / 2) as i32;
    let min_negative = -((fft_size / 2) as i32) + 1;

    if is_negative {
        logical_freq >= min_negative && logical_freq < 0
    } else {
        logical_freq > 0 && logical_freq <= max_positive
    }
}

// Helper functions

fn array_index_to_logical_freq(index: usize, fft_size: usize) -> i32 {
    if index <= fft_size / 2 {
        index as i32
    } else {
        (index as i32) - (fft_size as i32)
    }
}

fn logical_freq_to_array_index(logical_freq: i32, fft_size: usize) -> Option<usize> {
    if logical_freq == 0 {
        Some(0)
    } else if logical_freq > 0 && logical_freq <= (fft_size / 2) as i32 {
        Some(logical_freq as usize)
    } else if logical_freq < 0 && logical_freq >= -(fft_size as i32 / 2) {
        Some((logical_freq + fft_size as i32) as usize)
    } else {
        None
    }
}

// Public utility for getting stretch effects description
impl TemporalFFTAnalysis {
    /// Get summary of stretch operations for display
    pub fn describe_temporal_stretch_effects(&self, stretch_factor: f64) -> String {
        let effect_description = if stretch_factor > 1.0 {
            format!("Expanding temporal frequencies by {}x", stretch_factor)
        } else if stretch_factor < 1.0 {
            format!("Compressing temporal frequencies to {}x", stretch_factor)
        } else {
            "No change (stretch factor = 1.0)".to_string()
        };

        format!(
            "Temporal Stretch Summary:\n\
             - Stretch factor: {}\n\
             - Effect: {}\n\
             - FFT size: {}\n\
             - Channels affected: {}\n\
             - Frequency bins per channel: {}\n\
             - DC component: Preserved\n\
             - Interpolation: Linear for fractional bin positions",
            stretch_factor,
            effect_description,
            self.config.fft_size,
            self.num_channels,
            self.num_frequency_bins
        )
    }
}
