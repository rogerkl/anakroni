//! Temporal shift operations

use super::{enforce_hermitian_symmetry, process_all_temporal_spectra, TemporalOperation};
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Basic temporal shift operation
pub struct TemporalShift {
    shift_frames: i32,
}

impl TemporalShift {
    pub fn new(shift_frames: i32) -> Self {
        Self { shift_frames }
    }
}

impl TemporalOperation for TemporalShift {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;

        log::info!(
            "Applying temporal shift of {} frames to all {} bins across {} channels",
            self.shift_frames,
            analysis.num_frequency_bins,
            analysis.num_channels
        );

        process_all_temporal_spectra(analysis, |_channel_idx, _bin_idx, temporal_spectrum| {
            shift_complex_spectrum(temporal_spectrum, self.shift_frames, fft_size);
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Shift"
    }

    fn description(&self) -> String {
        let direction = if self.shift_frames > 0 {
            "forward"
        } else {
            "backward"
        };
        format!(
            "Shift temporal evolution by {} frames {} in time",
            self.shift_frames.abs(),
            direction
        )
    }
}

/// Circular temporal shift (wrap-around)
pub struct CircularTemporalShift {
    shift_frames: i32,
}

impl CircularTemporalShift {
    pub fn new(shift_frames: i32) -> Self {
        Self { shift_frames }
    }
}

impl TemporalOperation for CircularTemporalShift {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;

        log::info!(
            "Applying circular temporal shift of {} frames",
            self.shift_frames
        );

        process_all_temporal_spectra(analysis, |_channel_idx, _bin_idx, temporal_spectrum| {
            circular_shift_complex_spectrum(temporal_spectrum, self.shift_frames, fft_size);
        });
    }

    fn name(&self) -> &'static str {
        "Circular Temporal Shift"
    }

    fn description(&self) -> String {
        format!(
            "Circular shift of {} frames with wrap-around",
            self.shift_frames
        )
    }
}

/// Dispersive temporal shift (frequency-dependent)
pub struct DispersiveTemporalShift {
    base_shift: i32,
    freq_factor: f64,
}

impl DispersiveTemporalShift {
    pub fn new(base_shift: i32, freq_factor: f64) -> Self {
        Self {
            base_shift,
            freq_factor,
        }
    }
}

impl TemporalOperation for DispersiveTemporalShift {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;
        let num_frequency_bins = analysis.num_frequency_bins;

        log::info!(
            "Applying dispersive temporal shift: base={}, factor={}",
            self.base_shift,
            self.freq_factor
        );

        for bin_fft in analysis.bin_ffts.iter_mut() {
            // Calculate frequency-dependent shift
            let freq_normalized = bin_fft.bin_index as f64 / num_frequency_bins as f64;
            let shift_amount = self.base_shift as f64
                + (self.freq_factor * freq_normalized * self.base_shift as f64);
            let bin_shift = shift_amount.round() as i32;

            shift_complex_spectrum(&mut bin_fft.temporal_spectrum, bin_shift, fft_size);
        }
    }

    fn name(&self) -> &'static str {
        "Dispersive Temporal Shift"
    }

    fn description(&self) -> String {
        format!(
            "Frequency-dependent shift: base {} frames, factor {:.2}",
            self.base_shift, self.freq_factor
        )
    }
}

// Internal shift functions

/// Internal function to shift a complex spectrum preserving positive/negative frequency relationships
fn shift_complex_spectrum(spectrum: &mut Vec<Complex64>, shift: i32, fft_size: usize) {
    if shift == 0 {
        return;
    }

    // Create temporary storage for shifted data
    let mut shifted = vec![Complex64::new(0.0, 0.0); fft_size];

    // Process all frequency bins
    for src_idx in 0..fft_size {
        let logical_freq = array_index_to_logical_freq(src_idx, fft_size);

        if logical_freq == 0 {
            // Always preserve DC at index 0
            shifted[0] = spectrum[0];
        } else {
            // Apply shift to the logical frequency
            if let Some(shifted_logical_freq) = shifted_logical_freq(logical_freq, shift) {
                if let Some(dst_idx) = logical_freq_to_array_index(shifted_logical_freq, fft_size) {
                    // Convert back to array index if within valid range
                    shifted[dst_idx] = spectrum[src_idx];
                }
            }
            // If out of range, the value remains zero (already initialized)
        }
    }

    // Ensure Hermitian symmetry for real-valued time domain signal
    enforce_hermitian_symmetry(&mut shifted);

    // Replace original spectrum with shifted data
    *spectrum = shifted;
}

/// Circular shift for complex spectrum
fn circular_shift_complex_spectrum(spectrum: &mut Vec<Complex64>, shift: i32, fft_size: usize) {
    if shift == 0 {
        return;
    }

    // Normalize shift to be within -fft_size to +fft_size
    let normalized_shift = shift % fft_size as i32;
    let actual_shift = if normalized_shift < 0 {
        (normalized_shift + fft_size as i32) as usize
    } else {
        normalized_shift as usize
    };

    // Create temporary storage
    let original = spectrum.clone();

    // Perform circular shift
    for i in 0..fft_size {
        let source_idx = (i + fft_size - actual_shift) % fft_size;
        spectrum[i] = original[source_idx];
    }

    // No need to enforce Hermitian symmetry for circular shift
    // as it preserves the existing symmetry
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

fn shifted_logical_freq(logical_freq: i32, shift: i32) -> Option<i32> {
    if logical_freq == 0 {
        None // DC component
    } else if logical_freq > 0 {
        let shifted = logical_freq + shift;
        if shifted < 0 {
            None
        } else {
            Some(shifted)
        }
    } else {
        let shifted = logical_freq - shift;
        if shifted >= 0 {
            None
        } else {
            Some(shifted)
        }
    }
}

// Public utility for getting shift effects description
impl TemporalFFTAnalysis {
    /// Get summary of shift operations for display
    pub fn describe_temporal_shift_effects(&self, shift_frames: i32) -> String {
        let time_shift_ms = (shift_frames as f64 * 1000.0) / 44100.0; // Assuming 44.1kHz

        format!(
            "Temporal Shift Summary:\n\
             - Shift amount: {} frames\n\
             - Time shift: {:.2} ms\n\
             - Direction: {}\n\
             - FFT size: {}\n\
             - Channels affected: {}\n\
             - Frequency bins per channel: {}",
            shift_frames,
            time_shift_ms,
            if shift_frames > 0 {
                "Forward (delay)"
            } else {
                "Backward (advance)"
            },
            self.config.fft_size,
            self.num_channels,
            self.num_frequency_bins
        )
    }
}
