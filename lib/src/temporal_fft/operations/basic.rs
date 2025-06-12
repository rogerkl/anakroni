//! Basic temporal FFT operations (DC removal, filters)

use super::{process_all_temporal_spectra, TemporalOperation};
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Zero out temporal DC component
pub struct ZeroTemporalDC;

impl TemporalOperation for ZeroTemporalDC {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        process_all_temporal_spectra(analysis, |_ch_idx, _bin_idx, temporal_spectrum| {
            if !temporal_spectrum.is_empty() {
                temporal_spectrum[0] = Complex64::new(0.0, 0.0);
            }
        });
    }

    fn name(&self) -> &'static str {
        "Zero Temporal DC"
    }

    fn description(&self) -> String {
        "Removes the DC component from all temporal frequency spectra".to_string()
    }
}

/// Temporal highpass filter
pub struct TemporalHighpass {
    cutoff_normalized: f64,
}

impl TemporalHighpass {
    pub fn new(cutoff_normalized: f64) -> Self {
        Self { cutoff_normalized }
    }
}

impl TemporalOperation for TemporalHighpass {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;
        let cutoff_bin = (self.cutoff_normalized * (fft_size as f64 / 2.0)) as usize;

        process_all_temporal_spectra(analysis, |_ch_idx, _bin_idx, temporal_spectrum| {
            // For complex FFT: [DC, pos_freq..., Nyquist, neg_freq...]
            // Highpass: zero out DC and low positive/negative frequencies

            // Zero DC component
            temporal_spectrum[0] = Complex64::new(0.0, 0.0);

            // Zero low positive frequencies (1 to cutoff_bin)
            for i in 1..=cutoff_bin.min(fft_size / 2) {
                let logical_freq = array_index_to_logical_freq(i, fft_size);
                temporal_spectrum[i] = Complex64::new(0.0, 0.0);
                // Zero corresponding low negative frequencies
                if let Some(neg_index) = logical_freq_to_array_index(-logical_freq, fft_size) {
                    temporal_spectrum[neg_index] = Complex64::new(0.0, 0.0);
                }
            }
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Highpass"
    }

    fn description(&self) -> String {
        format!(
            "Temporal highpass filter with cutoff at {:.3} normalized frequency",
            self.cutoff_normalized
        )
    }
}

/// Temporal lowpass filter
pub struct TemporalLowpass {
    cutoff_normalized: f64,
}

impl TemporalLowpass {
    pub fn new(cutoff_normalized: f64) -> Self {
        Self { cutoff_normalized }
    }
}

impl TemporalOperation for TemporalLowpass {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;
        let cutoff_bin = (self.cutoff_normalized * (fft_size as f64 / 2.0)) as usize;

        process_all_temporal_spectra(analysis, |_ch_idx, _bin_idx, temporal_spectrum| {
            // For complex FFT: [DC, pos_freq..., Nyquist, neg_freq...]
            // Lowpass: zero out high positive/negative frequencies, keep DC and low freqs

            // Zero high positive frequencies (cutoff_bin+1 to N/2)
            for i in (cutoff_bin + 1)..=(fft_size / 2) {
                let logical_freq = array_index_to_logical_freq(i, fft_size);
                temporal_spectrum[i] = Complex64::new(0.0, 0.0);
                // Zero corresponding high negative frequencies
                if let Some(neg_index) = logical_freq_to_array_index(-logical_freq, fft_size) {
                    temporal_spectrum[neg_index] = Complex64::new(0.0, 0.0);
                }
            }
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Lowpass"
    }

    fn description(&self) -> String {
        format!(
            "Temporal lowpass filter with cutoff at {:.3} normalized frequency",
            self.cutoff_normalized
        )
    }
}

// Helper functions from original implementation

/// Convert array index to logical frequency index
fn array_index_to_logical_freq(index: usize, fft_size: usize) -> i32 {
    if index <= fft_size / 2 {
        index as i32
    } else {
        (index as i32) - (fft_size as i32)
    }
}

/// Convert logical frequency index to array index
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
