//! Temporal convolution operation - convolve with another temporal FFT

use super::TemporalOperation;
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Temporal convolution operation that multiplies temporal spectra with another temporal FFT
///
/// In the frequency domain, convolution becomes multiplication, so this operation
/// multiplies each temporal frequency bin with the corresponding bin from another
/// temporal FFT analysis.
pub struct TemporalConvolve {
    /// The temporal FFT to convolve with
    impulse_temporal_fft: TemporalFFTAnalysis,
}

impl TemporalConvolve {
    /// Create a new temporal convolution operation
    pub fn new(impulse_temporal_fft: TemporalFFTAnalysis) -> Self {
        Self {
            impulse_temporal_fft,
        }
    }
}

impl TemporalOperation for TemporalConvolve {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        // Validate that the analyses are compatible
        if analysis.num_channels != self.impulse_temporal_fft.num_channels {
            log::warn!(
                "Channel count mismatch: {} vs {}. Using minimum channel count.",
                analysis.num_channels,
                self.impulse_temporal_fft.num_channels
            );
        }

        if analysis.num_frequency_bins != self.impulse_temporal_fft.num_frequency_bins {
            log::warn!(
                "Frequency bin count mismatch: {} vs {}. Using minimum bin count.",
                analysis.num_frequency_bins,
                self.impulse_temporal_fft.num_frequency_bins
            );
        }

        if analysis.config.fft_size != self.impulse_temporal_fft.config.fft_size {
            log::warn!(
                "Temporal FFT size mismatch: {} vs {}",
                analysis.config.fft_size,
                self.impulse_temporal_fft.config.fft_size
            );
        }

        let min_channels = analysis
            .num_channels
            .min(self.impulse_temporal_fft.num_channels);
        let min_freq_bins = analysis
            .num_frequency_bins
            .min(self.impulse_temporal_fft.num_frequency_bins);

        log::info!(
            "Applying temporal convolution: {} channels, {} frequency bins",
            min_channels,
            min_freq_bins
        );

        // Process each frequency bin
        for bin_fft in analysis.bin_ffts.iter_mut() {
            // Skip if this bin is outside the range we can process
            if bin_fft.channel_index >= min_channels || bin_fft.bin_index >= min_freq_bins {
                continue;
            }

            // Find the corresponding bin in the impulse temporal FFT
            if let Some(impulse_bin) = self.impulse_temporal_fft.bin_ffts.iter().find(|b| {
                b.channel_index == bin_fft.channel_index && b.bin_index == bin_fft.bin_index
            }) {
                // Convolve (multiply in frequency domain)
                convolve_temporal_spectra(
                    &mut bin_fft.temporal_spectrum,
                    &impulse_bin.temporal_spectrum,
                );
            } else {
                log::warn!(
                    "Could not find matching impulse bin for channel {} bin {}",
                    bin_fft.channel_index,
                    bin_fft.bin_index
                );
            }
        }
    }

    fn name(&self) -> &'static str {
        "Temporal Convolve"
    }

    fn description(&self) -> String {
        format!(
            "Convolve with temporal FFT ({} channels, {} frequency bins)",
            self.impulse_temporal_fft.num_channels, self.impulse_temporal_fft.num_frequency_bins
        )
    }
}

/// Convolve (multiply) two temporal spectra
fn convolve_temporal_spectra(spectrum: &mut Vec<Complex64>, impulse: &Vec<Complex64>) {
    let min_len = spectrum.len().min(impulse.len());

    // Multiply corresponding bins
    for i in 0..min_len {
        spectrum[i] *= impulse[i];
    }

    // If the impulse is shorter, the remaining bins are multiplied by zero
    // (which means they become zero - this is correct convolution behavior)
    if impulse.len() < spectrum.len() {
        for i in impulse.len()..spectrum.len() {
            spectrum[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Ensure DC and Nyquist remain real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
    if spectrum.len() % 2 == 0 {
        let nyquist_idx = spectrum.len() / 2;
        let nyquist_real = spectrum[nyquist_idx].re;
        spectrum[nyquist_idx] = Complex64::new(nyquist_real, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_fft::{TemporalBinFFT, TemporalFFTConfig};

    fn create_test_temporal_fft(num_channels: usize, num_freq_bins: usize) -> TemporalFFTAnalysis {
        let config = TemporalFFTConfig::new(16).unwrap();
        let mut bin_ffts = Vec::new();

        for ch in 0..num_channels {
            for bin in 0..num_freq_bins {
                let mut spectrum = vec![Complex64::new(1.0, 0.0); 16];
                // Add some variation
                spectrum[1] = Complex64::new(2.0, 0.5);
                spectrum[2] = Complex64::new(1.5, -0.3);

                bin_ffts.push(TemporalBinFFT {
                    bin_index: bin,
                    channel_index: ch,
                    temporal_spectrum: spectrum,
                    original_frames: 16,
                });
            }
        }

        TemporalFFTAnalysis {
            bin_ffts,
            config,
            num_frequency_bins: num_freq_bins,
            num_channels,
        }
    }

    #[test]
    fn test_temporal_convolve() {
        let mut analysis = create_test_temporal_fft(1, 4);
        let impulse = create_test_temporal_fft(1, 4);

        // Store original for comparison
        let original_dc = analysis.bin_ffts[0].temporal_spectrum[0];

        // Apply convolution
        let convolve = TemporalConvolve::new(impulse);
        convolve.apply(&mut analysis);

        // Check that values changed
        assert_ne!(original_dc, analysis.bin_ffts[0].temporal_spectrum[0]);

        // Check that DC is still real
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);
    }

    #[test]
    fn test_convolve_mismatched_sizes() {
        let mut analysis = create_test_temporal_fft(2, 8);
        let impulse = create_test_temporal_fft(1, 4); // Fewer channels and bins

        // Should still work, using minimum counts
        let convolve = TemporalConvolve::new(impulse);
        convolve.apply(&mut analysis);

        // Only first channel and first 4 bins should be processed
        // Others should remain unchanged
        assert_eq!(
            analysis.bin_ffts[7].temporal_spectrum[1],
            Complex64::new(2.0, 0.5)
        );
    }
}
