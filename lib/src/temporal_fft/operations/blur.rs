//! Temporal blur operation - smooths temporal frequency spectrum

use super::{process_all_temporal_spectra, TemporalOperation};
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Temporal blur operation that smooths the temporal evolution by averaging adjacent temporal frequency bins
pub struct TemporalBlur {
    /// Blur radius (number of adjacent bins to average)
    radius: usize,
    /// Blur strength (0.0 = no blur, 1.0 = full averaging)
    strength: f64,
}

impl TemporalBlur {
    pub fn new(radius: usize, strength: f64) -> Self {
        Self { radius, strength }
    }
}

impl TemporalOperation for TemporalBlur {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        let fft_size = analysis.config.fft_size;

        log::info!(
            "Applying temporal blur with radius {} and strength {}",
            self.radius,
            self.strength
        );

        process_all_temporal_spectra(analysis, |_ch_idx, _bin_idx, temporal_spectrum| {
            blur_spectrum(temporal_spectrum, self.radius, self.strength, fft_size);
        });
    }

    fn name(&self) -> &'static str {
        "Temporal Blur"
    }

    fn description(&self) -> String {
        format!(
            "Blur temporal frequencies with radius {} and strength {:.2}",
            self.radius, self.strength
        )
    }
}

/// Apply blur to a single spectrum
fn blur_spectrum(spectrum: &mut Vec<Complex64>, radius: usize, strength: f64, fft_size: usize) {
    if radius == 0 || strength == 0.0 {
        return; // No blur needed
    }

    // Clamp strength to valid range
    let strength = strength.clamp(0.0, 1.0);

    // Create a copy for reading original values
    let original = spectrum.clone();

    // Apply blur to each bin
    for i in 0..fft_size {
        let mut sum = Complex64::new(0.0, 0.0);
        let mut count = 0;

        // Average with neighboring bins
        for offset in 0..=radius {
            // Add bins on both sides
            if offset == 0 {
                sum += original[i];
                count += 1;
            } else {
                // Positive offset
                if i + offset < fft_size {
                    sum += original[i + offset];
                    count += 1;
                }
                // Negative offset
                if i >= offset {
                    sum += original[i - offset];
                    count += 1;
                }
            }
        }

        // Apply averaged value with strength
        if count > 0 {
            let averaged = sum / count as f64;
            spectrum[i] = original[i] * (1.0 - strength) + averaged * strength;
        }
    }

    // Ensure DC and Nyquist remain real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
    if fft_size % 2 == 0 {
        spectrum[fft_size / 2] = Complex64::new(spectrum[fft_size / 2].re, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::STFTFrame;
    use crate::temporal_fft::{TemporalFFTAnalyzer, TemporalFFTConfig};

    fn create_test_analysis() -> TemporalFFTAnalysis {
        // Create test STFT frames
        let frames = vec![vec![
            STFTFrame {
                spectrum: vec![Complex64::new(1.0, 0.0); 32],
                frame_index: 0,
                time_position: 0,
            };
            16
        ]];

        let config = TemporalFFTConfig::new(16).unwrap();
        let analyzer = TemporalFFTAnalyzer::new(config);
        analyzer.analyze(&frames).unwrap()
    }

    #[test]
    fn test_temporal_blur() {
        let mut analysis = create_test_analysis();
        let blur = TemporalBlur::new(1, 0.5);

        // Store original for comparison
        let original_spectrum = analysis.bin_ffts[0].temporal_spectrum.clone();

        // Apply blur
        blur.apply(&mut analysis);

        // Verify changes were made
        let blurred_spectrum = &analysis.bin_ffts[0].temporal_spectrum;
        assert_ne!(original_spectrum, *blurred_spectrum);

        // Verify DC is still real
        assert_eq!(blurred_spectrum[0].im, 0.0);
    }

    #[test]
    fn test_blur_no_effect() {
        let mut analysis = create_test_analysis();

        // Test zero radius
        let blur = TemporalBlur::new(0, 0.5);
        let original = analysis.bin_ffts[0].temporal_spectrum.clone();
        blur.apply(&mut analysis);
        assert_eq!(original, analysis.bin_ffts[0].temporal_spectrum);

        // Test zero strength
        let blur = TemporalBlur::new(3, 0.0);
        blur.apply(&mut analysis);
        assert_eq!(original, analysis.bin_ffts[0].temporal_spectrum);
    }

    #[test]
    fn test_blur_strength_clamping() {
        let mut analysis = create_test_analysis();

        // Test strength > 1.0 (should be clamped to 1.0)
        let blur = TemporalBlur::new(2, 1.5);
        blur.apply(&mut analysis);

        // Should not crash and should apply blur
        let blurred_spectrum = &analysis.bin_ffts[0].temporal_spectrum;
        assert_eq!(blurred_spectrum[0].im, 0.0); // DC still real
    }
}
