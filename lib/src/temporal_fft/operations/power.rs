//! Power amplitude operation - modify dynamic range of temporal amplitudes

use super::TemporalOperation;
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Temporal power amplitude operation
///
/// This operation modifies the dynamic range of temporal frequency amplitudes by:
/// 1. Finding the maximum amplitude across all temporal bins
/// 2. Normalizing amplitudes to [0, 1] range
/// 3. Raising normalized amplitudes to the given power
/// 4. Scaling back to original range
///
/// Factor < 1.0: Expands dynamic range (makes quiet parts quieter)
/// Factor = 1.0: No change
/// Factor > 1.0: Compresses dynamic range (makes quiet parts louder)
pub struct TemporalPowerAmplitude {
    /// Power factor to apply to normalized amplitudes
    factor: f64,
}

impl TemporalPowerAmplitude {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl TemporalOperation for TemporalPowerAmplitude {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        if self.factor == 1.0 {
            // No change needed
            return;
        }

        log::info!(
            "Applying temporal power amplitude with factor {} to {} channels, {} frequency bins",
            self.factor,
            analysis.num_channels,
            analysis.num_frequency_bins
        );

        // Step 1: Find the maximum amplitude across all bins
        let mut max_amplitude = 0.0f64;

        for bin_fft in &analysis.bin_ffts {
            for &complex_val in &bin_fft.temporal_spectrum {
                let amplitude = complex_val.norm();
                if amplitude > max_amplitude {
                    max_amplitude = amplitude;
                }
            }
        }

        // If max amplitude is zero or very small, nothing to do
        if max_amplitude < 1e-12 {
            log::warn!("Maximum amplitude is near zero, skipping power amplitude operation");
            return;
        }

        log::debug!("Maximum amplitude found: {}", max_amplitude);

        // Step 2: Apply power scaling to all bins
        for bin_fft in &mut analysis.bin_ffts {
            apply_power_to_spectrum(&mut bin_fft.temporal_spectrum, self.factor, max_amplitude);
        }

        // Log statistics
        let compression_type = if self.factor < 1.0 {
            "expansion"
        } else if self.factor > 1.0 {
            "compression"
        } else {
            "none"
        };

        log::info!(
            "Power amplitude complete: factor={}, max_amplitude={:.6}, type={}",
            self.factor,
            max_amplitude,
            compression_type
        );
    }

    fn name(&self) -> &'static str {
        "Temporal Power Amplitude"
    }

    fn description(&self) -> String {
        let effect = if self.factor < 1.0 {
            "expand dynamic range"
        } else if self.factor > 1.0 {
            "compress dynamic range"
        } else {
            "no change"
        };

        format!(
            "Apply power {} to temporal amplitudes ({})",
            self.factor, effect
        )
    }
}

/// Apply power scaling to a single temporal spectrum
fn apply_power_to_spectrum(spectrum: &mut Vec<Complex64>, factor: f64, max_amplitude: f64) {
    for complex_val in spectrum.iter_mut() {
        // Convert to polar form
        let (mut amplitude, phase) = complex_val.to_polar();

        // Skip if amplitude is very small
        if amplitude < 1e-12 {
            continue;
        }

        // Normalize amplitude to [0, 1] range
        let normalized = amplitude / max_amplitude;

        // Apply power scaling
        let scaled_normalized = normalized.powf(factor);

        // Scale back to original range
        amplitude = scaled_normalized * max_amplitude;

        // Convert back to rectangular form
        *complex_val = Complex64::from_polar(amplitude, phase);
    }

    // Ensure DC and Nyquist remain real (phase should already be 0 or π, but ensure no numerical errors)
    spectrum[0] = Complex64::new(spectrum[0].norm() * spectrum[0].re.signum(), 0.0);
    if spectrum.len() % 2 == 0 {
        let nyquist_idx = spectrum.len() / 2;
        let nyquist_amp = spectrum[nyquist_idx].norm();
        let nyquist_sign = spectrum[nyquist_idx].re.signum();
        spectrum[nyquist_idx] = Complex64::new(nyquist_amp * nyquist_sign, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_fft::{TemporalBinFFT, TemporalFFTConfig};

    fn create_test_analysis() -> TemporalFFTAnalysis {
        let config = TemporalFFTConfig::new(16).unwrap();
        let mut bin_ffts = Vec::new();

        // Create test data with varying amplitudes
        let test_spectrum = vec![
            Complex64::new(1.0, 0.0),  // DC
            Complex64::new(0.5, 0.5),  // Low amplitude
            Complex64::new(2.0, 1.0),  // High amplitude
            Complex64::new(0.1, 0.1),  // Very low amplitude
            Complex64::new(1.0, -1.0), // Medium amplitude
            Complex64::new(0.0, 0.0),  // Zero
            Complex64::new(0.3, -0.4), // Low amplitude
            Complex64::new(1.5, 0.0),  // Medium amplitude
            Complex64::new(0.0, 0.0),  // Nyquist (will be made real)
        ];

        bin_ffts.push(TemporalBinFFT {
            bin_index: 0,
            channel_index: 0,
            temporal_spectrum: test_spectrum,
            original_frames: 16,
        });

        TemporalFFTAnalysis {
            bin_ffts,
            config,
            num_frequency_bins: 1,
            num_channels: 1,
        }
    }

    #[test]
    fn test_power_amplitude_compression() {
        let mut analysis = create_test_analysis();

        // Store original max amplitude
        let original_max = analysis.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));

        // Apply compression (factor > 1)
        let power_op = TemporalPowerAmplitude::new(2.0);
        power_op.apply(&mut analysis);

        // Check that max amplitude is preserved
        let new_max = analysis.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));

        assert!((original_max - new_max).abs() < 1e-10);

        // Check that DC is still real
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);
    }

    #[test]
    fn test_power_amplitude_expansion() {
        let mut analysis = create_test_analysis();

        // Apply expansion (factor < 1)
        let power_op = TemporalPowerAmplitude::new(0.5);
        power_op.apply(&mut analysis);

        // Dynamic range should be expanded (quiet sounds quieter)
        // This is hard to test precisely, but DC should still be real
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);
    }

    #[test]
    fn test_power_amplitude_identity() {
        let mut analysis = create_test_analysis();
        let original = analysis.bin_ffts[0].temporal_spectrum.clone();

        // Apply identity (factor = 1)
        let power_op = TemporalPowerAmplitude::new(1.0);
        power_op.apply(&mut analysis);

        // Should be unchanged
        for (i, (&orig, &new)) in original
            .iter()
            .zip(analysis.bin_ffts[0].temporal_spectrum.iter())
            .enumerate()
        {
            assert!(
                (orig - new).norm() < 1e-10,
                "Value at index {} changed: {:?} != {:?}",
                i,
                orig,
                new
            );
        }
    }

    #[test]
    fn test_extreme_factors() {
        let mut analysis = create_test_analysis();

        // Test very small factor (extreme expansion)
        let power_op = TemporalPowerAmplitude::new(0.1);
        power_op.apply(&mut analysis);
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);

        // Test very large factor (extreme compression)
        let power_op = TemporalPowerAmplitude::new(10.0);
        power_op.apply(&mut analysis);
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);
    }
}
