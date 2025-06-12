//! Per-bin power amplitude operation - modify dynamic range per temporal spectrum

use super::TemporalOperation;
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Temporal power amplitude per-bin operation
///
/// This operation modifies the dynamic range of temporal frequency amplitudes
/// independently for each frequency bin. Unlike the global version, this preserves
/// the relative amplitude differences between frequency bins while modifying the
/// dynamic range within each bin's temporal evolution.
///
/// Factor < 1.0: Expands dynamic range within each bin
/// Factor = 1.0: No change
/// Factor > 1.0: Compresses dynamic range within each bin
pub struct TemporalPowerAmplitudePerBin {
    /// Power factor to apply to normalized amplitudes
    factor: f64,
}

impl TemporalPowerAmplitudePerBin {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl TemporalOperation for TemporalPowerAmplitudePerBin {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        if self.factor == 1.0 {
            // No change needed
            return;
        }

        log::info!(
            "Applying temporal power amplitude per-bin with factor {} to {} channels, {} frequency bins",
            self.factor,
            analysis.num_channels,
            analysis.num_frequency_bins
        );

        let mut bins_processed = 0;
        let mut bins_skipped = 0;

        // Process each frequency bin independently
        for bin_fft in &mut analysis.bin_ffts {
            // Find the maximum amplitude for this specific temporal spectrum
            let mut max_amplitude = 0.0f64;

            for &complex_val in &bin_fft.temporal_spectrum {
                let amplitude = complex_val.norm();
                if amplitude > max_amplitude {
                    max_amplitude = amplitude;
                }
            }

            // If max amplitude is zero or very small, skip this bin
            if max_amplitude < 1e-12 {
                log::debug!(
                    "Skipping channel {} bin {} (max amplitude near zero)",
                    bin_fft.channel_index,
                    bin_fft.bin_index
                );
                bins_skipped += 1;
                continue;
            }

            // Apply power scaling to this temporal spectrum
            apply_power_to_spectrum_normalized(
                &mut bin_fft.temporal_spectrum,
                self.factor,
                max_amplitude,
            );

            bins_processed += 1;

            log::debug!(
                "Processed channel {} bin {}: max_amplitude={:.6}",
                bin_fft.channel_index,
                bin_fft.bin_index,
                max_amplitude
            );
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
            "Per-bin power amplitude complete: factor={}, type={}, bins_processed={}, bins_skipped={}",
            self.factor,
            compression_type,
            bins_processed,
            bins_skipped
        );
    }

    fn name(&self) -> &'static str {
        "Temporal Power Amplitude Per-Bin"
    }

    fn description(&self) -> String {
        let effect = if self.factor < 1.0 {
            "expand dynamic range per bin"
        } else if self.factor > 1.0 {
            "compress dynamic range per bin"
        } else {
            "no change"
        };

        format!(
            "Apply power {} to temporal amplitudes independently per frequency bin ({})",
            self.factor, effect
        )
    }
}

/// Apply power scaling to a single temporal spectrum using its own maximum
fn apply_power_to_spectrum_normalized(
    spectrum: &mut Vec<Complex64>,
    factor: f64,
    max_amplitude: f64,
) {
    for complex_val in spectrum.iter_mut() {
        // Convert to polar form
        let (mut amplitude, phase) = complex_val.to_polar();

        // Skip if amplitude is very small
        if amplitude < 1e-12 {
            continue;
        }

        // Normalize amplitude to [0, 1] range using this spectrum's maximum
        let normalized = amplitude / max_amplitude;

        // Apply power scaling
        let scaled_normalized = normalized.powf(factor);

        // Scale back to original range
        amplitude = scaled_normalized * max_amplitude;

        // Convert back to rectangular form
        *complex_val = Complex64::from_polar(amplitude, phase);
    }

    // Ensure DC and Nyquist remain real
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

        // Create two bins with different amplitude ranges
        // Bin 0: Low amplitude range
        let low_amp_spectrum = vec![
            Complex64::new(0.1, 0.0),   // DC
            Complex64::new(0.05, 0.05), // Low
            Complex64::new(0.2, 0.1),   // High (relative to this bin)
            Complex64::new(0.01, 0.01), // Very low
        ];

        // Bin 1: High amplitude range
        let high_amp_spectrum = vec![
            Complex64::new(1.0, 0.0), // DC
            Complex64::new(0.5, 0.5), // Low (relative to this bin)
            Complex64::new(2.0, 1.0), // High
            Complex64::new(0.1, 0.1), // Very low (relative to this bin)
        ];

        bin_ffts.push(TemporalBinFFT {
            bin_index: 0,
            channel_index: 0,
            temporal_spectrum: low_amp_spectrum,
            original_frames: 16,
        });

        bin_ffts.push(TemporalBinFFT {
            bin_index: 1,
            channel_index: 0,
            temporal_spectrum: high_amp_spectrum,
            original_frames: 16,
        });

        TemporalFFTAnalysis {
            bin_ffts,
            config,
            num_frequency_bins: 2,
            num_channels: 1,
        }
    }

    #[test]
    fn test_per_bin_independence() {
        let mut analysis = create_test_analysis();

        // Store original max amplitudes for each bin
        let max_amp_0 = analysis.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));
        let max_amp_1 = analysis.bin_ffts[1]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));

        // Apply per-bin compression
        let power_op = TemporalPowerAmplitudePerBin::new(2.0);
        power_op.apply(&mut analysis);

        // Check that each bin's max amplitude is preserved
        let new_max_0 = analysis.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));
        let new_max_1 = analysis.bin_ffts[1]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |max, amp| max.max(amp));

        assert!((max_amp_0 - new_max_0).abs() < 1e-10);
        assert!((max_amp_1 - new_max_1).abs() < 1e-10);

        // Verify the two bins still have different amplitude ranges
        assert!(new_max_1 > new_max_0 * 5.0); // Bin 1 should still be much louder
    }

    #[test]
    fn test_per_bin_expansion() {
        let mut analysis = create_test_analysis();

        // Apply expansion
        let power_op = TemporalPowerAmplitudePerBin::new(0.5);
        power_op.apply(&mut analysis);

        // Check that DC is still real for both bins
        assert_eq!(analysis.bin_ffts[0].temporal_spectrum[0].im, 0.0);
        assert_eq!(analysis.bin_ffts[1].temporal_spectrum[0].im, 0.0);
    }

    #[test]
    fn test_per_bin_identity() {
        let mut analysis = create_test_analysis();
        let original_0 = analysis.bin_ffts[0].temporal_spectrum.clone();
        let original_1 = analysis.bin_ffts[1].temporal_spectrum.clone();

        // Apply identity (factor = 1)
        let power_op = TemporalPowerAmplitudePerBin::new(1.0);
        power_op.apply(&mut analysis);

        // Should be unchanged
        for (i, (&orig, &new)) in original_0
            .iter()
            .zip(analysis.bin_ffts[0].temporal_spectrum.iter())
            .enumerate()
        {
            assert!(
                (orig - new).norm() < 1e-10,
                "Bin 0 value at index {} changed: {:?} != {:?}",
                i,
                orig,
                new
            );
        }

        for (i, (&orig, &new)) in original_1
            .iter()
            .zip(analysis.bin_ffts[1].temporal_spectrum.iter())
            .enumerate()
        {
            assert!(
                (orig - new).norm() < 1e-10,
                "Bin 1 value at index {} changed: {:?} != {:?}",
                i,
                orig,
                new
            );
        }
    }

    #[test]
    fn test_zero_bin_handling() {
        let mut analysis = create_test_analysis();

        // Set one bin to all zeros
        analysis.bin_ffts[0].temporal_spectrum = vec![Complex64::new(0.0, 0.0); 4];

        // Should not crash when processing
        let power_op = TemporalPowerAmplitudePerBin::new(2.0);
        power_op.apply(&mut analysis);

        // Zero bin should remain zero
        assert!(analysis.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .all(|&c| c.norm() < 1e-12));
    }
}
