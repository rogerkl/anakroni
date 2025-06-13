//! In-place temporal cross-synthesis - use existing magnitude with external phase

use super::TemporalOperation;
use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// In-place temporal cross-synthesis operation that keeps magnitude from the current
/// temporal FFT and replaces phase with phase from another temporal FFT.
///
/// This modifies the current analysis to have its original magnitude evolution
/// but with the phase evolution from the provided source.
pub struct TemporalCrossSynthesize {
    /// The temporal FFT to take phase from
    phase_source: TemporalFFTAnalysis,
}

impl TemporalCrossSynthesize {
    /// Create a new in-place temporal cross-synthesis operation
    pub fn new(phase_source: TemporalFFTAnalysis) -> Self {
        Self { phase_source }
    }
}

impl TemporalOperation for TemporalCrossSynthesize {
    fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
        // Validate compatibility
        if analysis.num_channels != self.phase_source.num_channels {
            log::warn!(
                "Channel count mismatch: current={}, phase={}. Using minimum.",
                analysis.num_channels,
                self.phase_source.num_channels
            );
        }

        if analysis.num_frequency_bins != self.phase_source.num_frequency_bins {
            log::warn!(
                "Frequency bin count mismatch: current={}, phase={}. Using minimum.",
                analysis.num_frequency_bins,
                self.phase_source.num_frequency_bins
            );
        }

        if analysis.config.fft_size != self.phase_source.config.fft_size {
            log::warn!(
                "Temporal FFT size mismatch: current={}, phase={}",
                analysis.config.fft_size,
                self.phase_source.config.fft_size
            );
        }

        let min_channels = analysis.num_channels.min(self.phase_source.num_channels);
        let min_freq_bins = analysis
            .num_frequency_bins
            .min(self.phase_source.num_frequency_bins);

        log::info!(
            "Applying in-place temporal cross-synthesis: {} channels, {} frequency bins",
            min_channels,
            min_freq_bins
        );
        log::info!("Keeping magnitude from current analysis, taking phase from external source");

        // Process each frequency bin
        for bin_fft in analysis.bin_ffts.iter_mut() {
            // Skip if outside processable range
            if bin_fft.channel_index >= min_channels || bin_fft.bin_index >= min_freq_bins {
                continue;
            }

            // Find corresponding bin in phase source
            if let Some(phase_bin) = self.phase_source.bin_ffts.iter().find(|b| {
                b.channel_index == bin_fft.channel_index && b.bin_index == bin_fft.bin_index
            }) {
                // Apply cross-synthesis in-place
                cross_synthesize_in_place(
                    &mut bin_fft.temporal_spectrum,
                    &phase_bin.temporal_spectrum,
                );
            } else {
                log::warn!(
                    "Could not find matching phase bin for channel {} bin {}",
                    bin_fft.channel_index,
                    bin_fft.bin_index
                );
            }
        }
    }

    fn name(&self) -> &'static str {
        "Temporal Cross-Synthesis (In-Place)"
    }

    fn description(&self) -> String {
        format!(
            "Cross-synthesize keeping current magnitude, phase from {} channels/{} bins",
            self.phase_source.num_channels, self.phase_source.num_frequency_bins
        )
    }
}

/// Perform in-place cross-synthesis on temporal spectrum
fn cross_synthesize_in_place(spectrum: &mut Vec<Complex64>, phase_source: &Vec<Complex64>) {
    let min_len = spectrum.len().min(phase_source.len());

    // Keep our magnitude, take phase from source
    for i in 0..min_len {
        let magnitude = spectrum[i].norm();
        let phase = phase_source[i].arg();

        // Combine our magnitude with source phase
        spectrum[i] = Complex64::from_polar(magnitude, phase);
    }

    // Ensure DC and Nyquist remain real
    // For DC, keep our magnitude but ensure phase is 0 or π based on source
    if !spectrum.is_empty() && !phase_source.is_empty() {
        let dc_mag = spectrum[0].norm();
        let dc_sign = if phase_source[0].re >= 0.0 { 1.0 } else { -1.0 };
        spectrum[0] = Complex64::new(dc_mag * dc_sign, 0.0);
    }

    // Handle Nyquist
    if spectrum.len() % 2 == 0 {
        let nyquist_idx = spectrum.len() / 2;
        if nyquist_idx < phase_source.len() {
            let nyquist_mag = spectrum[nyquist_idx].norm();
            let nyquist_sign = if phase_source[nyquist_idx].re >= 0.0 {
                1.0
            } else {
                -1.0
            };
            spectrum[nyquist_idx] = Complex64::new(nyquist_mag * nyquist_sign, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_fft::{TemporalBinFFT, TemporalFFTConfig};

    fn create_test_temporal_fft(magnitude: f64, phase_pattern: bool) -> TemporalFFTAnalysis {
        let config = TemporalFFTConfig::new(16).unwrap();
        let mut bin_ffts = Vec::new();

        let spectrum = if phase_pattern {
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),  // 90 degrees
                Complex64::new(-1.0, 0.0), // 180 degrees
                Complex64::new(0.0, -1.0), // 270 degrees
            ]
        } else {
            vec![
                Complex64::new(magnitude, 0.0),
                Complex64::new(magnitude * 0.8, magnitude * 0.2),
                Complex64::new(magnitude * 0.6, -magnitude * 0.4),
                Complex64::new(magnitude * 0.4, magnitude * 0.3),
            ]
        };

        bin_ffts.push(TemporalBinFFT {
            bin_index: 0,
            channel_index: 0,
            temporal_spectrum: spectrum,
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
    fn test_in_place_cross_synthesis() {
        let mut target = create_test_temporal_fft(10.0, false);
        let phase_source = create_test_temporal_fft(1.0, true);

        // Store original magnitudes
        let original_mags: Vec<f64> = target.bin_ffts[0]
            .temporal_spectrum
            .iter()
            .map(|c| c.norm())
            .collect();

        let cross_synth = TemporalCrossSynthesize::new(phase_source);
        cross_synth.apply(&mut target);

        // Check that magnitudes are preserved
        for (i, &orig_mag) in original_mags.iter().enumerate() {
            let new_mag = target.bin_ffts[0].temporal_spectrum[i].norm();
            assert!((new_mag - orig_mag).abs() < 1e-10);
        }

        // Check that phases come from phase source (90 degrees for bin 1)
        assert!(
            (target.bin_ffts[0].temporal_spectrum[1].arg() - std::f64::consts::FRAC_PI_2).abs()
                < 1e-10
        );

        // DC should be real
        assert_eq!(target.bin_ffts[0].temporal_spectrum[0].im, 0.0);
    }
}
