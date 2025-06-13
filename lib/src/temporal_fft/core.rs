//! Core types and structures for temporal FFT analysis

use crate::Result;
use num_complex::Complex64;

/// Configuration for temporal FFT analysis
#[derive(Debug, Clone, Copy)]
pub struct TemporalFFTConfig {
    /// FFT size for temporal analysis (must be power of 2)
    pub fft_size: usize,
    /// Length multiplier for extending FFT size beyond frame count
    pub length_multiplier: usize,
    /// Whether to repeat frame data when using multiplier (default: false = zero padding)
    pub repeat_data: bool,
}

impl TemporalFFTConfig {
    /// Create a new temporal FFT configuration
    pub fn new(fft_size: usize) -> Result<Self> {
        if !fft_size.is_power_of_two() || fft_size < 2 {
            return Err(format!(
                "FFT size must be a power of 2 and at least 2, got {}",
                fft_size
            ));
        }

        Ok(Self {
            fft_size,
            length_multiplier: 1,
            repeat_data: false,
        })
    }

    /// Create a new temporal FFT configuration with length multiplier
    pub fn new_with_multiplier(fft_size: usize, length_multiplier: usize) -> Result<Self> {
        Self::new_with_options(fft_size, length_multiplier, false)
    }

    /// Create a new temporal FFT configuration with full options
    pub fn new_with_options(
        fft_size: usize,
        length_multiplier: usize,
        repeat_data: bool,
    ) -> Result<Self> {
        if !fft_size.is_power_of_two() || fft_size < 2 {
            return Err(format!(
                "FFT size must be a power of 2 and at least 2, got {}",
                fft_size
            ));
        }

        if length_multiplier == 0 {
            return Err("Length multiplier must be at least 1".to_string());
        }

        Ok(Self {
            fft_size,
            length_multiplier,
            repeat_data,
        })
    }

    /// Calculate the next power of 2 that fits the given number of frames
    pub fn next_power_of_2(frames: usize) -> usize {
        if frames == 0 {
            return 2;
        }
        let mut size = 1;
        while size < frames {
            size *= 2;
        }
        size
    }

    /// Create a configuration with automatic size calculation and optional multiplier
    pub fn auto_size_with_multiplier(num_frames: usize, length_multiplier: usize) -> Self {
        Self::auto_size_with_options(num_frames, length_multiplier, false)
    }

    /// Create a configuration with automatic size calculation and full options
    pub fn auto_size_with_options(
        num_frames: usize,
        length_multiplier: usize,
        repeat_data: bool,
    ) -> Self {
        let base_fft_size = Self::next_power_of_2(num_frames);
        let final_fft_size = if repeat_data {
            // If repeating data, we need to fit the repeated frames
            Self::next_power_of_2(num_frames * length_multiplier)
        } else {
            // If zero padding, we size for the original frames but allow larger FFT
            Self::next_power_of_2(num_frames * length_multiplier)
        };

        Self {
            fft_size: final_fft_size,
            length_multiplier,
            repeat_data,
        }
    }

    /// Create a configuration with automatic size calculation (multiplier = 1)
    pub fn auto_size(num_frames: usize, zero_padding: bool) -> Self {
        Self::auto_size_with_multiplier(num_frames, 1)
    }

    /// Get the effective length after applying multiplier
    pub fn effective_length(&self, original_frames: usize) -> usize {
        original_frames * self.length_multiplier
    }
}

impl Default for TemporalFFTConfig {
    fn default() -> Self {
        Self {
            fft_size: 256,
            length_multiplier: 1,
            repeat_data: false,
        }
    }
}

/// Temporal FFT data for a single frequency bin across all time frames
#[derive(Debug, Clone)]
pub struct TemporalBinFFT {
    /// Original frequency bin index
    pub bin_index: usize,
    /// Channel index this bin belongs to
    pub channel_index: usize,
    /// Temporal FFT spectrum (complex values representing temporal frequencies)
    pub temporal_spectrum: Vec<Complex64>,
    /// Number of original time frames (before padding)
    pub original_frames: usize,
}

/// Complete temporal FFT analysis containing all channels and frequency bins
#[derive(Debug, Clone)]
pub struct TemporalFFTAnalysis {
    /// Temporal FFT for each frequency bin across all channels
    /// Format: bin_ffts[channel_idx * num_frequency_bins + bin_idx]
    pub bin_ffts: Vec<TemporalBinFFT>,
    /// Configuration used for analysis
    pub config: TemporalFFTConfig,
    /// Number of frequency bins from original STFT
    pub num_frequency_bins: usize,
    /// Number of channels
    pub num_channels: usize,
}

impl TemporalFFTAnalysis {
    /// Get the temporal frequency resolution
    pub fn temporal_frequency_resolution(&self) -> f64 {
        // This represents the resolution in the "temporal frequency" domain
        // The actual interpretation depends on the STFT hop size and sample rate
        1.0 / self.config.fft_size as f64
    }

    /// Get the number of temporal frequency bins
    pub fn num_temporal_bins(&self) -> usize {
        self.config.fft_size
    }

    /// Get temporal spectrum for a specific frequency bin in a specific channel
    pub fn get_bin_temporal_spectrum(
        &self,
        channel: usize,
        bin_index: usize,
    ) -> Option<&Vec<Complex64>> {
        self.bin_ffts
            .iter()
            .find(|bin_fft| bin_fft.channel_index == channel && bin_fft.bin_index == bin_index)
            .map(|bin_fft| &bin_fft.temporal_spectrum)
    }

    /// Get temporal magnitude spectrum for a specific frequency bin in a specific channel
    pub fn get_bin_temporal_magnitudes(
        &self,
        channel: usize,
        bin_index: usize,
    ) -> Option<Vec<f64>> {
        self.get_bin_temporal_spectrum(channel, bin_index)
            .map(|spectrum| spectrum.iter().map(|c| c.norm()).collect())
    }

    /// Get the effective number of temporal bins considering the multiplier
    pub fn effective_temporal_bins(&self) -> usize {
        self.config
            .effective_length(self.bin_ffts[0].original_frames)
    }

    /// Get a description of the temporal FFT configuration including multiplier effects
    pub fn describe_config(&self) -> String {
        let original_frames = if !self.bin_ffts.is_empty() {
            self.bin_ffts[0].original_frames
        } else {
            0
        };

        let effective_bins = self.effective_temporal_bins();

        if self.config.length_multiplier > 1 {
            format!(
                "Temporal FFT Configuration:\n\
                 - Original frames: {}\n\
                 - Length multiplier: {}x\n\
                 - Data mode: {}\n\
                 - Effective output length: {}\n\
                 - FFT size: {}\n\
                 - Temporal frequency resolution: {:.6}\n\
                 - Zero padding: {} samples",
                original_frames,
                self.config.length_multiplier,
                if self.config.repeat_data {
                    "Repeat frames"
                } else {
                    "Zero padding"
                },
                effective_bins,
                self.config.fft_size,
                self.temporal_frequency_resolution(),
                self.config
                    .fft_size
                    .saturating_sub(if self.config.repeat_data {
                        original_frames * self.config.length_multiplier
                    } else {
                        original_frames
                    })
            )
        } else {
            format!(
                "Temporal FFT Configuration:\n\
                 - Frames: {}\n\
                 - FFT size: {}\n\
                 - Temporal frequency resolution: {:.6}\n\
                 - Zero padding: {} samples",
                original_frames,
                self.config.fft_size,
                self.temporal_frequency_resolution(),
                self.config.fft_size.saturating_sub(original_frames)
            )
        }
    }

    /// Generate a 2D visualization of temporal FFT data
    /// Returns magnitude data organized as [frequency_bins x temporal_bins]
    pub fn to_temporal_spectrogram(&self, channel: usize, use_db: bool) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.num_frequency_bins);

        // For each frequency bin
        for freq_bin in 0..self.num_frequency_bins {
            // Get the temporal spectrum for this frequency bin
            if let Some(temporal_spectrum) = self.get_bin_temporal_spectrum(channel, freq_bin) {
                // Convert complex values to magnitudes
                let magnitudes: Vec<f64> = temporal_spectrum
                    .iter()
                    .map(|c| {
                        let mag = c.norm();
                        if use_db && mag > 0.0 {
                            20.0 * mag.log10()
                        } else {
                            mag
                        }
                    })
                    .collect();
                result.push(magnitudes);
            } else {
                // If bin not found, create empty row
                result.push(vec![0.0; self.config.fft_size]);
            }
        }

        result
    }

    /// Get the frequency (in normalized units) corresponding to a temporal FFT bin
    /// Returns values from 0.0 to 1.0, where 1.0 represents the Nyquist frequency
    pub fn temporal_bin_to_normalized_frequency(&self, bin: usize) -> f64 {
        let fft_size = self.config.fft_size;

        if bin == 0 {
            0.0 // DC
        } else if bin <= fft_size / 2 {
            // Positive frequencies: bin / (fft_size/2) gives 0.0 to 1.0
            bin as f64 / (fft_size as f64 / 2.0)
        } else {
            // Negative frequencies: map to negative normalized frequencies
            // bin N-1 corresponds to -1/N, bin N-2 to -2/N, etc.
            let neg_bin = fft_size - bin;
            -(neg_bin as f64 / (fft_size as f64 / 2.0))
        }
    }
}
