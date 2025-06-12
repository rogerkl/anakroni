//! Temporal FFT analysis implementation

use super::core::{TemporalBinFFT, TemporalFFTAnalysis, TemporalFFTConfig};
use crate::stft::STFTFrame;
use crate::Result;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Analyzer for performing temporal FFT on STFT frames
pub struct TemporalFFTAnalyzer {
    config: TemporalFFTConfig,
    fft_planner: FftPlanner<f64>,
    fft_forward: Arc<dyn Fft<f64>>,
}

impl TemporalFFTAnalyzer {
    /// Create a new temporal FFT analyzer
    pub fn new(config: TemporalFFTConfig) -> Self {
        let mut fft_planner = FftPlanner::new();
        let fft_forward = fft_planner.plan_fft_forward(config.fft_size);

        Self {
            config,
            fft_planner,
            fft_forward,
        }
    }

    /// Analyze STFT frames to create temporal FFT representation for all channels
    pub fn analyze(&self, stft_frames: &[Vec<STFTFrame>]) -> Result<TemporalFFTAnalysis> {
        if stft_frames.is_empty() {
            return Err("No STFT frames provided".to_string());
        }

        let num_channels = stft_frames.len();
        let num_time_frames = stft_frames[0].len();

        if num_time_frames == 0 {
            return Err("No time frames in STFT data".to_string());
        }

        let num_frequency_bins = stft_frames[0][0].spectrum.len();

        // Verify all channels have consistent dimensions
        for (ch_idx, channel_frames) in stft_frames.iter().enumerate() {
            if channel_frames.len() != num_time_frames {
                return Err(format!(
                    "Channel {} has {} frames, expected {}",
                    ch_idx,
                    channel_frames.len(),
                    num_time_frames
                ));
            }

            for (frame_idx, frame) in channel_frames.iter().enumerate() {
                if frame.spectrum.len() != num_frequency_bins {
                    return Err(format!(
                        "Channel {}, frame {} has {} bins, expected {}",
                        ch_idx,
                        frame_idx,
                        frame.spectrum.len(),
                        num_frequency_bins
                    ));
                }
            }
        }

        // Calculate effective length with multiplier
        let effective_length = self.config.effective_length(num_time_frames);

        log::info!(
            "Performing temporal FFT analysis: {} channels, {} time frames, {} frequency bins",
            num_channels,
            num_time_frames,
            num_frequency_bins
        );

        if self.config.length_multiplier > 1 {
            log::info!(
                "Using length multiplier {}: effective length {} -> FFT size {}",
                self.config.length_multiplier,
                effective_length,
                self.config.fft_size
            );
        }

        let mut bin_ffts = Vec::with_capacity(num_channels * num_frequency_bins);

        // Process each channel separately
        for (ch_idx, channel_frames) in stft_frames.iter().enumerate() {
            log::info!(
                "Processing temporal FFT for channel {}/{}",
                ch_idx + 1,
                num_channels
            );

            // Process each frequency bin in this channel
            for bin_idx in 0..num_frequency_bins {
                log::debug!(
                    "Channel {}, processing frequency bin {}/{}",
                    ch_idx,
                    bin_idx + 1,
                    num_frequency_bins
                );

                // Extract the temporal evolution of this frequency bin for this channel
                let mut temporal_data = Vec::with_capacity(self.config.fft_size);

                if self.config.repeat_data && self.config.length_multiplier > 1 {
                    // Repeat frame data according to multiplier
                    for _multiplier_idx in 0..self.config.length_multiplier {
                        for frame in channel_frames {
                            temporal_data.push(frame.spectrum[bin_idx]);
                        }
                    }
                } else {
                    // Default behavior: just use original frames once
                    for frame in channel_frames {
                        temporal_data.push(frame.spectrum[bin_idx]);
                    }
                }

                // Pad to required FFT size if needed
                let padding_needed = self.config.fft_size.saturating_sub(temporal_data.len());
                if padding_needed > 0 {
                    // Zero padding
                    temporal_data.resize(self.config.fft_size, Complex64::new(0.0, 0.0));
                }

                // Perform FFT on temporal data
                let mut temporal_spectrum = temporal_data;
                self.fft_forward.process(&mut temporal_spectrum);

                bin_ffts.push(TemporalBinFFT {
                    bin_index: bin_idx,
                    channel_index: ch_idx,
                    temporal_spectrum,
                    original_frames: num_time_frames,
                });
            }
        }

        log::info!(
            "Temporal FFT analysis complete: {} channels, {} frequency bins processed",
            num_channels,
            num_frequency_bins
        );

        Ok(TemporalFFTAnalysis {
            bin_ffts,
            config: self.config,
            num_frequency_bins,
            num_channels,
        })
    }
}
