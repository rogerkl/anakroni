//! Temporal FFT synthesis implementation

use super::core::{TemporalFFTAnalysis, TemporalFFTConfig};
use crate::stft::STFTFrame;
use crate::Result;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Synthesizer for converting temporal FFT back to STFT frames
pub struct TemporalFFTSynthesizer {
    config: TemporalFFTConfig,
    fft_planner: FftPlanner<f64>,
    fft_inverse: Arc<dyn Fft<f64>>,
}

impl TemporalFFTSynthesizer {
    /// Create a new temporal FFT synthesizer
    pub fn new(config: TemporalFFTConfig) -> Self {
        let mut fft_planner = FftPlanner::new();
        let fft_inverse = fft_planner.plan_fft_inverse(config.fft_size);

        Self {
            config,
            fft_planner,
            fft_inverse,
        }
    }

    /// Synthesize STFT frames from temporal FFT analysis for all channels - OPTIMIZED VERSION
    pub fn synthesize(
        &self,
        analysis: &TemporalFFTAnalysis,
        hop_size: usize,
    ) -> Result<Vec<Vec<STFTFrame>>> {
        if analysis.bin_ffts.is_empty() {
            return Err("No temporal FFT data to synthesize".to_string());
        }

        let num_channels = analysis.num_channels;
        let num_frequency_bins = analysis.num_frequency_bins;
        let original_frames = analysis.bin_ffts[0].original_frames;
        let effective_length = analysis.config.effective_length(original_frames);

        log::info!(
            "Synthesizing STFT frames from temporal FFT: {} channels, {} frequency bins, {} original frames",
            num_channels, num_frequency_bins, original_frames
        );

        if analysis.config.length_multiplier > 1 {
            log::info!(
                "Using length multiplier {}: synthesizing {} effective frames",
                analysis.config.length_multiplier,
                effective_length
            );
        }

        let mut all_channel_frames = Vec::with_capacity(num_channels);

        // Initialize all channels with empty frames for the effective length
        for ch_idx in 0..num_channels {
            let mut channel_frames: Vec<STFTFrame> = Vec::with_capacity(effective_length);
            for frame_idx in 0..effective_length {
                channel_frames.push(STFTFrame {
                    spectrum: vec![Complex64::new(0.0, 0.0); num_frequency_bins],
                    frame_index: frame_idx,
                    time_position: frame_idx * hop_size,
                });
            }
            all_channel_frames.push(channel_frames);
        }

        // Process each frequency bin once - do inverse FFT only once per bin
        for bin_fft in &analysis.bin_ffts {
            log::debug!(
                "Processing temporal synthesis for channel {}, frequency bin {}/{}",
                bin_fft.channel_index,
                bin_fft.bin_index + 1,
                num_frequency_bins
            );

            // Perform inverse FFT on temporal spectrum
            let mut temporal_data = bin_fft.temporal_spectrum.clone();
            self.fft_inverse.process(&mut temporal_data);

            // Normalize by FFT size (rustfft doesn't auto-normalize)
            let scale = 1.0 / (self.config.fft_size as f64);
            for sample in temporal_data.iter_mut() {
                *sample *= scale;
            }

            // Distribute the reconstructed temporal data to the appropriate frame bins
            let channel_frames = &mut all_channel_frames[bin_fft.channel_index];

            // Only use the first effective_length samples, ignoring any padding
            for frame_idx in 0..effective_length.min(temporal_data.len()) {
                if frame_idx < channel_frames.len() {
                    channel_frames[frame_idx].spectrum[bin_fft.bin_index] =
                        temporal_data[frame_idx];
                }
            }
        }

        // Fix Hermitian symmetry for all frames to ensure they can be inverse-transformed to real signals
        for channel_frames in &mut all_channel_frames {
            for frame in channel_frames.iter_mut() {
                Self::enforce_hermitian_symmetry(&mut frame.spectrum);
            }
        }

        log::info!(
            "Temporal synthesis complete: {} channels, {} frames per channel reconstructed",
            num_channels,
            effective_length
        );

        Ok(all_channel_frames)
    }

    /// Enforce Hermitian symmetry on a frequency spectrum to ensure real IFFT output
    /// For a real-valued signal, the FFT must satisfy: X[k] = X*[N-k] for k = 1..N/2-1
    /// Also, X[0] and X[N/2] (DC and Nyquist) must be real-valued
    fn enforce_hermitian_symmetry(spectrum: &mut [Complex64]) {
        if spectrum.is_empty() {
            return;
        }

        // Ensure DC component (bin 0) is real
        if spectrum[0].im.abs() > 1e-10 {
            log::debug!(
                "Fixing DC component: {} + {}i -> {} + 0i",
                spectrum[0].re,
                spectrum[0].im,
                spectrum[0].re
            );
        }
        spectrum[0] = Complex64::new(spectrum[0].re, 0.0);

        // Ensure Nyquist component (last bin) is real if it exists
        // For realfft, we typically have N/2+1 bins, so the last bin is the Nyquist frequency
        if spectrum.len() > 1 {
            let nyquist_idx = spectrum.len() - 1;
            if spectrum[nyquist_idx].im.abs() > 1e-10 {
                log::debug!(
                    "Fixing Nyquist component: {} + {}i -> {} + 0i",
                    spectrum[nyquist_idx].re,
                    spectrum[nyquist_idx].im,
                    spectrum[nyquist_idx].re
                );
            }
            spectrum[nyquist_idx] = Complex64::new(spectrum[nyquist_idx].re, 0.0);
        }

        // For the bins in between, ensure Hermitian symmetry if we had the full spectrum
        // However, since we're using realfft which only gives us the positive frequencies,
        // we don't need to explicitly enforce symmetry as realfft handles this internally.
        // The main issue is just ensuring DC and Nyquist are real.
    }

    /// Synthesize STFT frames using oscillator bank with parameter control
    ///
    /// This is a convenience method that wraps the TemporalFFTAnalysis oscillator bank synthesis
    pub fn synthesize_with_oscillator_bank(
        &self,
        analysis: &TemporalFFTAnalysis,
        hop_size: usize,
        stretch_start: f64,
        shift_start: Option<f64>,
        dispersion_start: Option<f64>,
        stretch_end: Option<f64>,
        shift_end: Option<f64>,
        dispersion_end: Option<f64>,
        output_frames: Option<usize>,
    ) -> Result<Vec<Vec<STFTFrame>>> {
        // Default output frames to original frame count if not specified
        let output_frames = output_frames.unwrap_or(if !analysis.bin_ffts.is_empty() {
            analysis
                .config
                .effective_length(analysis.bin_ffts[0].original_frames)
        } else {
            0
        });

        let mut result = synthesize_with_oscillator_bank_impl(
            analysis,
            stretch_start,
            shift_start,
            dispersion_start,
            stretch_end,
            shift_end,
            dispersion_end,
            output_frames,
            hop_size,
        );

        log::info!("checking hermitian symmetry");

        if let Ok(channels) = &mut result {
            log::info!("channels: {}", channels.len());
            for channel in channels.iter_mut() {
                log::info!("frames: {}", channel.len());
                for frame in channel.iter_mut() {
                    Self::enforce_hermitian_symmetry(&mut frame.spectrum);
                }
            }
        }

        log::info!("synthesize_with_oscillator_bank finished");

        result
    }
}

/// Synthesize STFT frames using oscillator bank with advanced temporal frequency manipulation
///
/// This method uses an oscillator bank approach instead of IFFT, allowing real-time frequency
/// scaling, shifting, and dispersion during synthesis. Each temporal FFT bin drives a complex
/// oscillator, and the parameters can vary linearly across the output frames.
fn synthesize_with_oscillator_bank_impl(
    analysis: &TemporalFFTAnalysis,
    stretch_start: f64,
    shift_start: Option<f64>,
    dispersion_start: Option<f64>,
    stretch_end: Option<f64>,
    shift_end: Option<f64>,
    dispersion_end: Option<f64>,
    output_frames: usize,
    hop_size: usize,
) -> Result<Vec<Vec<STFTFrame>>> {
    // Set defaults for optional parameters
    let shift_start = shift_start.unwrap_or(0.0);
    let dispersion_start = dispersion_start.unwrap_or(1.0);
    let stretch_end = stretch_end.unwrap_or(stretch_start);
    let shift_end = shift_end.unwrap_or(shift_start);
    let dispersion_end = dispersion_end.unwrap_or(dispersion_start);

    // Validate parameters
    if stretch_start <= 0.0 || stretch_end <= 0.0 {
        return Err("Stretch factors must be positive".to_string());
    }
    if shift_start < -1.0 || shift_start > 1.0 || shift_end < -1.0 || shift_end > 1.0 {
        return Err("Shift parameters must be between -1.0 and 1.0".to_string());
    }
    if output_frames == 0 {
        return Err("Output frames must be greater than 0".to_string());
    }

    log::info!(
        "Oscillator bank synthesis: {} channels, {} frequency bins, {} output frames",
        analysis.num_channels,
        analysis.num_frequency_bins,
        output_frames
    );
    log::info!(
        "Parameters: stretch {:.3}→{:.3}, shift {:.3}→{:.3}, dispersion {:.3}→{:.3}",
        stretch_start,
        stretch_end,
        shift_start,
        shift_end,
        dispersion_start,
        dispersion_end
    );

    let mut all_channel_frames = Vec::with_capacity(analysis.num_channels);

    // Process each channel
    for ch_idx in 0..analysis.num_channels {
        log::info!(
            "Processing oscillator bank synthesis for channel {}",
            ch_idx
        );

        let mut channel_frames = Vec::with_capacity(output_frames);

        // Generate each output frame
        for frame_idx in 0..output_frames {
            let frame_progress = if output_frames > 1 {
                frame_idx as f64 / (output_frames - 1) as f64
            } else {
                0.0
            };

            log::debug!(
                "Processing oscillator bank synthesis for channel {}, frame {}",
                ch_idx,
                frame_idx
            );

            // Interpolate parameters linearly across frames
            let current_stretch = lerp(stretch_start, stretch_end, frame_progress);
            let current_shift = lerp(shift_start, shift_end, frame_progress);
            let current_dispersion = lerp(dispersion_start, dispersion_end, frame_progress);

            // Initialize spectrum for this frame
            let mut frame_spectrum = vec![Complex64::new(0.0, 0.0); analysis.num_frequency_bins];

            // Process each frequency bin in the STFT frame
            for freq_bin_idx in 0..analysis.num_frequency_bins {
                // Get temporal spectrum for this frequency bin
                if let Some(temporal_spectrum) =
                    analysis.get_bin_temporal_spectrum(ch_idx, freq_bin_idx)
                {
                    // Calculate frequency-dependent dispersion
                    let freq_normalized =
                        freq_bin_idx as f64 / (analysis.num_frequency_bins - 1) as f64;
                    let dispersion_factor =
                        current_dispersion + (freq_normalized * (current_dispersion - 1.0));

                    // Synthesize this frequency bin using oscillator bank
                    frame_spectrum[freq_bin_idx] = synthesize_bin_with_oscillators(
                        temporal_spectrum,
                        frame_idx,
                        current_stretch * dispersion_factor,
                        current_shift,
                    );
                }
            }

            // Create STFT frame
            channel_frames.push(STFTFrame {
                spectrum: frame_spectrum,
                frame_index: frame_idx,
                time_position: frame_idx * hop_size,
            });
        }

        all_channel_frames.push(channel_frames);
    }

    log::info!("Oscillator bank synthesis complete");
    Ok(all_channel_frames)
}

/// Synthesize a single frequency bin using complex oscillator bank
///
/// Each temporal FFT bin drives a complex oscillator. The method handles both
/// positive and negative temporal frequencies correctly.
fn synthesize_bin_with_oscillators(
    temporal_spectrum: &[Complex64],
    frame_index: usize,
    stretch_factor: f64,
    shift_factor: f64,
) -> Complex64 {
    let fft_size = temporal_spectrum.len();
    let nyquist_temporal_freq = 0.5; // Normalized temporal Nyquist
    let mut real_sum = 0.0;
    let mut imag_sum = 0.0;

    // Process each temporal frequency bin
    for temp_bin_idx in 0..fft_size {
        let magnitude = temporal_spectrum[temp_bin_idx].norm();
        let phase_offset = temporal_spectrum[temp_bin_idx].arg();

        if magnitude < 1e-12 {
            continue; // Skip near-zero components
        }

        // Convert array index to logical temporal frequency
        let logical_temp_freq = array_index_to_logical_temp_freq(temp_bin_idx, fft_size);
        let normalized_temp_freq = logical_temp_freq as f64 / (fft_size as f64 / 2.0);

        // Apply transformations
        let shifted_freq = normalized_temp_freq + shift_factor;
        let scaled_freq = shifted_freq * stretch_factor;

        // Check bounds: temporal frequency must be within [-1, 1] normalized range
        if scaled_freq.abs() > 1.0 {
            continue; // Frequency out of bounds, skip this component
        }

        // Convert back to actual temporal frequency (cycles per frame)
        let actual_temp_freq = scaled_freq * nyquist_temporal_freq;

        // Generate complex oscillator for this temporal frequency
        // phase = 2π * freq * time + phase_offset
        let phase =
            2.0 * std::f64::consts::PI * actual_temp_freq * frame_index as f64 + phase_offset;

        // Complex exponential: magnitude * e^(j*phase)
        let oscillator_real = magnitude * phase.cos();
        let oscillator_imag = magnitude * phase.sin();

        real_sum += oscillator_real;
        imag_sum += oscillator_imag;
    }

    // Normalize by FFT size (similar to IFFT normalization)
    let normalization = 1.0 / fft_size as f64;
    Complex64::new(real_sum * normalization, imag_sum * normalization)
}

/// Convert array index to logical temporal frequency index
///
/// For temporal FFT:
/// - Index 0: DC (0 Hz)
/// - Indices 1 to N/2: positive frequencies
/// - Indices N/2+1 to N-1: negative frequencies
fn array_index_to_logical_temp_freq(index: usize, fft_size: usize) -> i32 {
    if index <= fft_size / 2 {
        index as i32
    } else {
        (index as i32) - (fft_size as i32)
    }
}

/// Linear interpolation helper
fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + t * (end - start)
}
