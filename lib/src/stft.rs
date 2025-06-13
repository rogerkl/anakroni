//! Short-Time Fourier Transform (STFT) implementation
//!
//! Provides STFT analysis and synthesis functionality using overlapping windows.
//! Based on the Ceres analysis with similar parameter defaults.

use crate::window::{coherent_gain, generate_window, WindowType};
use crate::Result;
use num_complex::Complex64;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// STFT configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct STFTConfig {
    /// Window size (FFT size), must be power of 2
    pub window_size: usize,
    /// Overlap factor (hop_size = window_size / overlap_factor)
    pub overlap_factor: usize,
    /// Window function type
    pub window_type: WindowType,
}

impl Default for STFTConfig {
    fn default() -> Self {
        // Based on Ceres defaults
        Self {
            window_size: 1024,                // Default from Ceres
            overlap_factor: 8,                // Default from Ceres (87.5% overlap)
            window_type: WindowType::Hanning, // Recommended default
        }
    }
}

impl STFTConfig {
    /// Create a new STFT configuration with validation
    pub fn new(window_size: usize, overlap_factor: usize, window_type: WindowType) -> Result<Self> {
        // Validate window size (must be power of 2)
        if !window_size.is_power_of_two() || window_size < 2 || window_size > 65536 {
            return Err(format!(
                "Window size must be a power of 2 between 256 and 4096, got {}",
                window_size
            ));
        }

        // Validate overlap factor (based on Ceres: 1-32)
        if overlap_factor < 1 || overlap_factor > 32 {
            return Err(format!(
                "Overlap factor must be between 1 and 32, got {}",
                overlap_factor
            ));
        }

        Ok(Self {
            window_size,
            overlap_factor,
            window_type,
        })
    }

    /// Get the hop size (step size between windows)
    pub fn hop_size(&self) -> usize {
        self.window_size / self.overlap_factor
    }

    /// Get the overlap percentage
    pub fn overlap_percent(&self) -> f64 {
        (1.0 - 1.0 / self.overlap_factor as f64) * 100.0
    }

    /// Get the number of FFT bins (complex values)
    pub fn fft_bins(&self) -> usize {
        self.window_size / 2 + 1
    }
}

/// STFT frame containing frequency domain data
#[derive(Debug, Clone)]
pub struct STFTFrame {
    /// Complex frequency domain data
    pub spectrum: Vec<Complex64>,
    /// Frame index in the analysis
    pub frame_index: usize,
    /// Time position in samples
    pub time_position: usize,
}

/// STFT analyzer for forward transform
pub struct STFTAnalyzer {
    config: STFTConfig,
    window: Vec<f64>,
    fft_planner: Arc<dyn RealToComplex<f64>>,
    coherent_gain: f64,
}

impl STFTAnalyzer {
    /// Create a new STFT analyzer
    pub fn new(config: STFTConfig) -> Result<Self> {
        let window = generate_window(config.window_type, config.window_size);
        let coherent_gain = coherent_gain(&window);

        let mut planner = RealFftPlanner::<f64>::new();
        let fft_planner = planner.plan_fft_forward(config.window_size);

        Ok(Self {
            config,
            window,
            fft_planner,
            coherent_gain,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &STFTConfig {
        &self.config
    }

    /// Analyze audio data and return STFT frames
    pub fn analyze(&self, audio_data: &[f64]) -> Result<Vec<STFTFrame>> {
        let hop_size = self.config.hop_size();
        let window_size = self.config.window_size;

        if audio_data.len() < window_size {
            return Err("Audio data is shorter than window size".to_string());
        }

        let num_frames = (audio_data.len() - window_size) / hop_size + 1;
        let mut frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start_pos = frame_idx * hop_size;
            let end_pos = start_pos + window_size;

            if end_pos > audio_data.len() {
                break;
            }

            // Apply window to the audio segment
            let mut windowed_data = vec![0.0; window_size];
            for (i, &sample) in audio_data[start_pos..end_pos].iter().enumerate() {
                windowed_data[i] = sample * self.window[i];
            }

            // Perform FFT
            let mut spectrum = vec![Complex64::new(0.0, 0.0); self.config.fft_bins()];
            self.fft_planner
                .process(&mut windowed_data, &mut spectrum)
                .map_err(|e| format!("FFT error: {}", e))?;

            frames.push(STFTFrame {
                spectrum,
                frame_index: frame_idx,
                time_position: start_pos,
            });
        }

        Ok(frames)
    }

    /// Get the expected number of frames for given audio length
    pub fn num_frames(&self, audio_length: usize) -> usize {
        if audio_length < self.config.window_size {
            0
        } else {
            (audio_length - self.config.window_size) / self.config.hop_size() + 1
        }
    }
}

/// STFT synthesizer for inverse transform
pub struct STFTSynthesizer {
    config: STFTConfig,
    window: Vec<f64>,
    ifft_planner: Arc<dyn ComplexToReal<f64>>,
    coherent_gain: f64,
}

impl STFTSynthesizer {
    /// Create a new STFT synthesizer
    pub fn new(config: STFTConfig) -> Result<Self> {
        let window = generate_window(config.window_type, config.window_size);
        let coherent_gain = coherent_gain(&window);

        let mut planner = RealFftPlanner::<f64>::new();
        let ifft_planner = planner.plan_fft_inverse(config.window_size);

        Ok(Self {
            config,
            window,
            ifft_planner,
            coherent_gain,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &STFTConfig {
        &self.config
    }

    /// Synthesize audio data from STFT frames
    pub fn synthesize(&self, frames: &[STFTFrame]) -> Result<Vec<f64>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let hop_size = self.config.hop_size();
        let window_size = self.config.window_size;

        // Calculate output length
        let last_frame = frames.last().unwrap();
        let output_length = last_frame.time_position + window_size;
        let mut output = vec![0.0; output_length];
        let mut window_sum = vec![0.0; output_length];

        // Process each frame
        for frame in frames {
            // Prepare spectrum for IFFT (make a copy since IFFT modifies input)
            let mut spectrum = frame.spectrum.clone();

            // Perform IFFT
            let mut time_data = vec![0.0; window_size];
            self.ifft_planner
                .process(&mut spectrum, &mut time_data)
                .map_err(|e| format!("IFFT error: {}", e))?;

            // Apply window and overlap-add
            let start_pos = frame.time_position;
            for (i, &sample) in time_data.iter().enumerate() {
                let pos = start_pos + i;
                if pos < output.len() {
                    let windowed_sample = sample * self.window[i];
                    output[pos] += windowed_sample;
                    window_sum[pos] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window overlap
        for (i, &sum) in window_sum.iter().enumerate() {
            if sum > 1e-10 {
                output[i] /= sum;
            }
        }

        // Scale by coherent gain to maintain amplitude
        let scale = 1.0 / (self.coherent_gain * window_size as f64);
        for sample in output.iter_mut() {
            *sample *= scale;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_stft_config() {
        let config = STFTConfig::default();
        assert_eq!(config.window_size, 1024);
        assert_eq!(config.overlap_factor, 8);
        assert_eq!(config.hop_size(), 128);
        assert_eq!(config.fft_bins(), 513);
        assert!((config.overlap_percent() - 87.5).abs() < 1e-10);
    }

    #[test]
    fn test_stft_config_validation() {
        // Valid configuration
        assert!(STFTConfig::new(512, 4, WindowType::Hanning).is_ok());

        // Invalid window size (not power of 2)
        assert!(STFTConfig::new(500, 4, WindowType::Hanning).is_err());

        // Invalid overlap factor
        assert!(STFTConfig::new(512, 0, WindowType::Hanning).is_err());
        assert!(STFTConfig::new(512, 33, WindowType::Hanning).is_err());
    }

    #[test]
    fn test_stft_analysis_synthesis() {
        let config = STFTConfig::default();
        let analyzer = STFTAnalyzer::new(config.clone()).unwrap();
        let synthesizer = STFTSynthesizer::new(config).unwrap();

        // Generate test signal (sine wave)
        let sample_rate = 44100.0;
        let duration = 1.0;
        let frequency = 440.0;
        let samples = (sample_rate * duration) as usize;

        let mut test_signal = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f64 / sample_rate;
            test_signal.push((2.0 * PI * frequency * t).sin());
        }

        // Analyze
        let frames = analyzer.analyze(&test_signal).unwrap();
        assert!(!frames.is_empty());

        // Check that we got the expected number of frames
        let expected_frames = analyzer.num_frames(test_signal.len());
        assert_eq!(frames.len(), expected_frames);

        // Synthesize
        let reconstructed = synthesizer.synthesize(&frames).unwrap();

        // Check that lengths are reasonable
        assert!(reconstructed.len() >= test_signal.len() - config.window_size);

        // Check that the signal is reasonably reconstructed
        // (Perfect reconstruction is not expected due to windowing at edges)
        let min_len = std::cmp::min(test_signal.len(), reconstructed.len());
        let mut error_sum = 0.0;
        let start_compare = config.window_size; // Skip edge effects
        let end_compare = min_len - config.window_size;

        for i in start_compare..end_compare {
            let error = (test_signal[i] - reconstructed[i]).abs();
            error_sum += error;
        }

        let mean_error = error_sum / (end_compare - start_compare) as f64;
        println!("Mean reconstruction error: {:.6}", mean_error);

        // Error should be small (typically < 1e-10 for perfect reconstruction conditions)
        assert!(
            mean_error < 1e-3,
            "Reconstruction error too large: {}",
            mean_error
        );
    }
}
