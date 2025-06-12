//! Spectrogram generation and visualization
//!
//! Provides functionality to generate spectrograms from STFT frames
//! for visualization and debugging purposes.

use crate::stft::STFTFrame;
use crate::Result;
use num_complex::Complex64;

/// Frequency scale type for spectrogram display
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrequencyScale {
    /// Linear frequency scale
    Linear,
    /// Logarithmic frequency scale
    Logarithmic,
}

/// Spectrogram data structure
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// Magnitude data (time x frequency)
    pub magnitudes: Vec<Vec<f64>>,
    /// Phase data (time x frequency)
    pub phases: Vec<Vec<f64>>,
    /// Number of time frames
    pub num_frames: usize,
    /// Number of frequency bins
    pub num_bins: usize,
    /// Sample rate (if known)
    pub sample_rate: Option<u32>,
    /// Hop size (if known)
    pub hop_size: Option<usize>,
}

impl Spectrogram {
    /// Create a spectrogram from STFT frames
    pub fn from_stft_frames(
        frames: &[STFTFrame],
        sample_rate: Option<u32>,
        hop_size: Option<usize>,
    ) -> Result<Self> {
        if frames.is_empty() {
            return Err("No STFT frames provided".to_string());
        }

        let num_frames = frames.len();
        let num_bins = frames[0].spectrum.len();

        // Verify all frames have the same number of bins
        for (i, frame) in frames.iter().enumerate() {
            if frame.spectrum.len() != num_bins {
                return Err(format!(
                    "Frame {} has {} bins, expected {}",
                    i,
                    frame.spectrum.len(),
                    num_bins
                ));
            }
        }

        let mut magnitudes = Vec::with_capacity(num_frames);
        let mut phases = Vec::with_capacity(num_frames);

        for frame in frames {
            let mut frame_mags = Vec::with_capacity(num_bins);
            let mut frame_phases = Vec::with_capacity(num_bins);

            for &complex_val in &frame.spectrum {
                frame_mags.push(complex_val.norm());
                frame_phases.push(complex_val.arg());
            }

            magnitudes.push(frame_mags);
            phases.push(frame_phases);
        }

        Ok(Self {
            magnitudes,
            phases,
            num_frames,
            num_bins,
            sample_rate,
            hop_size,
        })
    }

    /// Convert magnitude to decibels
    pub fn to_db(&self, reference: f64) -> Vec<Vec<f64>> {
        let min_db = -100.0; // Floor value for very small magnitudes

        self.magnitudes
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|&mag| {
                        if mag > 0.0 {
                            (20.0 * (mag / reference).log10()).max(min_db)
                        } else {
                            min_db
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Get time axis values in seconds
    pub fn time_axis(&self) -> Vec<f64> {
        match (self.sample_rate, self.hop_size) {
            (Some(sr), Some(hop)) => (0..self.num_frames)
                .map(|i| (i * hop) as f64 / sr as f64)
                .collect(),
            _ => (0..self.num_frames).map(|i| i as f64).collect(),
        }
    }

    /// Get frequency axis values in Hz
    pub fn frequency_axis(&self) -> Vec<f64> {
        match self.sample_rate {
            Some(sr) => {
                let nyquist = sr as f64 / 2.0;
                (0..self.num_bins)
                    .map(|i| i as f64 * nyquist / (self.num_bins - 1) as f64)
                    .collect()
            }
            None => (0..self.num_bins).map(|i| i as f64).collect(),
        }
    }

    /// Get a specific frequency band (sum of bins in range)
    pub fn get_frequency_band(&self, start_bin: usize, end_bin: usize) -> Vec<f64> {
        self.magnitudes
            .iter()
            .map(|frame| {
                frame[start_bin.min(self.num_bins - 1)..end_bin.min(self.num_bins)]
                    .iter()
                    .sum()
            })
            .collect()
    }

    /// Get temporal evolution of a specific frequency bin
    pub fn get_bin_evolution(&self, bin_index: usize) -> Vec<f64> {
        if bin_index >= self.num_bins {
            return vec![0.0; self.num_frames];
        }

        self.magnitudes
            .iter()
            .map(|frame| frame[bin_index])
            .collect()
    }

    /// Apply smoothing to the spectrogram
    pub fn smooth(&mut self, time_window: usize, freq_window: usize) {
        if time_window <= 1 && freq_window <= 1 {
            return;
        }

        let mut smoothed = vec![vec![0.0; self.num_bins]; self.num_frames];

        for t in 0..self.num_frames {
            for f in 0..self.num_bins {
                let mut sum = 0.0;
                let mut count = 0;

                // Average over time window
                let t_start = t.saturating_sub(time_window / 2);
                let t_end = (t + time_window / 2 + 1).min(self.num_frames);

                // Average over frequency window
                let f_start = f.saturating_sub(freq_window / 2);
                let f_end = (f + freq_window / 2 + 1).min(self.num_bins);

                for ti in t_start..t_end {
                    for fi in f_start..f_end {
                        sum += self.magnitudes[ti][fi];
                        count += 1;
                    }
                }

                if count > 0 {
                    smoothed[t][f] = sum / count as f64;
                }
            }
        }

        self.magnitudes = smoothed;
    }
}

/// Generate an image from spectrogram data
#[cfg(not(target_arch = "wasm32"))]
pub mod image {
    use super::*;
    use ::image::{ImageBuffer, Rgb, RgbImage};
    use std::path::Path;

    /// Color map types for spectrogram visualization
    #[derive(Debug, Clone, Copy)]
    pub enum ColorMap {
        Viridis,
        Plasma,
        Inferno,
        Magma,
        Turbo,
        Grayscale,
        Jet,
    }

    /// Convert a value (0.0 to 1.0) to RGB color using the specified colormap
    fn value_to_color(value: f64, colormap: ColorMap) -> Rgb<u8> {
        let v = value.clamp(0.0, 1.0);

        match colormap {
            ColorMap::Viridis => {
                // Viridis colormap approximation
                let r = (v * v * v * 0.3 + v * 0.1) * 255.0;
                let g = (v.sqrt() * 0.8 + v * 0.2) * 255.0;
                let b = (v.powf(0.3) * 0.9 + v * 0.1) * 255.0;
                Rgb([r as u8, g as u8, b as u8])
            }
            ColorMap::Plasma => {
                // Plasma colormap approximation
                let r = ((v * 2.0).min(1.0) * 0.9 + 0.1) * 255.0;
                let g = (v * v * 0.5) * 255.0;
                let b = ((1.0 - v).powf(2.0) * 0.7 + v * 0.3) * 255.0;
                Rgb([r as u8, g as u8, b as u8])
            }
            ColorMap::Inferno => {
                // Inferno colormap approximation
                let r = (v * v * v * 0.5 + v * v * 0.5) * 255.0;
                let g = (v * v * 0.8) * 255.0;
                let b = (v.powf(4.0)) * 255.0;
                Rgb([r as u8, g as u8, b as u8])
            }
            ColorMap::Magma => {
                // Magma colormap approximation
                let r = (v * v * v * 0.4 + v * v * 0.6) * 255.0;
                let g = (v * v * 0.5) * 255.0;
                let b = (v.powf(3.0) * 0.8 + v * 0.2) * 255.0;
                Rgb([r as u8, g as u8, b as u8])
            }
            ColorMap::Turbo => {
                // Turbo colormap approximation (Google's improved Jet)
                let r = if v < 0.5 {
                    (v * 2.0 * 0.3 + 0.2) * 255.0
                } else {
                    ((v - 0.5) * 2.0 * 0.7 + 0.3) * 255.0
                };
                let g = ((-4.0 * v * v + 4.0 * v) * 0.9 + 0.1) * 255.0;
                let b = if v < 0.5 {
                    ((0.5 - v) * 2.0 * 0.7 + 0.3) * 255.0
                } else {
                    (0.3 - (v - 0.5) * 2.0 * 0.3) * 255.0
                };
                Rgb([r as u8, g as u8, b as u8])
            }
            ColorMap::Grayscale => {
                let gray = (v * 255.0) as u8;
                Rgb([gray, gray, gray])
            }
            ColorMap::Jet => {
                // Classic Jet colormap
                let r = if v < 0.375 {
                    0.0
                } else if v < 0.625 {
                    (v - 0.375) * 4.0
                } else if v < 0.875 {
                    1.0
                } else {
                    1.0 - (v - 0.875) * 4.0
                };

                let g = if v < 0.125 {
                    0.0
                } else if v < 0.375 {
                    (v - 0.125) * 4.0
                } else if v < 0.625 {
                    1.0
                } else if v < 0.875 {
                    1.0 - (v - 0.625) * 4.0
                } else {
                    0.0
                };

                let b = if v < 0.125 {
                    0.5 + v * 4.0
                } else if v < 0.375 {
                    1.0
                } else if v < 0.625 {
                    1.0 - (v - 0.375) * 4.0
                } else {
                    0.0
                };

                Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
            }
        }
    }

    /// Options for spectrogram image generation
    pub struct SpectrogramImageOptions {
        /// Width of the output image in pixels
        pub width: u32,
        /// Height of the output image in pixels
        pub height: u32,
        /// Color map to use
        pub colormap: ColorMap,
        /// Dynamic range in dB (for dB scale)
        pub dynamic_range_db: f64,
        /// Reference value for dB conversion
        pub db_reference: f64,
        /// Whether to use dB scale
        pub use_db_scale: bool,
        /// Frequency range to display (None for full range)
        pub freq_range: Option<(f64, f64)>,
        /// Frequency scale type
        pub frequency_scale: FrequencyScale,
        /// Minimum frequency for logarithmic scale (Hz)
        pub log_freq_min: f64,
    }

    impl Default for SpectrogramImageOptions {
        fn default() -> Self {
            Self {
                width: 800,
                height: 600,
                colormap: ColorMap::Viridis,
                dynamic_range_db: 80.0,
                db_reference: 1.0,
                use_db_scale: true,
                freq_range: None,
                frequency_scale: FrequencyScale::Logarithmic,
                log_freq_min: 20.0, // 20 Hz default minimum for log scale
            }
        }
    }

    /// Convert linear frequency to logarithmic scale position
    fn freq_to_log_position(freq: f64, min_freq: f64, max_freq: f64) -> f64 {
        if freq <= min_freq {
            0.0
        } else if freq >= max_freq {
            1.0
        } else {
            (freq.ln() - min_freq.ln()) / (max_freq.ln() - min_freq.ln())
        }
    }

    /// Convert logarithmic position back to frequency
    fn log_position_to_freq(position: f64, min_freq: f64, max_freq: f64) -> f64 {
        min_freq * (max_freq / min_freq).powf(position)
    }

    /// Generate a spectrogram image
    pub fn generate_spectrogram_image(
        spectrogram: &Spectrogram,
        options: &SpectrogramImageOptions,
    ) -> RgbImage {
        let mut img = ImageBuffer::new(options.width, options.height);

        // Get magnitude data (in dB if requested)
        let magnitudes = if options.use_db_scale {
            spectrogram.to_db(options.db_reference)
        } else {
            spectrogram.magnitudes.clone()
        };

        // Find min and max values for normalization
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        // Get frequency axis
        let freq_axis = spectrogram.frequency_axis();

        // Determine frequency range
        let (min_freq, max_freq) = if let Some((fmin, fmax)) = options.freq_range {
            (fmin, fmax)
        } else if options.frequency_scale == FrequencyScale::Logarithmic {
            // For log scale, use specified minimum and actual maximum
            let actual_max = freq_axis.last().copied().unwrap_or(1000.0);
            (options.log_freq_min.max(1.0), actual_max)
        } else {
            // For linear scale, use full range
            (0.0, freq_axis.last().copied().unwrap_or(1000.0))
        };

        // Determine bin range for linear scale
        let (start_bin, end_bin) = if options.frequency_scale == FrequencyScale::Linear {
            if let Some((min_f, max_f)) = options.freq_range {
                let start = freq_axis.iter().position(|&f| f >= min_f).unwrap_or(0);
                let end = freq_axis
                    .iter()
                    .rposition(|&f| f <= max_f)
                    .unwrap_or(spectrogram.num_bins - 1)
                    + 1;
                (start, end.min(spectrogram.num_bins))
            } else {
                (0, spectrogram.num_bins)
            }
        } else {
            // For log scale, we'll use all bins but map them differently
            (0, spectrogram.num_bins)
        };

        // Find min/max values
        for frame in &magnitudes {
            for i in 0..spectrogram.num_bins {
                if options.frequency_scale == FrequencyScale::Linear {
                    // Only consider bins in range for linear scale
                    if i >= start_bin && i < end_bin {
                        let val = frame[i];
                        if val.is_finite() {
                            min_val = min_val.min(val);
                            max_val = max_val.max(val);
                        }
                    }
                } else {
                    // For log scale, consider bins that map to the frequency range
                    if i < freq_axis.len() && freq_axis[i] >= min_freq && freq_axis[i] <= max_freq {
                        let val = frame[i];
                        if val.is_finite() {
                            min_val = min_val.min(val);
                            max_val = max_val.max(val);
                        }
                    }
                }
            }
        }

        // Apply dynamic range limiting for dB scale
        if options.use_db_scale {
            min_val = max_val - options.dynamic_range_db;
        }

        let range = max_val - min_val;
        if range <= 0.0 {
            return img; // Return blank image if no range
        }

        // Fill the image
        match options.frequency_scale {
            FrequencyScale::Linear => {
                // Linear frequency scale (original implementation)
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    // Map pixel coordinates to spectrogram indices
                    let time_idx = (x as f64 * (spectrogram.num_frames - 1) as f64
                        / (options.width - 1) as f64) as usize;

                    // Flip Y axis so low frequencies are at bottom
                    let y_flipped = options.height - 1 - y;
                    let freq_idx = start_bin
                        + (y_flipped as f64 * (end_bin - start_bin - 1) as f64
                            / (options.height - 1) as f64) as usize;

                    if time_idx < spectrogram.num_frames && freq_idx < spectrogram.num_bins {
                        let value = magnitudes[time_idx][freq_idx];
                        let normalized = ((value - min_val) / range).clamp(0.0, 1.0);
                        *pixel = value_to_color(normalized, options.colormap);
                    }
                }
            }
            FrequencyScale::Logarithmic => {
                // Logarithmic frequency scale
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    // Map pixel coordinates to time index
                    let time_idx = (x as f64 * (spectrogram.num_frames - 1) as f64
                        / (options.width - 1) as f64) as usize;

                    // Flip Y axis so low frequencies are at bottom
                    let y_flipped = options.height - 1 - y;
                    let y_normalized = y_flipped as f64 / (options.height - 1) as f64;

                    // Convert Y position to frequency using logarithmic mapping
                    let target_freq = log_position_to_freq(y_normalized, min_freq, max_freq);

                    // Find the closest frequency bin
                    let mut best_bin = 0;
                    let mut best_diff = f64::INFINITY;
                    for (bin_idx, &bin_freq) in freq_axis.iter().enumerate() {
                        let diff = (bin_freq - target_freq).abs();
                        if diff < best_diff {
                            best_diff = diff;
                            best_bin = bin_idx;
                        }
                    }

                    // Interpolate between adjacent bins for smoother visualization
                    let mut value = magnitudes[time_idx][best_bin];

                    // Optional: interpolate with neighboring bins
                    if best_bin > 0 && best_bin < spectrogram.num_bins - 1 {
                        let prev_freq = freq_axis[best_bin - 1];
                        let curr_freq = freq_axis[best_bin];
                        let next_freq = freq_axis[best_bin + 1];

                        if target_freq < curr_freq && prev_freq < target_freq {
                            // Interpolate with previous bin
                            let t = (target_freq - prev_freq) / (curr_freq - prev_freq);
                            value = magnitudes[time_idx][best_bin - 1] * (1.0 - t)
                                + magnitudes[time_idx][best_bin] * t;
                        } else if target_freq > curr_freq && next_freq > target_freq {
                            // Interpolate with next bin
                            let t = (target_freq - curr_freq) / (next_freq - curr_freq);
                            value = magnitudes[time_idx][best_bin] * (1.0 - t)
                                + magnitudes[time_idx][best_bin + 1] * t;
                        }
                    }

                    if time_idx < spectrogram.num_frames {
                        let normalized = ((value - min_val) / range).clamp(0.0, 1.0);
                        *pixel = value_to_color(normalized, options.colormap);
                    }
                }
            }
        }

        img
    }

    /// Save a spectrogram to an image file
    pub fn save_spectrogram<P: AsRef<Path>>(
        spectrogram: &Spectrogram,
        path: P,
        options: &SpectrogramImageOptions,
    ) -> Result<()> {
        let img = generate_spectrogram_image(spectrogram, options);
        img.save(path)
            .map_err(|e| format!("Failed to save spectrogram image: {}", e))
    }
}

/// Generate data suitable for web visualization
#[cfg(target_arch = "wasm32")]
pub mod web {
    use super::*;

    /// Spectrogram data formatted for web visualization
    #[derive(Debug, Clone)]
    pub struct WebSpectrogramData {
        /// Flattened magnitude data (row-major order)
        pub magnitudes: Vec<f32>,
        /// Number of time frames
        pub num_frames: usize,
        /// Number of frequency bins
        pub num_bins: usize,
        /// Time axis values
        pub time_axis: Vec<f32>,
        /// Frequency axis values
        pub freq_axis: Vec<f32>,
        /// Min and max magnitude values
        pub magnitude_range: (f32, f32),
    }

    /// Convert spectrogram to web-friendly format
    pub fn to_web_format(
        spectrogram: &Spectrogram,
        use_db: bool,
        db_reference: f64,
    ) -> WebSpectrogramData {
        let magnitudes = if use_db {
            spectrogram.to_db(db_reference)
        } else {
            spectrogram.magnitudes.clone()
        };

        // Flatten and convert to f32
        let mut flat_mags = Vec::with_capacity(spectrogram.num_frames * spectrogram.num_bins);
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for frame in &magnitudes {
            for &val in frame {
                let val_f32 = val as f32;
                flat_mags.push(val_f32);
                if val_f32.is_finite() {
                    min_val = min_val.min(val_f32);
                    max_val = max_val.max(val_f32);
                }
            }
        }

        WebSpectrogramData {
            magnitudes: flat_mags,
            num_frames: spectrogram.num_frames,
            num_bins: spectrogram.num_bins,
            time_axis: spectrogram
                .time_axis()
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            freq_axis: spectrogram
                .frequency_axis()
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            magnitude_range: (min_val, max_val),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::STFTFrame;
    use num_complex::Complex64;

    fn create_test_frames() -> Vec<STFTFrame> {
        let num_frames = 10;
        let num_bins = 5;

        (0..num_frames)
            .map(|t| STFTFrame {
                spectrum: (0..num_bins)
                    .map(|f| {
                        let mag = ((t as f64 * f as f64) / 10.0).sin().abs();
                        Complex64::from_polar(mag, 0.0)
                    })
                    .collect(),
                frame_index: t,
                time_position: t * 128,
            })
            .collect()
    }

    #[test]
    fn test_spectrogram_creation() {
        let frames = create_test_frames();
        let spec = Spectrogram::from_stft_frames(&frames, Some(44100), Some(128)).unwrap();

        assert_eq!(spec.num_frames, 10);
        assert_eq!(spec.num_bins, 5);
        assert_eq!(spec.magnitudes.len(), 10);
        assert_eq!(spec.magnitudes[0].len(), 5);
    }

    #[test]
    fn test_db_conversion() {
        let frames = create_test_frames();
        let spec = Spectrogram::from_stft_frames(&frames, None, None).unwrap();
        let db_values = spec.to_db(1.0);

        assert_eq!(db_values.len(), spec.num_frames);
        for frame in &db_values {
            for &val in frame {
                assert!(val <= 0.0 || val == -100.0); // dB values should be negative or floor
            }
        }
    }

    #[test]
    fn test_axes() {
        let frames = create_test_frames();
        let spec = Spectrogram::from_stft_frames(&frames, Some(44100), Some(128)).unwrap();

        let time_axis = spec.time_axis();
        let freq_axis = spec.frequency_axis();

        assert_eq!(time_axis.len(), spec.num_frames);
        assert_eq!(freq_axis.len(), spec.num_bins);
        assert!(time_axis[0] < time_axis[1]); // Time should increase
        assert!(freq_axis[0] < freq_axis[1]); // Frequency should increase
    }
}
