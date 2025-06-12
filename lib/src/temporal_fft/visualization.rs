//! Temporal FFT visualization functionality

use super::core::TemporalFFTAnalysis;
use crate::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

/// Options for temporal FFT visualization
pub struct TemporalFFTImageOptions {
    /// Whether to use dB scale for magnitudes
    pub use_db_scale: bool,
    /// Dynamic range in dB (if using dB scale)
    pub dynamic_range_db: f64,
    /// Reference value for dB conversion
    pub db_reference: f64,
    /// Color map to use (reusing from spectrogram module)
    pub colormap: crate::spectrogram::image::ColorMap,
}

impl Default for TemporalFFTImageOptions {
    fn default() -> Self {
        Self {
            use_db_scale: false,
            dynamic_range_db: 80.0,
            db_reference: 1.0,
            colormap: crate::spectrogram::image::ColorMap::Viridis,
        }
    }
}

/// Generate a temporal FFT visualization image
/// Y-axis: frequency bins (0 at top)
/// X-axis: temporal FFT bins
pub fn generate_temporal_fft_image(
    analysis: &TemporalFFTAnalysis,
    channel: usize,
    options: &TemporalFFTImageOptions,
) -> Result<RgbImage> {
    if channel >= analysis.num_channels {
        return Err(format!(
            "Channel {} out of range (0-{})",
            channel,
            analysis.num_channels - 1
        ));
    }

    // Get the 2D magnitude data
    let magnitude_data = analysis.to_temporal_spectrogram(channel, options.use_db_scale);

    if magnitude_data.is_empty() {
        return Err("No temporal FFT data available".to_string());
    }

    let height = magnitude_data.len() as u32; // Number of frequency bins
    let width = magnitude_data[0].len() as u32; // Number of temporal FFT bins

    log::info!(
        "Generating temporal FFT image: {}x{} pixels ({}x{} bins)",
        width,
        height,
        width,
        height
    );

    // Find min/max values for normalization
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for row in &magnitude_data {
        for &val in row {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    // Apply dynamic range if using dB scale
    if options.use_db_scale {
        min_val = max_val - options.dynamic_range_db;
    }

    let range = max_val - min_val;
    log::info!("min: {}, max: {}, range: {}", min_val, max_val, range);
    if range <= 0.0 {
        return Err("No dynamic range in temporal FFT data".to_string());
    }

    // Create the image
    let mut img = ImageBuffer::new(width, height);

    // Fill the image
    for (y, row) in magnitude_data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let normalized = ((value - min_val) / range).clamp(0.0, 1.0);
            let color = value_to_color(normalized, options.colormap);
            img.put_pixel(x as u32, y as u32, color);
        }
    }

    Ok(img)
}

/// Save temporal FFT visualization to file
pub fn save_temporal_fft_image<P: AsRef<Path>>(
    analysis: &TemporalFFTAnalysis,
    channel: usize,
    path: P,
    options: &TemporalFFTImageOptions,
) -> Result<()> {
    let img = generate_temporal_fft_image(analysis, channel, options)?;
    img.save(path)
        .map_err(|e| format!("Failed to save temporal FFT image: {}", e))
}

/// Convert a value (0.0 to 1.0) to RGB color using the specified colormap
fn value_to_color(value: f64, colormap: crate::spectrogram::image::ColorMap) -> Rgb<u8> {
    let v = value.clamp(0.0, 1.0);

    match colormap {
        crate::spectrogram::image::ColorMap::Viridis => {
            let r = (v * v * v * 0.3 + v * 0.1) * 255.0;
            let g = (v.sqrt() * 0.8 + v * 0.2) * 255.0;
            let b = (v.powf(0.3) * 0.9 + v * 0.1) * 255.0;
            Rgb([r as u8, g as u8, b as u8])
        }
        crate::spectrogram::image::ColorMap::Plasma => {
            let r = ((v * 2.0).min(1.0) * 0.9 + 0.1) * 255.0;
            let g = (v * v * 0.5) * 255.0;
            let b = ((1.0 - v).powf(2.0) * 0.7 + v * 0.3) * 255.0;
            Rgb([r as u8, g as u8, b as u8])
        }
        crate::spectrogram::image::ColorMap::Inferno => {
            let r = (v * v * v * 0.5 + v * v * 0.5) * 255.0;
            let g = (v * v * 0.8) * 255.0;
            let b = (v.powf(4.0)) * 255.0;
            Rgb([r as u8, g as u8, b as u8])
        }
        crate::spectrogram::image::ColorMap::Magma => {
            let r = (v * v * v * 0.4 + v * v * 0.6) * 255.0;
            let g = (v * v * 0.5) * 255.0;
            let b = (v.powf(3.0) * 0.8 + v * 0.2) * 255.0;
            Rgb([r as u8, g as u8, b as u8])
        }
        crate::spectrogram::image::ColorMap::Turbo => {
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
        crate::spectrogram::image::ColorMap::Grayscale => {
            let gray = (v * 255.0) as u8;
            Rgb([gray, gray, gray])
        }
        crate::spectrogram::image::ColorMap::Jet => {
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
