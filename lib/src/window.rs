//! Window functions for STFT analysis
//!
//! Implements various window functions used in spectral analysis.
//! Based on the Ceres analysis, we support Hanning, Hamming, Rectangular, and Bartlett windows.

use std::f64::consts::PI;
use std::fmt;

/// Window function types available for STFT analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hanning window (recommended default)
    Hanning,
    /// Hamming window
    Hamming,
    /// Rectangular window (no windowing)
    Rectangular,
    /// Bartlett (triangular) window
    Bartlett,
}

impl Default for WindowType {
    fn default() -> Self {
        WindowType::Hanning
    }
}

impl fmt::Display for WindowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl WindowType {
    /// Get all available window types
    pub fn all() -> &'static [WindowType] {
        &[
            WindowType::Hanning,
            WindowType::Hamming,
            WindowType::Rectangular,
            WindowType::Bartlett,
        ]
    }

    /// Get the name of the window type
    pub fn name(&self) -> &'static str {
        match self {
            WindowType::Hanning => "Hanning",
            WindowType::Hamming => "Hamming",
            WindowType::Rectangular => "Rectangular",
            WindowType::Bartlett => "Bartlett",
        }
    }

    /// Check if this window type is recommended
    pub fn is_recommended(&self) -> bool {
        matches!(self, WindowType::Hanning | WindowType::Bartlett)
    }
}

/// Generate a window function of the specified type and size
pub fn generate_window(window_type: WindowType, size: usize) -> Vec<f64> {
    let mut window = vec![0.0; size];

    match window_type {
        WindowType::Hanning => generate_hanning(&mut window),
        WindowType::Hamming => generate_hamming(&mut window),
        WindowType::Rectangular => generate_rectangular(&mut window),
        WindowType::Bartlett => generate_bartlett(&mut window),
    }

    window
}

/// Generate Hanning window (recommended)
fn generate_hanning(window: &mut [f64]) {
    let n = window.len();
    for (i, w) in window.iter_mut().enumerate() {
        *w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
    }
}

/// Generate Hamming window
fn generate_hamming(window: &mut [f64]) {
    let n = window.len();
    for (i, w) in window.iter_mut().enumerate() {
        *w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
    }
}

/// Generate Rectangular window (no windowing)
fn generate_rectangular(window: &mut [f64]) {
    window.fill(1.0);
}

/// Generate Bartlett (triangular) window
fn generate_bartlett(window: &mut [f64]) {
    let n = window.len();
    let n_half = n / 2;

    for (i, w) in window.iter_mut().enumerate() {
        if i <= n_half {
            *w = 2.0 * i as f64 / (n - 1) as f64;
        } else {
            *w = 2.0 * (n - 1 - i) as f64 / (n - 1) as f64;
        }
    }
}

/// Calculate the coherent gain of a window (sum of window values)
pub fn coherent_gain(window: &[f64]) -> f64 {
    window.iter().sum()
}

/// Calculate the power gain of a window (sum of squared window values)  
pub fn power_gain(window: &[f64]) -> f64 {
    window.iter().map(|&w| w * w).sum()
}

/// Calculate the normalized effective noise bandwidth (NENBW) of a window
pub fn nenbw(window: &[f64]) -> f64 {
    let n = window.len() as f64;
    let sum_squared: f64 = window.iter().map(|&w| w * w).sum();
    let sum: f64 = window.iter().sum();

    n * sum_squared / (sum * sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_generation() {
        let size = 512;

        // Test all window types
        for &window_type in WindowType::all() {
            let window = generate_window(window_type, size);
            assert_eq!(window.len(), size);

            // All windows should have non-negative values
            assert!(window.iter().all(|&w| w >= 0.0));

            // Rectangular window should be all ones
            if window_type == WindowType::Rectangular {
                assert!(window.iter().all(|&w| (w - 1.0).abs() < 1e-10));
            }
        }
    }

    #[test]
    fn test_hanning_symmetry() {
        let window = generate_window(WindowType::Hanning, 512);

        // Hanning window should be symmetric
        for i in 0..window.len() / 2 {
            let left = window[i];
            let right = window[window.len() - 1 - i];
            assert!(
                (left - right).abs() < 1e-10,
                "Window not symmetric at position {}: {} != {}",
                i,
                left,
                right
            );
        }
    }

    #[test]
    fn test_window_properties() {
        let window = generate_window(WindowType::Hanning, 512);

        let cg = coherent_gain(&window);
        let pg = power_gain(&window);
        let nenbw_val = nenbw(&window);

        assert!(cg > 0.0);
        assert!(pg > 0.0);
        assert!(nenbw_val > 0.0);

        println!("Hanning window properties:");
        println!("  Coherent gain: {:.3}", cg);
        println!("  Power gain: {:.3}", pg);
        println!("  NENBW: {:.3}", nenbw_val);
    }
}
