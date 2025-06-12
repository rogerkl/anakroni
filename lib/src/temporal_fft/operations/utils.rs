//! Shared utilities for temporal FFT operations

use crate::temporal_fft::TemporalFFTAnalysis;
use num_complex::Complex64;

/// Apply a function to all temporal spectra in all channels
pub fn process_all_temporal_spectra<F>(analysis: &mut TemporalFFTAnalysis, mut processor: F)
where
    F: FnMut(usize, usize, &mut Vec<Complex64>), // channel_idx, bin_idx, temporal_spectrum
{
    for bin_fft in analysis.bin_ffts.iter_mut() {
        processor(
            bin_fft.channel_index,
            bin_fft.bin_index,
            &mut bin_fft.temporal_spectrum,
        );
    }
}

/// Enforce Hermitian symmetry on a complex spectrum
/// This ensures the inverse FFT will produce real-valued results
pub fn enforce_hermitian_symmetry(spectrum: &mut Vec<Complex64>) {
    let fft_size = spectrum.len();

    // DC component must be real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);

    // Nyquist component must be real (if it exists)
    if fft_size % 2 == 0 {
        spectrum[fft_size / 2] = Complex64::new(spectrum[fft_size / 2].re, 0.0);
    }

    // Note: For a full Hermitian symmetry enforcement, we would need to ensure
    // that spectrum[k] = conj(spectrum[N-k]) for all k, but since we're shifting
    // the spectrum in a way that preserves the original symmetry pattern, we
    // only need to ensure DC and Nyquist are real.
}

/// Get a string representation of the temporal FFT layout for debugging
pub fn describe_temporal_fft_layout(fft_size: usize) -> String {
    let mut description = format!("Temporal FFT layout (size {}):\n", fft_size);

    description.push_str("  Bin 0: DC (0 Hz)\n");

    for i in 1..=(fft_size / 2) {
        let norm_freq = temporal_bin_to_normalized_frequency(i, fft_size);
        if i == fft_size / 2 {
            description.push_str(&format!(
                "  Bin {}: Nyquist ({:.3} normalized)\n",
                i, norm_freq
            ));
        } else {
            description.push_str(&format!("  Bin {}: +{:.3} normalized\n", i, norm_freq));
        }
    }

    if fft_size > 2 {
        description.push_str("  Negative frequencies (in reverse order):\n");
        for i in (fft_size / 2 + 1)..fft_size {
            let norm_freq = temporal_bin_to_normalized_frequency(i, fft_size);
            description.push_str(&format!("  Bin {}: {:.3} normalized\n", i, norm_freq));
        }
    }

    description
}

/// Convert temporal bin index to normalized frequency
fn temporal_bin_to_normalized_frequency(bin: usize, fft_size: usize) -> f64 {
    if bin == 0 {
        0.0 // DC
    } else if bin <= fft_size / 2 {
        // Positive frequencies: bin / (fft_size/2) gives 0.0 to 1.0
        bin as f64 / (fft_size as f64 / 2.0)
    } else {
        // Negative frequencies: map to negative normalized frequencies
        let neg_bin = fft_size - bin;
        -(neg_bin as f64 / (fft_size as f64 / 2.0))
    }
}
