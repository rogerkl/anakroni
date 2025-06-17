//! Utility functions for audio processing and formatting
//!
//! Provides helper functions for file I/O, data conversion, and formatting
//! that are used by client applications.

use crate::audio_io::{read_audio_file, write_audio_file, AudioInfo};
use crate::processor::STFTProcessor;
use crate::stft::STFTConfig;
use crate::Result;

/// Load audio from file into an STFTProcessor with analysis
#[cfg(not(target_arch = "wasm32"))]
pub fn load_and_analyze<P: AsRef<std::path::Path>>(
    processor: &mut STFTProcessor,
    path: P,
) -> Result<()> {
    match read_audio_file(path.as_ref()) {
        Ok((audio_info, channel_data)) => {
            log::info!(
                "Loaded audio: {} channels, {} Hz, {:.2}s",
                audio_info.channels,
                audio_info.sample_rate,
                audio_info.duration_seconds
            );

            processor.load_audio(audio_info, channel_data)?;
            processor.analyze()?;

            log::info!(
                "STFT analysis complete: {} frames per channel",
                processor.num_frames().unwrap_or(0)
            );

            Ok(())
        }
        Err(e) => Err(format!("Failed to load audio file: {}", e)),
    }
}

/// Save processed audio from STFTProcessor to file
#[cfg(not(target_arch = "wasm32"))]
pub fn synthesize_and_save<P: AsRef<std::path::Path>>(
    processor: &STFTProcessor,
    path: P,
) -> Result<()> {
    let audio_info = processor
        .audio_info()
        .ok_or("No audio information available")?;

    let synthesized_channels = processor.synthesize()?;

    // Update audio info with potentially new duration
    let new_duration_samples = if !synthesized_channels.is_empty() {
        synthesized_channels[0].len()
    } else {
        0
    };

    let updated_info = AudioInfo::new(
        audio_info.sample_rate,
        audio_info.channels,
        new_duration_samples,
    );

    match write_audio_file(path.as_ref(), &updated_info, &synthesized_channels) {
        Ok(_) => {
            log::info!(
                "Saved audio: {} channels, {:.2}s to {}. Hop size: {}, Window size: {}, Overlap: {}",
                updated_info.channels,
                updated_info.duration_seconds,
                path.as_ref().display(),
                processor.config().hop_size(),
                processor.config().window_size,
                processor.config().overlap_factor
            );
            Ok(())
        }
        Err(e) => Err(format!("Failed to save audio file: {}", e)),
    }
}

/// Format a frequency value for display
pub fn format_frequency(freq_hz: f64) -> String {
    if freq_hz >= 1000.0 {
        format!("{:.2} kHz", freq_hz / 1000.0)
    } else {
        format!("{:.1} Hz", freq_hz)
    }
}

/// Format a time value for display
pub fn format_time(time_sec: f64) -> String {
    if time_sec >= 60.0 {
        let minutes = (time_sec / 60.0).floor();
        let seconds = time_sec % 60.0;
        format!("{:.0}m {:.1}s", minutes, seconds)
    } else {
        format!("{:.2}s", time_sec)
    }
}

/// Format duration in samples to time string
pub fn format_duration(samples: usize, sample_rate: u32) -> String {
    let seconds = samples as f64 / sample_rate as f64;
    format_time(seconds)
}

/// Calculate the frequency corresponding to a given FFT bin
pub fn bin_to_frequency(bin: usize, sample_rate: u32, fft_size: usize) -> f64 {
    bin as f64 * sample_rate as f64 / fft_size as f64
}

/// Calculate the FFT bin corresponding to a given frequency
pub fn frequency_to_bin(frequency: f64, sample_rate: u32, fft_size: usize) -> usize {
    (frequency * fft_size as f64 / sample_rate as f64).round() as usize
}

/// Distribute bins linearly, using either real or complex layout
/// num_freq_bin is the number if frequencies we have
/// Example we do fft on 1024 sample points = 512 num_freq_bins
/// Real fft: dc + num_freqs = 513 bins (0=dc, 512=nyquist)
/// Complex fft: dc + num_freqs + num_freqs-1 = 1024 bins (0=dc, 512=nyquist)
pub fn distribute_lin_bins(num_freq_bins: usize, num_parts: usize, complex: bool) -> Vec<usize> {
    let num_bins = if complex {
        num_freq_bins * 2
    } else {
        num_freq_bins + 1
    };
    let mut bin_assignments = vec![0; num_bins];

    // Assign each bin to a part
    for bin in 0..num_freq_bins + 1 {
        if bin == 0 {
            // DC component - assign to first part
            bin_assignments[bin] = 0;
        } else {
            bin_assignments[bin] = ((((bin - 1) as f32) / (num_freq_bins as f32))
                * (num_parts as f32))
                .floor() as usize;
            if complex && bin <= num_freq_bins {
                bin_assignments[num_bins - bin] = bin_assignments[bin];
            }
        }
    }
    bin_assignments
}

/// Distribute bins logarithmicly, using either real or complex layout
/// num_freq_bin is the number if frequencies we have
/// Example we do fft on 1024 sample points = 512 num_freq_bins
/// Real fft: dc + num_freqs = 513 bins (0=dc, 512=nyquist)
/// Complex fft: dc + num_freqs + num_freqs-1 = 1024 bins (0=dc, 512=nyquist)
pub fn distribute_log_bins(num_freq_bins: usize, num_parts: usize, complex: bool) -> Vec<usize> {
    let num_bins = if complex {
        num_freq_bins * 2
    } else {
        num_freq_bins + 1
    };
    let mut bin_assignments = vec![0; num_bins];

    // don't really need correct samplerate ...
    let sample_rate = num_freq_bins as f32;
    // Calculate frequency for each bin (excluding DC and Nyquist)
    // Start from bin 1 to avoid log(0)
    let min_freq = sample_rate / (2.0 * num_freq_bins as f32); // Frequency of bin 1
    let max_freq = sample_rate / 2.0; // Nyquist frequency

    // Calculate log frequency range
    let log_min = min_freq.ln();
    let log_max = max_freq.ln();
    let log_range = log_max - log_min;

    // Assign each bin to a part
    for bin in 0..num_freq_bins + 1 {
        if bin == 0 {
            // DC component - assign to first part
            bin_assignments[bin] = 0;
        } else {
            // Calculate frequency for this bin
            let freq = (bin as f32) * sample_rate / (2.0 * num_freq_bins as f32);

            // Convert to log scale
            let log_freq = freq.ln();

            // Normalize to 0-1 range
            let normalized = (log_freq - log_min) / log_range;

            // Calculate which part this bin belongs to
            let part = (normalized * num_parts as f32).floor() as usize;

            // Clamp to valid range (in case of floating point errors)
            bin_assignments[bin] = part.min(num_parts - 1);
            if complex && bin <= num_freq_bins {
                bin_assignments[num_bins - bin] = bin_assignments[bin];
            }
        }
    }
    bin_assignments
}

/// Distribute bins with a group size, should pass either &distribute_lin_bins or &distribute_log_bins as the func parameter
/// for linear distribution this is straightforward, say real fft, 16 bins,group size 2, num parts 4
/// bin[0] = 0 (DC, special case)
/// bin[1] = 0
/// bin[2] = 0
/// bin[3] = 1 // group size == 2, so change value after 2 bins
/// bin[4] = 1
/// bin[5] = 2
/// bin[6] = 2
/// bin[7] = 3
/// bin[8] = 3
/// bin[9] = 0 // number of parts reached, start at 0 again
/// bin[10] = 0
/// bin[11] = 1
/// bin[12] = 1
/// bin[13] = 2
/// bin[14] = 2
/// bin[15] = 3
/// bin[16] = 4
/// For the logarithmic case it's a little more complicated, the algorithm is:
///   we distribute first in num_freq_bins/group_size parts
///   then remap to num_parts using modulu operator
/// The result is that we should still have the same musical frequencies per part as for
/// the normal logarithmic distribution
pub fn distribute_grouped(
    num_freq_bins: usize,
    num_parts: usize,
    complex: bool,
    group_size: usize,
    func: &dyn Fn(usize, usize, bool) -> Vec<usize>,
) -> Vec<usize> {
    if group_size < 1 {
        return func(num_freq_bins, num_parts, complex);
    }
    let mut bin_assignments = func(num_freq_bins, num_freq_bins / group_size, complex);
    for bin in 1..num_freq_bins + 1 {
        bin_assignments[bin] = bin_assignments[bin] % num_parts;
    }
    bin_assignments
}

/// Get analysis summary string for an STFTProcessor
pub fn analysis_summary(processor: &STFTProcessor) -> String {
    let mut summary = String::new();

    if let Some(info) = processor.audio_info() {
        summary.push_str(&format!(
            "Audio: {} channels, {} Hz, {}\n",
            info.channels,
            info.sample_rate,
            format_duration(info.duration_samples, info.sample_rate)
        ));
    }

    let config = processor.config();
    summary.push_str("STFT Config:\n");
    summary.push_str(&format!("  Window size: {} samples\n", config.window_size));
    summary.push_str(&format!(
        "  Overlap: {}% ({} factor)\n",
        config.overlap_percent(),
        config.overlap_factor
    ));
    summary.push_str(&format!("  Hop size: {} samples\n", config.hop_size()));
    summary.push_str(&format!("  Window type: {}\n", config.window_type.name()));
    summary.push_str(&format!("  FFT bins: {}\n", config.fft_bins()));

    if let Some(num_frames) = processor.num_frames() {
        summary.push_str(&format!("  Frames per channel: {}\n", num_frames));

        if let Some(info) = processor.audio_info() {
            let frame_duration = config.hop_size() as f64 / info.sample_rate as f64;
            let total_duration = num_frames as f64 * frame_duration;
            summary.push_str(&format!(
                "  Frame duration: {:.2}ms\n",
                frame_duration * 1000.0
            ));
            summary.push_str(&format!(
                "  Analysis duration: {}\n",
                format_time(total_duration)
            ));
        }
    }

    // Add temporal FFT information if available
    if let Some(temporal) = processor.temporal_fft() {
        summary.push_str("Temporal FFT Analysis:\n");
        summary.push_str(&format!("  Channels analyzed: {}\n", temporal.num_channels));
        summary.push_str(&format!(
            "  Temporal FFT size: {}\n",
            temporal.config.fft_size
        ));

        // Show multiplier information
        if temporal.config.length_multiplier > 1 {
            summary.push_str(&format!(
                "  Length multiplier: {}x\n",
                temporal.config.length_multiplier
            ));
            summary.push_str(&format!(
                "  Effective temporal length: {}\n",
                temporal.effective_temporal_bins()
            ));
        }

        summary.push_str(&format!(
            "  Temporal frequency resolution: {:.6}\n",
            temporal.temporal_frequency_resolution()
        ));
        summary.push_str(&format!(
            "  Frequency bins per channel: {}\n",
            temporal.num_frequency_bins
        ));

        // Show per-channel information if multiple channels
        if temporal.num_channels > 1 {
            summary.push_str("  Per-channel details:\n");
            for ch_idx in 0..temporal.num_channels {
                let channel_bins: Vec<_> = temporal
                    .bin_ffts
                    .iter()
                    .filter(|bin_fft| bin_fft.channel_index == ch_idx)
                    .collect();
                summary.push_str(&format!(
                    "    Channel {}: {} frequency bins\n",
                    ch_idx,
                    channel_bins.len()
                ));
            }
        }
    }

    summary
}

/// Validate STFT configuration for given audio parameters
pub fn validate_config_for_audio(
    config: &STFTConfig,
    sample_rate: u32,
    duration_samples: usize,
) -> Result<()> {
    if duration_samples < config.window_size {
        return Err(format!(
            "Audio duration ({} samples) is shorter than window size ({})",
            duration_samples, config.window_size
        ));
    }

    let nyquist = sample_rate as f64 / 2.0;
    let freq_resolution = sample_rate as f64 / config.window_size as f64;

    log::info!("STFT validation:");
    log::info!(
        "  Frequency resolution: {}",
        format_frequency(freq_resolution)
    );
    log::info!("  Nyquist frequency: {}", format_frequency(nyquist));
    log::info!(
        "  Time resolution: {:.2}ms",
        config.hop_size() as f64 / sample_rate as f64 * 1000.0
    );

    if freq_resolution > 50.0 {
        log::warn!(
            "Frequency resolution ({}) is quite coarse, consider larger window size",
            format_frequency(freq_resolution)
        );
    }

    if config.hop_size() as f64 / sample_rate as f64 > 0.1 {
        log::warn!(
            "Time resolution ({:.2}ms) is quite coarse, consider higher overlap factor",
            config.hop_size() as f64 / sample_rate as f64 * 1000.0
        );
    }

    Ok(())
}

/// Create commonly used STFT configurations
pub mod presets {
    use super::*;
    use crate::window::WindowType;

    /// Preset information structure
    pub struct PresetInfo {
        pub id: usize,
        pub name: &'static str,
        pub description: &'static str,
        pub config: STFTConfig,
    }

    /// Default
    pub fn default() -> STFTConfig {
        STFTConfig::new(1024, 8, WindowType::Hanning).unwrap()
    }

    /// High time resolution (small hop size)
    pub fn high_time_resolution() -> STFTConfig {
        STFTConfig::new(1024, 16, WindowType::Hanning).unwrap()
    }

    /// High frequency resolution (large window)
    pub fn high_freq_resolution() -> STFTConfig {
        STFTConfig::new(4096, 8, WindowType::Hanning).unwrap()
    }

    /// Fast processing (less overlap)
    pub fn fast_processing() -> STFTConfig {
        STFTConfig::new(512, 2, WindowType::Hanning).unwrap()
    }

    /// Music analysis (balanced resolution)
    pub fn music_analysis() -> STFTConfig {
        STFTConfig::new(2048, 8, WindowType::Hanning).unwrap()
    }

    /// Speech analysis (optimized for voice)
    pub fn speech_analysis() -> STFTConfig {
        STFTConfig::new(512, 4, WindowType::Hamming).unwrap()
    }

    /// Get all presets with their names
    pub fn all_presets() -> Vec<(&'static str, STFTConfig)> {
        vec![
            ("Default", default()),
            ("High Time Resolution", high_time_resolution()),
            ("High Frequency Resolution", high_freq_resolution()),
            ("Fast Processing", fast_processing()),
            ("Music Analysis", music_analysis()),
            ("Speech Analysis", speech_analysis()),
        ]
    }

    /// List all presets with detailed info
    pub fn list_presets() -> Vec<PresetInfo> {
        vec![
            PresetInfo {
                id: 0,
                name: "Default",
                description: "Window=1024, Overlap=8",
                config: default(),
            },
            PresetInfo {
                id: 1,
                name: "High Time Resolution",
                description: "Window=1024, Overlap=16",
                config: high_time_resolution(),
            },
            PresetInfo {
                id: 2,
                name: "High Frequency Resolution",
                description: "Window=4096, Overlap=8",
                config: high_freq_resolution(),
            },
            PresetInfo {
                id: 3,
                name: "Fast Processing",
                description: "Window=512, Overlap=2",
                config: fast_processing(),
            },
            PresetInfo {
                id: 4,
                name: "Music Analysis",
                description: "Window=2048, Overlap=8",
                config: music_analysis(),
            },
            PresetInfo {
                id: 5,
                name: "Speech Analysis",
                description: "Window=512, Overlap=4",
                config: speech_analysis(),
            },
        ]
    }

    /// Get a preset by ID
    pub fn get_preset(id: usize) -> Option<PresetInfo> {
        list_presets().into_iter().find(|p| p.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_formatting() {
        assert_eq!(format_frequency(440.0), "440.0 Hz");
        assert_eq!(format_frequency(1000.0), "1.00 kHz");
        assert_eq!(format_frequency(22050.0), "22.05 kHz");
    }

    #[test]
    fn test_time_formatting() {
        assert_eq!(format_time(0.5), "0.50s");
        assert_eq!(format_time(45.0), "45.00s");
        assert_eq!(format_time(75.5), "1m 15.5s");
        assert_eq!(format_time(125.0), "2m 5.0s");
    }

    #[test]
    fn test_bin_frequency_conversion() {
        let sample_rate = 44100;
        let fft_size = 1024;

        // Test bin 0 (DC)
        assert_eq!(bin_to_frequency(0, sample_rate, fft_size), 0.0);

        // Test Nyquist bin
        let nyquist_bin = fft_size / 2;
        let nyquist_freq = bin_to_frequency(nyquist_bin, sample_rate, fft_size);
        assert!((nyquist_freq - 22050.0).abs() < 1e-10);

        // Test round-trip conversion
        let test_freq = 440.0;
        let bin = frequency_to_bin(test_freq, sample_rate, fft_size);
        let back_to_freq = bin_to_frequency(bin, sample_rate, fft_size);
        assert!((test_freq - back_to_freq).abs() < 50.0); // Within reasonable tolerance
    }

    #[test]
    fn test_presets() {
        let presets = presets::all_presets();
        assert_eq!(presets.len(), 5);

        for (name, config) in presets {
            println!("Testing preset: {}", name);
            assert!(config.window_size >= 256);
            assert!(config.window_size <= 4096);
            assert!(config.overlap_factor >= 1);
            assert!(config.overlap_factor <= 32);
        }
    }
}
