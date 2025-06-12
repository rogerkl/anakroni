//! STFT Audio Processor CLI
//!
//! Command-line interface for the STFT Audio Processor library.
//! Provides an interactive shell for STFT-based audio processing operations.

use std::process;

use clap::{Arg, Command};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use stft_audio_lib::{
    processor::STFTProcessor,
    stft::STFTConfig,
    temporal_fft::TemporalFFTConfig,
    utils::{self, presets},
    window::WindowType,
    Complex64,
};

#[cfg(feature = "image")]
use stft_audio_lib::spectrogram::{
    image::{save_spectrogram, ColorMap, SpectrogramImageOptions},
    FrequencyScale, Spectrogram,
};

/// Application state
struct AppState {
    processor: STFTProcessor,
    current_file: Option<String>,
}

impl AppState {
    fn new() -> Self {
        Self {
            processor: STFTProcessor::new(),
            current_file: None,
        }
    }
}

/// Print the help message showing available commands
fn print_help() {
    println!("Available commands:");
    println!("  load <filename>                    - Load an audio file and perform STFT analysis");
    println!("  save <filename>                    - Save the processed audio to a file");
    println!("  config                             - Show current STFT configuration");
    println!(
        "  set window_size <size>             - Set FFT window size (256, 512, 1024, 2048, 4096)"
    );
    println!("  set overlap <factor>               - Set overlap factor (1-32)");
    println!("  set window_type <type>             - Set window type (hanning, hamming, rectangular, bartlett)");
    println!("  set temporal_size <size>           - Set temporal FFT size (power of 2)");
    println!("  preset <n>                         - Load a configuration preset");
    println!("  presets                            - List available presets");
    println!("  info                               - Show information about loaded audio");
    println!(
        "  analyze                            - Re-analyze current audio with current settings"
    );
    println!("  temporal_analyze [multiplier]      - Perform temporal FFT analysis on STFT frames");
    println!("  temporal_config                    - Show temporal FFT configuration");
    println!("  temporal_process <operation>       - Apply temporal processing operations");
    println!("  temporal_synthesize                - Convert temporal FFT back to STFT frames");
    println!("  temporal_synthesize_osc <stretch> [params...]   - Oscillator bank synthesis with parameter control");
    println!("  process <operation> [params...]    - Apply processing operations");
    println!("  spectrogram <filename> [options]   - Generate spectrogram image");
    println!("  temporal_spectrogram <filename>    - Generate temporal FFT visualization");
    println!("  status                             - Show processor status");
    println!("  help                               - Show this help message");
    println!("  quit                               - Exit the program");
    println!();
    println!("Processing operations:");
    println!("  zero_dc                            - Zero out DC component");
    println!("  highpass <freq_hz>                 - Simple highpass filter");
    println!("  lowpass <freq_hz>                  - Simple lowpass filter");
    println!();
    println!("Temporal FFT Analysis:");
    println!("  multiplier: Optional length multiplier (default: 1)");
    println!();
    println!("Temporal processing operations:");
    print_temporal_processing_operations();
    println!();
    println!("Examples:");
    println!("  load test.wav");
    println!("  set window_size 2048");
    println!("  set overlap 16");
    println!("  preset music_analysis");
    println!("  process highpass 80");
    println!("  temporal_analyze          # Standard temporal analysis");
    println!("  temporal_analyze 2        # 2x length multiplier");
    println!("  temporal_process temporal_highpass 0.1");
    println!("  temporal_process temporal_shift 100");
    println!("  temporal_process temporal_stretch 0.25");
    println!("  temporal_process temporal_stretch 2.0");
    println!("  temporal_synthesize");
    println!("  spectrogram output.png");
    println!("  spectrogram spec.png all plasma 1024 768");
    println!("  spectrogram spec_log.png 0 viridis 800 600 log");
    println!("  temporal_spectrogram temporal_vis.png");
    println!("  temporal_spectrogram temporal_all.png all inferno");
    println!("  save processed.wav");
}

fn print_temporal_processing_operations() {
    println!("  zero_temporal_dc                   - Zero out temporal DC component");
    println!("  temporal_highpass <cutoff>         - Temporal highpass filter (0.0-1.0)");
    println!("  temporal_lowpass <cutoff>          - Temporal lowpass filter (0.0-1.0)");
    println!("  temporal_shift <frames>            - Shift temporal fft bins");
    println!("  temporal_shift_dispersive <base> <factor> - Frequency-dependent shift");
    println!("  temporal_shift_circular <frames>   - Circular shift (wrap-around)");
    println!("  temporal_stretch <factor>          - Stretch temporal FFT - <1 time stretch, >1 time compress");
    println!("  temporal_phase_multiply <factor>   - Multiply all phases by factor");
    println!("  temporal_phase_multiply_dispersive <base> <scaling> - Frequency-dependent phase multiply");
    println!("  temporal_phase_reversal            - Apply phase reversal (-1.0 factor)");
    println!("  temporal_phase_scrambling <intensity> <seed> - Pseudo-random phase scrambling");
    println!("  temporal_blur <radius> <strength>  - Smooth temporal frequencies");
    println!("  temporal_convolve <impulse_file>   - Convolve with impulse response");
    println!("  temporal_power_amplitude <factor>  - Raise amplitude to factor, normalized by all frequency bins");
    println!("  temporal_power_amplitude_per_bin <factor> - Raise amplitude to factor, normalized by current frequency bin");
}

/// Process a user command
fn process_command(command: &str, state: &mut AppState) {
    let parts: Vec<&str> = command.split_whitespace().collect();

    if parts.is_empty() {
        return;
    }

    match parts[0] {
        "load" => {
            if parts.len() != 2 {
                println!("Usage: load <filename>");
                return;
            }

            let filename = parts[1];
            println!("Loading file: {}", filename);

            match utils::load_and_analyze(&mut state.processor, filename) {
                Ok(_) => {
                    state.current_file = Some(filename.to_string());
                    println!("File loaded successfully!");
                    println!("{}", utils::analysis_summary(&state.processor));
                }
                Err(e) => println!("Error loading file: {}", e),
            }
        }

        "save" => {
            if parts.len() != 2 {
                println!("Usage: save <filename>");
                return;
            }

            if !state.processor.has_analysis() {
                println!("No audio loaded or analyzed. Load a file first.");
                return;
            }

            let filename = parts[1];
            println!("Saving to file: {}", filename);

            match utils::synthesize_and_save(&state.processor, filename) {
                Ok(_) => println!("File saved successfully!"),
                Err(e) => println!("Error saving file: {}", e),
            }
        }

        "config" => {
            let config = state.processor.config();
            println!("Current STFT Configuration:");
            println!("  Window size: {}", config.window_size);
            println!("  Overlap factor: {}", config.overlap_factor);
            println!("  Hop size: {}", config.hop_size());
            println!("  Window type: {:?}", config.window_type);
        }

        "temporal_config" => {
            let config = state.processor.temporal_config();
            println!("Current Temporal FFT Configuration:");
            println!("  FFT size: {}", config.fft_size);
            println!("  Length multiplier: {}x", config.length_multiplier);

            if let Some(temporal) = state.processor.temporal_fft() {
                println!("{}", temporal.describe_config());
            }
        }

        "set" => {
            if parts.len() < 3 {
                println!("Usage: set <parameter> <value>");
                println!("Parameters: window_size, overlap, window_type, temporal_size, temporal_multiplier");
                return;
            }

            let param = parts[1];
            let value = parts[2];

            match param {
                "window_size" => match value.parse::<usize>() {
                    Ok(size) => {
                        let current_config = state.processor.config().clone();
                        match STFTConfig::new(
                            size,
                            current_config.overlap_factor,
                            current_config.window_type,
                        ) {
                            Ok(new_config) => {
                                state.processor.set_config(new_config);
                                println!("Window size set to {}", size);
                            }
                            Err(e) => println!("Error setting window size: {}", e),
                        }
                    }
                    Err(_) => println!("Invalid window size: {}", value),
                },

                "overlap" => match value.parse::<usize>() {
                    Ok(overlap) => {
                        let current_config = state.processor.config().clone();
                        match STFTConfig::new(
                            current_config.window_size,
                            overlap,
                            current_config.window_type,
                        ) {
                            Ok(new_config) => {
                                state.processor.set_config(new_config);
                                println!("Overlap factor set to {}", overlap);
                            }
                            Err(e) => println!("Error setting overlap: {}", e),
                        }
                    }
                    Err(_) => println!("Invalid overlap factor: {}", value),
                },

                "window_type" => {
                    let window_type = match value {
                        "hanning" => WindowType::Hanning,
                        "hamming" => WindowType::Hamming,
                        "rectangular" => WindowType::Rectangular,
                        "bartlett" => WindowType::Bartlett,
                        _ => {
                            println!("Invalid window type: {}", value);
                            println!("Valid types: hanning, hamming, rectangular, bartlett");
                            return;
                        }
                    };

                    let current_config = state.processor.config().clone();
                    match STFTConfig::new(
                        current_config.window_size,
                        current_config.overlap_factor,
                        window_type,
                    ) {
                        Ok(new_config) => {
                            state.processor.set_config(new_config);
                            println!("Window type set to {}", window_type);
                        }
                        Err(e) => println!("Error setting window type: {}", e),
                    }
                }

                "temporal_size" => match value.parse::<usize>() {
                    Ok(size) => match TemporalFFTConfig::new(size) {
                        Ok(config) => {
                            state.processor.set_temporal_config(config);
                            println!("Temporal FFT size set to {}", size);
                        }
                        Err(e) => println!("Error setting temporal size: {}", e),
                    },
                    Err(_) => println!("Invalid temporal size: {}", value),
                },

                "temporal_multiplier" => {
                    match value.parse::<usize>() {
                        Ok(multiplier) if multiplier >= 1 => {
                            let current_config = *state.processor.temporal_config();
                            match TemporalFFTConfig::new_with_multiplier(
                                current_config.fft_size,
                                multiplier,
                            ) {
                                Ok(new_config) => {
                                    state.processor.set_temporal_config(new_config);
                                    println!("Temporal length multiplier set to {}x", multiplier);
                                    if multiplier > 1 {
                                        println!("Note: Re-run temporal_analyze to apply the new multiplier");
                                    }
                                }
                                Err(e) => println!("Error setting temporal multiplier: {}", e),
                            }
                        }
                        Ok(multiplier) => {
                            println!("Temporal multiplier must be at least 1, got {}", multiplier)
                        }
                        Err(_) => println!("Invalid temporal multiplier: {}", value),
                    }
                }

                _ => {
                    println!("Unknown parameter: {}", param);
                    println!("Valid parameters: window_size, overlap, window_type, temporal_size, temporal_multiplier");
                }
            }
        }

        "preset" => {
            if parts.len() != 2 {
                println!("Usage: preset <number>");
                return;
            }

            match parts[1].parse::<usize>() {
                Ok(n) => {
                    if let Some(preset) = presets::get_preset(n) {
                        state.processor.set_config(preset.config.clone());
                        println!("Applied preset {}: {}", n, preset.name);
                        println!("{}", preset.description);
                    } else {
                        println!("Invalid preset number: {}", n);
                    }
                }
                Err(_) => println!("Invalid preset number: {}", parts[1]),
            }
        }

        "presets" => {
            println!("Available presets:");
            for preset in presets::list_presets() {
                println!("  {}: {} - {}", preset.id, preset.name, preset.description);
            }
        }

        "info" => {
            if state.processor.has_audio() {
                println!("{}", utils::analysis_summary(&state.processor));
            } else {
                println!("No audio data available");
            }
        }

        "analyze" => {
            if !state.processor.has_audio() {
                println!("No audio loaded. Load a file first.");
                return;
            }

            println!("Re-analyzing audio with current configuration...");
            match state.processor.analyze() {
                Ok(_) => {
                    println!("Analysis complete!");
                    println!(
                        "Generated {} frames per channel",
                        state.processor.num_frames().unwrap_or(0)
                    );
                }
                Err(e) => println!("Error during analysis: {}", e),
            }
        }

        "temporal_analyze" => {
            if !state.processor.has_analysis() {
                println!("No STFT analysis available. Load and analyze a file first.");
                return;
            }

            // Parse optional multiplier parameter
            let length_multiplier = if parts.len() >= 2 {
                match parts[1].parse::<usize>() {
                    Ok(mult) if mult >= 1 => mult,
                    Ok(mult) => {
                        println!("Error: multiplier must be at least 1, got {}", mult);
                        return;
                    }
                    Err(_) => {
                        println!("Error: multiplier must be a positive integer");
                        return;
                    }
                }
            } else {
                1 // Default multiplier
            };

            if length_multiplier > 1 {
                println!(
                    "Performing temporal FFT analysis with {}x length multiplier...",
                    length_multiplier
                );
            } else {
                println!("Performing temporal FFT analysis...");
            }

            // Get the number of frames to calculate appropriate FFT size
            let num_frames = state.processor.num_frames().unwrap_or(0);
            if num_frames == 0 {
                println!("No STFT frames available for temporal analysis");
                return;
            }

            // Create temporal config with multiplier
            let temporal_config = if length_multiplier > 1 {
                TemporalFFTConfig::auto_size_with_multiplier(num_frames, length_multiplier)
            } else {
                // Use existing config or create auto-sized one
                if state.processor.temporal_config().fft_size < num_frames {
                    TemporalFFTConfig::auto_size_with_multiplier(num_frames, 1)
                } else {
                    *state.processor.temporal_config()
                }
            };

            // Set the new config
            state.processor.set_temporal_config(temporal_config);

            match state.processor.analyze_temporal() {
                Ok(_) => {
                    println!("Temporal FFT analysis complete!");
                    if let Some(temporal) = state.processor.temporal_fft() {
                        println!("{}", temporal.describe_config());
                        println!(
                            "Processed {} frequency bins with {} temporal bins each",
                            temporal.num_frequency_bins,
                            temporal.num_temporal_bins()
                        );

                        if length_multiplier > 1 {
                            println!(
                                "Effective temporal length: {} frames ({}x multiplier applied)",
                                temporal.effective_temporal_bins(),
                                length_multiplier
                            );
                            println!("Benefits:");
                            println!("  - Higher temporal frequency resolution");
                            println!(
                                "  - Better stretching capabilities (up to {}x expansion)",
                                length_multiplier
                            );
                            println!("  - More detailed temporal frequency analysis");
                        }
                    }
                }
                Err(e) => println!("Error during temporal analysis: {}", e),
            }
        }
        "temporal_synthesize" => {
            if !state.processor.has_temporal_analysis() {
                println!("No temporal FFT analysis available. Run temporal_analyze first.");
                return;
            }

            println!("Synthesizing STFT frames from temporal FFT...");
            match state.processor.synthesize_from_temporal() {
                Ok(_) => println!("Temporal synthesis complete!"),
                Err(e) => println!("Error during temporal synthesis: {}", e),
            }
        }

        "temporal_synthesize_osc" => {
            if parts.len() < 2 {
                println!("Usage: temporal_synthesize_osc <stretch_start> [shift_start] [dispersion_start] [stretch_end] [shift_end] [dispersion_end] [output_frames]");
                println!("  stretch_start: Temporal frequency scaling factor (1.0 = no change)");
                println!("  shift_start: Temporal frequency shift (-1.0 to 1.0, default: 0.0)");
                println!("  dispersion_start: Frequency-dependent scaling (default: 1.0)");
                println!("  stretch_end: Final stretch factor (default: same as start)");
                println!("  shift_end: Final shift factor (default: same as start)");
                println!("  dispersion_end: Final dispersion factor (default: same as start)");
                println!(
                    "  output_frames: Number of output frames (default: original frame count)"
                );
                println!();
                println!("Examples:");
                println!("temporal_synthesize_osc 1.0                    # No modification");
                println!("temporal_synthesize_osc 0.5                    # Compress temporal frequencies");
                println!(
                    "temporal_synthesize_osc 2.0 0.1                # Expand + slight upward shift"
                );
                println!("temporal_synthesize_osc 1.0 0.0 1.5            # Increase dispersion");
                println!("temporal_synthesize_osc 0.5 -0.1 0.8 2.0 0.1 1.2  # Full sweep from compressed to expanded");
                println!(
                    "temporal_synthesize_osc 1.0 0.0 1.0 1.0 0.0 1.0 64  # Synthesize 64 frames"
                );
                return;
            }

            if !state.processor.has_temporal_analysis() {
                println!("No temporal FFT analysis available. Run temporal_analyze first.");
                return;
            }

            // Parse stretch_start (required)
            let stretch_start = match parts[1].parse::<f64>() {
                Ok(v) => {
                    if v <= 0.0 {
                        println!("Error: stretch_start must be positive, got {}", v);
                        return;
                    }
                    v
                }
                Err(_) => {
                    println!("Error: stretch_start must be a positive number");
                    return;
                }
            };

            // Parse optional parameters
            let shift_start = if parts.len() > 2 {
                match parts[2].parse::<f64>() {
                    Ok(v) => {
                        if v < -1.0 || v > 1.0 {
                            println!("Error: shift_start must be between -1.0 and 1.0, got {}", v);
                            return;
                        }
                        Some(v)
                    }
                    Err(_) => {
                        println!("Error: shift_start must be a number between -1.0 and 1.0");
                        return;
                    }
                }
            } else {
                None
            };

            let dispersion_start = if parts.len() > 3 {
                match parts[3].parse::<f64>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        println!("Error: dispersion_start must be a number");
                        return;
                    }
                }
            } else {
                None
            };

            let stretch_end = if parts.len() > 4 {
                match parts[4].parse::<f64>() {
                    Ok(v) => {
                        if v <= 0.0 {
                            println!("Error: stretch_end must be positive, got {}", v);
                            return;
                        }
                        Some(v)
                    }
                    Err(_) => {
                        println!("Error: stretch_end must be a positive number");
                        return;
                    }
                }
            } else {
                None
            };

            let shift_end = if parts.len() > 5 {
                match parts[5].parse::<f64>() {
                    Ok(v) => {
                        if v < -1.0 || v > 1.0 {
                            println!("Error: shift_end must be between -1.0 and 1.0, got {}", v);
                            return;
                        }
                        Some(v)
                    }
                    Err(_) => {
                        println!("Error: shift_end must be a number between -1.0 and 1.0");
                        return;
                    }
                }
            } else {
                None
            };

            let dispersion_end = if parts.len() > 6 {
                match parts[6].parse::<f64>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        println!("Error: dispersion_end must be a number");
                        return;
                    }
                }
            } else {
                None
            };

            let output_frames = if parts.len() > 7 {
                match parts[7].parse::<usize>() {
                    Ok(v) => {
                        if v == 0 {
                            println!("Error: output_frames must be greater than 0");
                            return;
                        }
                        Some(v)
                    }
                    Err(_) => {
                        println!("Error: output_frames must be a positive integer");
                        return;
                    }
                }
            } else {
                None
            };

            // Get the current hop size
            let hop_size = state.processor.config().hop_size();

            // Perform oscillator bank synthesis using the processor method
            match state.processor.synthesize_temporal_with_oscillator_bank(
                stretch_start,
                shift_start,
                dispersion_start,
                stretch_end,
                shift_end,
                dispersion_end,
                output_frames,
            ) {
                Ok(_) => {
                    println!("Oscillator bank synthesis complete!");

                    // Get the validated parameters for display
                    if let Ok((stretch_s, shift_s, disp_s, stretch_e, shift_e, disp_e)) =
                        state.processor.validate_oscillator_params(
                            stretch_start,
                            shift_start,
                            dispersion_start,
                            stretch_end,
                            shift_end,
                            dispersion_end,
                        )
                    {
                        let frame_count = if let Some(frames) = state.processor.stft_frames() {
                            frames.first().map(|ch| ch.len()).unwrap_or(0)
                        } else {
                            0
                        };

                        // Display detailed synthesis description
                        let description = state.processor.describe_oscillator_synthesis(
                            stretch_s,
                            shift_s,
                            disp_s,
                            stretch_e,
                            shift_e,
                            disp_e,
                            frame_count,
                        );
                        println!("{}", description);
                    }

                    println!("Synthesized frames are now ready for saving or further processing.");
                }
                Err(e) => println!("Error during oscillator bank synthesis: {}", e),
            }
        }

        "temporal_synthesize_osc_preset" => {
            if parts.len() != 2 {
                println!("Usage: temporal_process temporal_synthesize_osc_preset <preset_name>");
                println!("Available presets: Identity,Time Stretch,Time Compress,Shift Up,Shift Down,");
                println!("                   Dispersive,Time Stretch to Compress,Frequency Sweep,");
                println!("                   Complex Morph,Extreme Stretch");
                println!("Use 'oscillator_presets' command to see detailed preset parameters.");
                return;
            }

            if !state.processor.has_temporal_analysis() {
                println!("No temporal FFT analysis available. Run temporal_analyze first.");
                return;
            }

            let preset_name = parts[1];
            let presets = state.processor.get_oscillator_presets();

            if let Some((_, (stretch_s, shift_s, disp_s, stretch_e, shift_e, disp_e))) = presets
                .iter()
                .find(|(name, _)| name.to_lowercase() == preset_name.to_lowercase())
            {
                match state.processor.synthesize_temporal_with_oscillator_bank(
                    *stretch_s,
                    Some(*shift_s),
                    Some(*disp_s),
                    *stretch_e,
                    *shift_e,
                    *disp_e,
                    None, // default frame count
                ) {
                    Ok(_) => {
                        println!("Applied oscillator bank preset: {}", preset_name);

                        // Show what was applied
                        if let Ok((
                            stretch_s_val,
                            shift_s_val,
                            disp_s_val,
                            stretch_e_val,
                            shift_e_val,
                            disp_e_val,
                        )) = state.processor.validate_oscillator_params(
                            *stretch_s,
                            Some(*shift_s),
                            Some(*disp_s),
                            *stretch_e,
                            *shift_e,
                            *disp_e,
                        ) {
                            let frame_count = if let Some(frames) = state.processor.stft_frames() {
                                frames.first().map(|ch| ch.len()).unwrap_or(0)
                            } else {
                                0
                            };

                            let description = state.processor.describe_oscillator_synthesis(
                                stretch_s_val,
                                shift_s_val,
                                disp_s_val,
                                stretch_e_val,
                                shift_e_val,
                                disp_e_val,
                                frame_count,
                            );
                            println!("{}", description);
                        }
                    }
                    Err(e) => println!("Error applying preset '{}': {}", preset_name, e),
                }
            } else {
                println!("Unknown preset: {}", preset_name);
                println!("Use 'oscillator_presets' to see available presets.");
            }
        }

        "oscillator_presets" | "osc_presets" => {
            println!("Available oscillator bank presets:");
            println!("Use: temporal_process temporal_synthesize_osc_preset <name>");
            println!();

            let presets = state.processor.get_oscillator_presets();
            for (name, (stretch_s, shift_s, disp_s, stretch_e, shift_e, disp_e)) in presets {
                println!("  {:<15} - stretch: {:.2}", name, stretch_s);
                if shift_s != 0.0 {
                    println!("                    shift: {:.2}", shift_s);
                }
                if disp_s != 1.0 {
                    println!("                    dispersion: {:.2}", disp_s);
                }
                if let Some(se) = stretch_e {
                    if se != stretch_s {
                        println!("                    → stretch: {:.2}", se);
                    }
                }
                if let Some(she) = shift_e {
                    if she != shift_s {
                        println!("                    → shift: {:.2}", she);
                    }
                }
                if let Some(de) = disp_e {
                    if de != disp_s {
                        println!("                    → dispersion: {:.2}", de);
                    }
                }
                println!();
            }
        }

        "process" => {
            if parts.len() < 2 {
                println!("Usage: process <operation> [parameters...]");
                println!("Available operations: zero_dc, highpass <freq>, lowpass <freq>");
                return;
            }

            if !state.processor.has_analysis() {
                println!("No STFT analysis available. Load and analyze a file first.");
                return;
            }

            let operation = parts[1];
            match operation {
                "zero_dc" => {
                    match state.processor.process_all_frames(|_ch, _frame, frame| {
                        frame.spectrum[0] = Complex64::new(0.0, 0.0);
                    }) {
                        Ok(_) => println!("DC component zeroed out"),
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "highpass" => {
                    if parts.len() != 3 {
                        println!("Usage: process highpass <frequency_hz>");
                        return;
                    }

                    match parts[2].parse::<f64>() {
                        Ok(freq_hz) => {
                            if let Some(info) = state.processor.audio_info() {
                                let sample_rate = info.sample_rate as f64;
                                let freq_bin =
                                    (freq_hz * state.processor.config().window_size as f64
                                        / sample_rate) as usize;

                                match state.processor.process_all_frames(|_ch, _frame, frame| {
                                    for i in 0..=freq_bin.min(frame.spectrum.len() - 1) {
                                        frame.spectrum[i] = Complex64::new(0.0, 0.0);
                                    }
                                }) {
                                    Ok(_) => println!(
                                        "Applied highpass filter at {}",
                                        utils::format_frequency(freq_hz)
                                    ),
                                    Err(e) => println!("Error processing: {}", e),
                                }
                            } else {
                                println!("No audio information available");
                            }
                        }
                        Err(_) => println!("Invalid frequency: {}", parts[2]),
                    }
                }

                "lowpass" => {
                    if parts.len() != 3 {
                        println!("Usage: process lowpass <frequency_hz>");
                        return;
                    }

                    match parts[2].parse::<f64>() {
                        Ok(freq_hz) => {
                            if let Some(info) = state.processor.audio_info() {
                                let sample_rate = info.sample_rate as f64;
                                let freq_bin =
                                    (freq_hz * state.processor.config().window_size as f64
                                        / sample_rate) as usize;

                                match state.processor.process_all_frames(|_ch, _frame, frame| {
                                    for i in freq_bin.saturating_add(1)..frame.spectrum.len() {
                                        frame.spectrum[i] = Complex64::new(0.0, 0.0);
                                    }
                                }) {
                                    Ok(_) => println!(
                                        "Applied lowpass filter at {}",
                                        utils::format_frequency(freq_hz)
                                    ),
                                    Err(e) => println!("Error processing: {}", e),
                                }
                            } else {
                                println!("No audio information available");
                            }
                        }
                        Err(_) => println!("Invalid frequency: {}", parts[2]),
                    }
                }

                _ => {
                    println!("Unknown operation: {}", operation);
                    println!("Available operations: zero_dc, highpass <freq>, lowpass <freq>");
                }
            }
        }

        "temporal_process" => {
            if parts.len() < 2 {
                println!("Usage: temporal_process <operation> [parameters...]");
                println!("Available operations:");
                print_temporal_processing_operations();
                return;
            }

            if !state.processor.has_temporal_analysis() {
                println!("No temporal FFT analysis available. Run temporal_analyze first.");
                return;
            }

            let operation = parts[1];
            match operation {
                "zero_temporal_dc" => {
                    match state.processor.process_temporal_fft(|temporal_analysis| {
                        temporal_analysis.zero_temporal_dc();
                    }) {
                        Ok(_) => println!("Temporal DC component zeroed out"),
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "temporal_highpass" => {
                    if parts.len() != 3 {
                        println!("Usage: temporal_process temporal_highpass <cutoff>");
                        println!("Cutoff should be between 0.0 and 1.0 (normalized frequency)");
                        return;
                    }

                    match parts[2].parse::<f64>() {
                        Ok(cutoff) => {
                            if !(0.0..=1.0).contains(&cutoff) {
                                println!("Cutoff must be between 0.0 and 1.0");
                                return;
                            }

                            match state.processor.process_temporal_fft(|temporal_analysis| {
                                temporal_analysis.temporal_highpass(cutoff);
                            }) {
                                Ok(_) => {
                                    println!("Applied temporal highpass filter at {:.3}", cutoff)
                                }
                                Err(e) => println!("Error processing: {}", e),
                            }
                        }
                        Err(_) => println!("Invalid cutoff frequency: {}", parts[2]),
                    }
                }

                "temporal_lowpass" => {
                    if parts.len() != 3 {
                        println!("Usage: temporal_process temporal_lowpass <cutoff>");
                        println!("Cutoff should be between 0.0 and 1.0 (normalized frequency)");
                        return;
                    }

                    match parts[2].parse::<f64>() {
                        Ok(cutoff) => {
                            if !(0.0..=1.0).contains(&cutoff) {
                                println!("Cutoff must be between 0.0 and 1.0");
                                return;
                            }

                            match state.processor.process_temporal_fft(|temporal_analysis| {
                                temporal_analysis.temporal_lowpass(cutoff);
                            }) {
                                Ok(_) => {
                                    println!("Applied temporal lowpass filter at {:.3}", cutoff)
                                }
                                Err(e) => println!("Error processing: {}", e),
                            }
                        }
                        Err(_) => println!("Invalid cutoff frequency: {}", parts[2]),
                    }
                }

                "temporal_shift" => {
                    if parts.len() < 3 {
                        println!("Usage: temporal_process temporal_shift <shift_frames>");
                        println!("  shift_frames: Number of frames to shift (positive = delay, negative = advance)");
                        return;
                    }

                    let shift_frames = match parts[2].parse::<i32>() {
                        Ok(v) => v,
                        Err(_) => {
                            println!("Error: shift_frames must be an integer");
                            return;
                        }
                    };

                    match state.processor.process_temporal_fft(|temporal_analysis| {
                        temporal_analysis.shift(shift_frames);
                    }) {
                        Ok(_) => {
                            println!(
                                "Applied temporal shift of {} frames to all channels",
                                shift_frames
                            );
                            if let Some(temporal) = state.processor.temporal_fft() {
                                println!(
                                    "{}",
                                    temporal.describe_temporal_shift_effects(shift_frames)
                                );
                            }
                        }
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "temporal_shift_dispersive" => {
                    if parts.len() < 4 {
                        println!("Usage: temporal_process temporal_shift_dispersive <base_shift> <freq_factor>");
                        println!("  base_shift: Base shift amount in frames");
                        println!("  freq_factor: How much shift varies with frequency");
                        println!("                0.0 = no variation");
                        println!("                1.0 = high frequencies shifted 2x base amount");
                        println!(
                            "               -1.0 = high frequencies shifted in opposite direction"
                        );
                        return;
                    }

                    let base_shift = match parts[2].parse::<i32>() {
                        Ok(v) => v,
                        Err(_) => {
                            println!("Error: base_shift must be an integer");
                            return;
                        }
                    };

                    let freq_factor = match parts[3].parse::<f64>() {
                        Ok(v) => v,
                        Err(_) => {
                            println!("Error: freq_factor must be a floating-point number");
                            return;
                        }
                    };

                    match state.processor.process_temporal_fft(|temporal_analysis| {
                        temporal_analysis.shift_dispersive(base_shift, freq_factor);
                    }) {
                        Ok(_) => {
                            println!(
                                "Applied dispersive temporal shift: base={}, factor={}",
                                base_shift, freq_factor
                            );
                            println!("Effect: Lower frequencies shifted by {} frames", base_shift);
                            println!(
                                "        Higher frequencies shifted by up to {} frames",
                                (base_shift as f64 * (1.0 + freq_factor)) as i32
                            );
                        }
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "temporal_shift_circular" => {
                    if parts.len() < 3 {
                        println!("Usage: temporal_process temporal_shift_circular <shift_frames>");
                        println!("  shift_frames: Number of frames to shift (with wrap-around)");
                        return;
                    }

                    let shift_frames = match parts[2].parse::<i32>() {
                        Ok(v) => v,
                        Err(_) => {
                            println!("Error: shift_frames must be an integer");
                            return;
                        }
                    };

                    match state.processor.process_temporal_fft(|temporal_analysis| {
                        temporal_analysis.shift_circular(shift_frames);
                    }) {
                        Ok(_) => {
                            println!("Applied circular temporal shift of {} frames", shift_frames);
                            if let Some(temporal) = state.processor.temporal_fft() {
                                println!(
                                    "{}",
                                    temporal.describe_temporal_shift_effects(shift_frames)
                                );
                            }
                        }
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "temporal_stretch" => {
                    if parts.len() < 3 {
                        println!("Usage: temporal_process temporal_stretch <stretch_factor>");
                        println!(
                            "  stretch_factor: Factor by which to stretch temporal frequencies"
                        );
                        println!("                  1.0 = no change");
                        println!(
                            "                  < 1.0 = time stretch"
                        );
                        println!("                  > 1.0 = time compress");
                        println!();
                        return;
                    }

                    let stretch_factor = match parts[2].parse::<f64>() {
                        Ok(v) => {
                            if v <= 0.0 {
                                println!("Error: stretch_factor must be positive, got {}", v);
                                return;
                            }
                            v
                        }
                        Err(_) => {
                            println!("Error: stretch_factor must be a positive number");
                            return;
                        }
                    };

                    match state.processor.process_temporal_fft(|temporal_analysis| {
                        temporal_analysis.stretch(stretch_factor);
                    }) {
                        Ok(_) => {
                            println!(
                                "Applied temporal stretch factor {} to all channels",
                                stretch_factor
                            );
                            if let Some(temporal) = state.processor.temporal_fft() {
                                println!(
                                    "{}",
                                    temporal.describe_temporal_stretch_effects(stretch_factor)
                                );
                            }
                        }
                        Err(e) => println!("Error processing: {}", e),
                    }
                }

                "temporal_phase_multiply" => {
                    if parts.len() < 3 {
                        eprintln!("Usage: temporal_process temporal_phase_multiply <factor>");
                        eprintln!("  factor: Phase multiplication factor");
                        return;
                    }

                    let factor: f64 = match parts[2].parse() {
                        Ok(f) => f,
                        Err(_) => {
                            eprintln!("Error: Invalid factor '{}'. Must be a number.", parts[2]);
                            return;
                        }
                    };

                    // Validate parameters
                    if let Err(e) = state.processor.validate_phase_multiply_params(factor) {
                        eprintln!("Error: {}", e);
                        return;
                    }

                    match state.processor.apply_temporal_phase_multiply(factor) {
                        Ok(_) => {
                            println!("Applied temporal phase multiply with factor {}", factor);
                        }
                        Err(e) => eprintln!("Error applying temporal phase multiply: {}", e),
                    }
                }

                "temporal_phase_multiply_dispersive" => {
                    if parts.len() < 4 {
                        eprintln!("Usage: temporal_process temporal_phase_multiply_dispersive <base_factor> <freq_scaling>");
                        eprintln!("  base_factor: Base phase multiplication factor");
                        eprintln!("  freq_scaling: Frequency-dependent scaling (-1.0 to 1.0)");
                        return;
                    }

                    let base_factor: f64 = match parts[2].parse() {
                        Ok(f) => f,
                        Err(_) => {
                            eprintln!(
                                "Error: Invalid base factor '{}'. Must be a number.",
                                parts[2]
                            );
                            return;
                        }
                    };

                    let freq_scaling: f64 = match parts[3].parse() {
                        Ok(f) => f,
                        Err(_) => {
                            eprintln!(
                                "Error: Invalid frequency scaling '{}'. Must be a number.",
                                parts[3]
                            );
                            return;
                        }
                    };

                    // Validate base factor
                    if let Err(e) = state.processor.validate_phase_multiply_params(base_factor) {
                        eprintln!("Error with base factor: {}", e);
                        return;
                    }

                    match state
                        .processor
                        .apply_dispersive_temporal_phase_multiply(base_factor, freq_scaling)
                    {
                        Ok(_) => {
                            println!("Applied dispersive temporal phase multiply with base factor {} and frequency scaling {}",
                                     base_factor, freq_scaling);
                        }
                        Err(e) => {
                            eprintln!("Error applying dispersive temporal phase multiply: {}", e)
                        }
                    }
                }

                "temporal_phase_reversal" => {
                    if parts.len() > 2 {
                        eprintln!("Usage: temporal_process temporal_phase_reversal");
                        eprintln!("  Applies phase reversal (factor = -1.0) to create time-reversal-like effects");
                        return;
                    }

                    match state.processor.apply_temporal_phase_reversal() {
                        Ok(_) => {
                            println!("Applied temporal phase reversal (time-reversal-like effect)");
                        }
                        Err(e) => eprintln!("Error applying temporal phase reversal: {}", e),
                    }
                }

                "temporal_phase_scrambling" => {
                    if parts.len() < 4 {
                        eprintln!(
                            "Usage: temporal_process temporal_phase_scrambling <intensity> <seed>"
                        );
                        eprintln!("  intensity: Scrambling intensity (0.0 = none, 1.0 = moderate, 2.0 = strong)");
                        eprintln!("  seed: Random seed for reproducible results");
                        return;
                    }

                    let intensity: f64 = match parts[2].parse() {
                        Ok(i) => i,
                        Err(_) => {
                            eprintln!("Error: Invalid intensity '{}'. Must be a number.", parts[2]);
                            return;
                        }
                    };

                    let seed: u64 = match parts[3].parse() {
                        Ok(s) => s,
                        Err(_) => {
                            eprintln!("Error: Invalid seed '{}'. Must be a number.", parts[3]);
                            return;
                        }
                    };

                    if intensity < 0.0 {
                        eprintln!("Warning: Negative intensity may produce unexpected results");
                    }

                    match state
                        .processor
                        .apply_temporal_phase_scrambling(intensity, seed)
                    {
                        Ok(_) => {
                            println!(
                                "Applied temporal phase scrambling with intensity {} and seed {}",
                                intensity, seed
                            );
                        }
                        Err(e) => eprintln!("Error applying temporal phase scrambling: {}", e),
                    }
                }

                "temporal_phase_stats" => {
                    if parts.len() > 2 {
                        eprintln!("Usage: temporal_process temporal_phase_stats");
                        eprintln!("  Shows statistical information about phase distribution in temporal FFT data");
                        return;
                    }

                    match state.processor.get_temporal_phase_statistics() {
                        Ok(stats) => {
                            println!("Temporal FFT Phase Statistics:");
                            println!("  {}", stats);
                            println!(
                                "  Phase range: {:.1}° to {:.1}°",
                                stats.min_phase * 180.0 / std::f64::consts::PI,
                                stats.max_phase * 180.0 / std::f64::consts::PI
                            );
                            println!(
                                "  Standard deviation: {:.3}rad ({:.1}°)",
                                stats.phase_variance.sqrt(),
                                stats.phase_variance.sqrt() * 180.0 / std::f64::consts::PI
                            );
                        }
                        Err(e) => eprintln!("Error getting temporal phase statistics: {}", e),
                    }
                }

                "temporal_blur" => {
                    if parts.len() < 4 {
                        println!("Usage: temporal_process temporal_blur <radius> <strength>");
                        println!("  radius: Number of adjacent bins to average (e.g., 1, 2, 3)");
                        println!("  strength: Blur strength (0.0 = none, 1.0 = full)");
                        return;
                    }

                    let radius = match parts[2].parse::<usize>() {
                        Ok(r) => r,
                        Err(_) => {
                            println!("Error: radius must be a non-negative integer");
                            return;
                        }
                    };

                    let strength = match parts[3].parse::<f64>() {
                        Ok(s) => {
                            if s < 0.0 || s > 1.0 {
                                println!("Error: strength must be between 0.0 and 1.0");
                                return;
                            }
                            s
                        }
                        Err(_) => {
                            println!("Error: strength must be a number between 0.0 and 1.0");
                            return;
                        }
                    };

                    match state.processor.apply_temporal_blur(radius, strength) {
                        Ok(_) => {
                            println!(
                                "Applied temporal blur with radius {} and strength {}",
                                radius, strength
                            );
                        }
                        Err(e) => println!("Error applying temporal blur: {}", e),
                    }
                }

                "temporal_convolve" => {
                    if parts.len() < 3 {
                        println!("Usage: temporal_process temporal_convolve <impulse_file>");
                        println!("  impulse_file: Audio file to use as convolution impulse");
                        println!();
                        println!("Description:");
                        println!("  Convolves the temporal FFT with an impulse response.");
                        println!(
                            "  The impulse file will be analyzed with the same STFT settings."
                        );
                        println!("  If the impulse is shorter, it will be padded with silence.");
                        println!("  If the impulse is longer, it will be truncated.");
                        return;
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let impulse_file = parts[2];

                        // Check if file exists
                        if !std::path::Path::new(impulse_file).exists() {
                            println!("Error: File '{}' not found", impulse_file);
                            return;
                        }

                        println!(
                            "Applying temporal convolution with impulse: {}",
                            impulse_file
                        );
                        match state.processor.apply_temporal_convolve(impulse_file) {
                            Ok(_) => {
                                println!("Temporal convolution complete!");
                                println!("Note: The result is stored in the temporal FFT data.");
                                println!("Use 'temporal_synthesize' to convert back to audio.");
                            }
                            Err(e) => println!("Error applying temporal convolution: {}", e),
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        println!("Error: File operations are not supported in WebAssembly");
                    }
                }

                "temporal_power_amplitude" => {
                    if parts.len() < 3 {
                        println!("Usage: temporal_process temporal_power_amplitude <factor>");
                        println!("  factor: Power to apply to normalized amplitudes");
                        println!("          < 1.0 = compress temporal amplitudes");
                        println!("          = 1.0 = no change");
                        println!("          > 1.0 = expand temporal amplitudes");
                        return;
                    }

                    let factor = match parts[2].parse::<f64>() {
                        Ok(f) => {
                            if f <= 0.0 {
                                println!("Error: factor must be positive, got {}", f);
                                return;
                            }
                            f
                        }
                        Err(_) => {
                            println!("Error: factor must be a positive number");
                            return;
                        }
                    };

                    match state.processor.apply_temporal_power_amplitude(factor) {
                        Ok(_) => {
                            let effect = if factor < 1.0 {
                                "Dynamic range expanded"
                            } else if factor > 1.0 {
                                "Dynamic range compressed"
                            } else {
                                "No change (factor = 1.0)"
                            };

                            println!("Applied temporal power amplitude with factor {}", factor);
                            println!("{}", effect);
                        }
                        Err(e) => println!("Error applying temporal power amplitude: {}", e),
                    }
                }

                "temporal_power_amplitude_per_bin" => {
                    if parts.len() < 3 {
                        println!(
                            "Usage: temporal_process temporal_power_amplitude_per_bin <factor>"
                        );
                        println!(
                            "  factor: Power to apply to normalized amplitudes per frequency bin"
                        );
                        println!("          < 1.0 = compress temporal amplitudes");
                        println!("          = 1.0 = no change");
                        println!("          > 1.0 = expand temporal amplitudes");
                        println!();
                        println!("Note: Unlike the global version, this preserves relative amplitudes between bins");
                        return;
                    }

                    let factor = match parts[2].parse::<f64>() {
                        Ok(f) => {
                            if f <= 0.0 {
                                println!("Error: factor must be positive, got {}", f);
                                return;
                            }
                            f
                        }
                        Err(_) => {
                            println!("Error: factor must be a positive number");
                            return;
                        }
                    };

                    match state
                        .processor
                        .apply_temporal_power_amplitude_per_bin(factor)
                    {
                        Ok(_) => {
                            let effect = if factor < 1.0 {
                                "Dynamic range expanded within each frequency bin"
                            } else if factor > 1.0 {
                                "Dynamic range compressed within each frequency bin"
                            } else {
                                "No change (factor = 1.0)"
                            };

                            println!(
                                "Applied temporal power amplitude per-bin with factor {}",
                                factor
                            );
                            println!("{}", effect);
                            println!("Note: Relative amplitudes between frequency bins preserved");
                        }
                        Err(e) => {
                            println!("Error applying temporal power amplitude per-bin: {}", e)
                        }
                    }
                }

                _ => {
                    println!("Unknown temporal operation: {}", operation);
                    println!("Available operations:");
                    println!("  zero_temporal_dc                 - Zero the DC component");
                    println!("  temporal_highpass <cutoff>       - Apply temporal highpass filter");
                    println!("  temporal_lowpass <cutoff>        - Apply temporal lowpass filter");
                    println!("  temporal_shift <frames>          - Shift temporal evolution");
                    println!(
                        "  temporal_shift_dispersive <base> <factor> - Frequency-dependent shift"
                    );
                    println!("  temporal_shift_circular <frames> - Circular shift (wrap-around)");
                    println!(
                        "  temporal_stretch <factor>        - Stretch temporal frequency spectrum"
                    );
                    println!("  temporal_phase_multiply <factor> - Multiply all phases by factor");
                    println!("  temporal_phase_multiply_dispersive <base> <scaling> - Frequency-dependent phase multiply");
                    println!(
                        "  temporal_phase_reversal          - Apply phase reversal (-1.0 factor)"
                    );
                    println!("  temporal_phase_scrambling <intensity> <seed> - Pseudo-random phase scrambling");
                    println!(
                        "  temporal_phase_stats             - Show phase distribution statistics"
                    );
                }
            }
        }

        "spectrogram" => {
            #[cfg(feature = "image")]
            {
                if parts.len() < 2 {
                    println!("Usage: spectrogram <filename> [channel] [colormap] [width] [height] [scale]");
                    println!("  channel: 0, 1, ... or 'all' (default: 0)");
                    println!("  colormap: viridis, plasma, inferno, magma, turbo, grayscale, jet (default: viridis)");
                    println!("  width: image width in pixels (default: 800)");
                    println!("  height: image height in pixels (default: 600)");
                    println!("  scale: linear or log (default: linear)");
                    return;
                }

                if !state.processor.has_analysis() {
                    println!("No STFT analysis available. Load and analyze a file first.");
                    return;
                }

                let filename = parts[1];
                let channel_spec = parts.get(2).unwrap_or(&"0");
                let colormap_name = parts.get(3).unwrap_or(&"viridis");
                let width = parts
                    .get(4)
                    .and_then(|w| w.parse::<u32>().ok())
                    .unwrap_or(800);
                let height = parts
                    .get(5)
                    .and_then(|h| h.parse::<u32>().ok())
                    .unwrap_or(600);
                let scale = parts.get(6).unwrap_or(&"linear");

                // Parse colormap
                let colormap = match colormap_name.to_lowercase().as_str() {
                    "viridis" => ColorMap::Viridis,
                    "plasma" => ColorMap::Plasma,
                    "inferno" => ColorMap::Inferno,
                    "magma" => ColorMap::Magma,
                    "turbo" => ColorMap::Turbo,
                    "grayscale" | "gray" => ColorMap::Grayscale,
                    "jet" => ColorMap::Jet,
                    _ => {
                        println!("Unknown colormap: {}. Using viridis.", colormap_name);
                        ColorMap::Viridis
                    }
                };

                // Parse frequency scale
                let frequency_scale = match scale.to_lowercase().as_str() {
                    "log" | "logarithmic" => {
                        stft_audio_lib::spectrogram::FrequencyScale::Logarithmic
                    }
                    "linear" => stft_audio_lib::spectrogram::FrequencyScale::Linear,
                    _ => {
                        println!("Unknown scale: {}. Using linear.", scale);
                        stft_audio_lib::spectrogram::FrequencyScale::Linear
                    }
                };

                let frames = state.processor.stft_frames().unwrap();
                let audio_info = state.processor.audio_info();
                let config = state.processor.config();

                // Determine which channels to process
                let channels_to_process: Vec<usize> = if *channel_spec == "all" {
                    (0..frames.len()).collect()
                } else {
                    match channel_spec.parse::<usize>() {
                        Ok(ch) if ch < frames.len() => vec![ch],
                        _ => {
                            println!("Invalid channel: {}. Using channel 0.", channel_spec);
                            vec![0]
                        }
                    }
                };

                for &channel_idx in &channels_to_process {
                    // Create spectrogram from STFT frames
                    let spectrogram = match Spectrogram::from_stft_frames(
                        &frames[channel_idx],
                        audio_info.map(|info| info.sample_rate),
                        Some(config.hop_size()),
                    ) {
                        Ok(spec) => spec,
                        Err(e) => {
                            println!("Error creating spectrogram: {}", e);
                            return;
                        }
                    };

                    // Configure image options
                    let mut options = SpectrogramImageOptions::default();
                    options.width = width;
                    options.height = height;
                    options.colormap = colormap;
                    options.use_db_scale = true;
                    options.dynamic_range_db = 80.0;
                    options.frequency_scale = frequency_scale;

                    // Generate filename with channel suffix if processing multiple channels
                    let output_filename = if channels_to_process.len() > 1 {
                        let path = std::path::Path::new(filename);
                        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                        let ext = path.extension().unwrap_or_default().to_string_lossy();
                        format!("{}_ch{}.{}", stem, channel_idx, ext)
                    } else {
                        filename.to_string()
                    };

                    // Save the spectrogram
                    match save_spectrogram(&spectrogram, &output_filename, &options) {
                        Ok(_) => {
                            println!(
                                "Saved spectrogram for channel {} to: {}",
                                channel_idx, output_filename
                            );
                            if let Some(info) = audio_info {
                                let time_axis = spectrogram.time_axis();
                                let duration = time_axis.last().copied().unwrap_or(0.0);
                                let max_freq = info.sample_rate as f64 / 2.0;
                                println!("  Time range: 0.0 - {:.2} seconds", duration);
                                println!(
                                    "  Frequency range: 0 - {}",
                                    utils::format_frequency(max_freq)
                                );
                                println!("  Image size: {}x{} pixels", width, height);
                                println!("  Color map: {:?}", colormap);
                                println!("  Frequency scale: {:?}", frequency_scale);
                            }
                        }
                        Err(e) => println!("Error saving spectrogram: {}", e),
                    }
                }
            }
            #[cfg(not(feature = "image"))]
            {
                println!("Spectrogram generation requires the 'image' feature to be enabled.");
                println!("Rebuild with: cargo build --release --features image");
            }
        }

        "temporal_spectrogram" => {
            #[cfg(feature = "image")]
            {
                if parts.len() < 2 {
                    println!("Usage: temporal_spectrogram <filename> [channel] [colormap]");
                    println!("  channel: 0, 1, ... or 'all' (default: 0)");
                    println!("  colormap: viridis, plasma, inferno, magma, turbo, grayscale, jet (default: viridis)");
                    println!();
                    println!("Creates a visualization of the temporal FFT analysis:");
                    println!("  Y-axis: Frequency bins (0 = DC at top)");
                    println!("  X-axis: Temporal FFT bins");
                    println!("  Color: Magnitude of temporal frequency components");
                    return;
                }

                if !state.processor.has_temporal_analysis() {
                    println!("No temporal FFT analysis available. Run temporal_analyze first.");
                    return;
                }

                let filename = parts[1];
                let channel_spec = parts.get(2).unwrap_or(&"0");
                let colormap_name = parts.get(3).unwrap_or(&"viridis");

                // Parse colormap
                let colormap = match colormap_name.to_lowercase().as_str() {
                    "viridis" => ColorMap::Viridis,
                    "plasma" => ColorMap::Plasma,
                    "inferno" => ColorMap::Inferno,
                    "magma" => ColorMap::Magma,
                    "turbo" => ColorMap::Turbo,
                    "grayscale" | "gray" => ColorMap::Grayscale,
                    "jet" => ColorMap::Jet,
                    _ => {
                        println!("Unknown colormap: {}. Using viridis.", colormap_name);
                        ColorMap::Viridis
                    }
                };

                let temporal_analysis = state.processor.temporal_fft().unwrap();

                // Determine which channels to process
                let channels_to_process: Vec<usize> = if *channel_spec == "all" {
                    (0..temporal_analysis.num_channels).collect()
                } else {
                    match channel_spec.parse::<usize>() {
                        Ok(ch) if ch < temporal_analysis.num_channels => vec![ch],
                        _ => {
                            println!("Invalid channel: {}. Using channel 0.", channel_spec);
                            vec![0]
                        }
                    }
                };

                for &channel_idx in &channels_to_process {
                    // Configure options
                    let options =
                        stft_audio_lib::temporal_fft::visualization::TemporalFFTImageOptions {
                            use_db_scale: false,
                            dynamic_range_db: 80.0,
                            db_reference: 1.0,
                            colormap,
                        };

                    // Generate filename with channel suffix if processing multiple channels
                    let output_filename = if channels_to_process.len() > 1 {
                        let path = std::path::Path::new(filename);
                        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                        let ext = path.extension().unwrap_or_default().to_string_lossy();
                        format!("{}_ch{}.{}", stem, channel_idx, ext)
                    } else {
                        filename.to_string()
                    };

                    // Save the temporal FFT visualization
                    match stft_audio_lib::temporal_fft::visualization::save_temporal_fft_image(
                        temporal_analysis,
                        channel_idx,
                        &output_filename,
                        &options,
                    ) {
                        Ok(_) => {
                            println!(
                                "Saved temporal FFT visualization for channel {} to: {}",
                                channel_idx, output_filename
                            );
                            println!(
                                "  Image dimensions: {}x{} pixels",
                                temporal_analysis.num_temporal_bins(),
                                temporal_analysis.num_frequency_bins
                            );
                            println!(
                                "  Y-axis: {} frequency bins",
                                temporal_analysis.num_frequency_bins
                            );
                            println!(
                                "  X-axis: {} temporal FFT bins",
                                temporal_analysis.num_temporal_bins()
                            );
                            println!("  Color map: {:?}", colormap);
                            println!("  Scale: dB (dynamic range: 80 dB)");
                        }
                        Err(e) => println!("Error saving temporal FFT visualization: {}", e),
                    }
                }
            }
            #[cfg(not(feature = "image"))]
            {
                println!("Temporal FFT visualization requires the 'image' feature to be enabled.");
                println!("Rebuild with: cargo build --release --features image");
            }
        }

        "status" => {
            println!("Processor Status:");
            println!(
                "  Audio loaded: {}",
                if state.processor.has_audio() {
                    "Yes"
                } else {
                    "No"
                }
            );
            println!(
                "  STFT analysis: {}",
                if state.processor.has_analysis() {
                    "Yes"
                } else {
                    "No"
                }
            );
            println!(
                "  Temporal analysis: {}",
                if state.processor.has_temporal_analysis() {
                    "Yes"
                } else {
                    "No"
                }
            );

            if let Some(filename) = &state.current_file {
                println!("  Current file: {}", filename);
            }

            if let Some(frames) = state.processor.num_frames() {
                println!("  STFT frames per channel: {}", frames);
            }

            if let Some(temporal) = state.processor.temporal_fft() {
                println!("  Temporal bins: {}", temporal.num_temporal_bins());
                println!("  Frequency bins: {}", temporal.num_frequency_bins);
            }
        }

        "help" => print_help(),

        "quit" | "exit" => {
            println!("Goodbye!");
            process::exit(0);
        }

        _ => {
            println!("Unknown command: '{}'", parts[0]);
            println!("Type 'help' for available commands");
        }
    }
}

fn main() {
    // Initialize environment logger
    //env_logger::init();

    // Parse command line arguments
    let matches = Command::new("STFT Audio Processor")
        .version(stft_audio_lib::VERSION)
        .about("Short-Time Fourier Transform audio processing tool")
        .arg(
            Arg::new("file")
                .help("Audio file to load on startup")
                .value_name("FILE")
                .index(1),
        )
        .arg(
            Arg::new("window-size")
                .long("window-size")
                .short('w')
                .help("Set window size (256, 512, 1024, 2048, 4096)")
                .value_name("SIZE"),
        )
        .arg(
            Arg::new("overlap")
                .long("overlap")
                .short('o')
                .help("Set overlap factor (1-32)")
                .value_name("FACTOR"),
        )
        .arg(
            Arg::new("window-type")
                .long("window-type")
                .short('t')
                .help("Set window type (hanning, hamming, rectangular, bartlett)")
                .value_name("TYPE"),
        )
        .get_matches();

    println!("STFT Audio Processor v{}", stft_audio_lib::VERSION);
    println!("Type 'help' for available commands\n");

    // Initialize the library
    stft_audio_lib::init();

    let mut state = AppState::new();

    // Apply command line configuration
    let mut config_changed = false;
    let mut current_config = state.processor.config().clone();

    if let Some(window_size_str) = matches.get_one::<String>("window-size") {
        if let Ok(size) = window_size_str.parse::<usize>() {
            if let Ok(new_config) = STFTConfig::new(
                size,
                current_config.overlap_factor,
                current_config.window_type,
            ) {
                current_config = new_config;
                config_changed = true;
                println!("Set window size to {}", size);
            } else {
                eprintln!("Invalid window size: {}", size);
            }
        }
    }

    if let Some(overlap_str) = matches.get_one::<String>("overlap") {
        if let Ok(overlap) = overlap_str.parse::<usize>() {
            if let Ok(new_config) = STFTConfig::new(
                current_config.window_size,
                overlap,
                current_config.window_type,
            ) {
                current_config = new_config;
                config_changed = true;
                println!("Set overlap factor to {}", overlap);
            } else {
                eprintln!("Invalid overlap factor: {}", overlap);
            }
        }
    }

    if let Some(window_type_str) = matches.get_one::<String>("window-type") {
        let window_type = match window_type_str.as_str() {
            "hanning" => Some(WindowType::Hanning),
            "hamming" => Some(WindowType::Hamming),
            "rectangular" => Some(WindowType::Rectangular),
            "bartlett" => Some(WindowType::Bartlett),
            _ => {
                eprintln!("Invalid window type: {}", window_type_str);
                None
            }
        };

        if let Some(wt) = window_type {
            if let Ok(new_config) = STFTConfig::new(
                current_config.window_size,
                current_config.overlap_factor,
                wt,
            ) {
                current_config = new_config;
                config_changed = true;
                println!("Set window type to {:?}", wt);
            }
        }
    }

    if config_changed {
        state.processor.set_config(current_config);
    }

    // Load file from command line if provided
    if let Some(filename) = matches.get_one::<String>("file") {
        println!("Loading file: {}", filename);
        match utils::load_and_analyze(&mut state.processor, filename) {
            Ok(_) => {
                state.current_file = Some(filename.to_string());
                println!("File loaded successfully!");
                println!("{}", utils::analysis_summary(&state.processor));
            }
            Err(e) => eprintln!("Error loading file: {}", e),
        }
    }

    // Setup readline
    let mut rl = DefaultEditor::new().expect("Failed to create readline");

    // Main command loop
    loop {
        let readline = rl.readline("stft> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    rl.add_history_entry(trimmed).ok();
                    process_command(trimmed, &mut state);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    println!("Goodbye!");
}
