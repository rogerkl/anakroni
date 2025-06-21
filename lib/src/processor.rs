//! Main STFT processor implementation
//!
//! Provides the main STFTProcessor struct that coordinates STFT analysis,
//! processing, and synthesis for multi-channel audio data.

use crate::audio_io::AudioInfo;
use crate::stft::{STFTAnalyzer, STFTConfig, STFTFrame, STFTSynthesizer};
use crate::temporal_fft::{
    TemporalBinFFT, TemporalFFTAnalysis, TemporalFFTAnalyzer, TemporalFFTConfig,
    TemporalFFTSynthesizer, TemporalOperation, TemporalStatistics,
};
use crate::utils::{distribute_grouped, distribute_lin_bins, distribute_log_bins};
use crate::Result;
use num_complex::{Complex, Complex64};

/// Main STFT processor for multi-channel audio
pub struct STFTProcessor {
    config: STFTConfig,
    audio_info: Option<AudioInfo>,
    channel_data: Option<Vec<Vec<f64>>>,
    stft_frames: Option<Vec<Vec<STFTFrame>>>, // frames for each channel
    temporal_fft: Option<TemporalFFTAnalysis>, // temporal FFT analysis
    temporal_config: TemporalFFTConfig,
}

impl STFTProcessor {
    /// Create a new STFT processor with default configuration
    pub fn new() -> Self {
        Self {
            config: STFTConfig::default(),
            audio_info: None,
            channel_data: None,
            stft_frames: None,
            temporal_fft: None,
            temporal_config: TemporalFFTConfig::default(),
        }
    }

    /// Create a new STFT processor with custom configuration
    pub fn with_config(config: STFTConfig) -> Self {
        Self {
            config,
            audio_info: None,
            channel_data: None,
            stft_frames: None,
            temporal_fft: None,
            temporal_config: TemporalFFTConfig::default(),
        }
    }

    fn from_processor(processor: &STFTProcessor) -> Self {
        let config = processor.config.clone();
        let audio_info = processor.audio_info.clone();
        let temporal_config = processor.temporal_config.clone();
        Self {
            config,
            audio_info,
            channel_data: None,
            stft_frames: None,
            temporal_fft: None,
            temporal_config,
        }
    }

    pub fn prepare_split_part(
        &mut self,
        part_index: usize,
        num_parts: usize,
        group_size: usize,
        log: bool,
        octaves: u8,
    ) -> Result<STFTProcessor> {
        log::info!("prepare_split_part  part_index:{} num_parts:{} group_size:{} log:{}",part_index,num_parts,group_size, log);
        let config = self.config.clone();
        let audio_info = self.audio_info.clone();
        let temporal_config = self.temporal_config.clone();

        let bin_assignments = if group_size == 0 {
            if log {
                distribute_log_bins(temporal_config.fft_size / 2, num_parts, true, octaves)
            } else {
                distribute_lin_bins(temporal_config.fft_size / 2, num_parts, true, octaves)
            }
        } else {
            distribute_grouped(
                temporal_config.fft_size / 2,
                num_parts,
                true,
                octaves,
                group_size,
                if log {
                    &distribute_log_bins
                } else {
                    &distribute_lin_bins
                },
            )
        };

        match &self.temporal_fft {
            None => {
                return Err("No temporal_fft in data".to_string());
            }
            Some(temporal_fft) => {
                let original_frames = temporal_fft.bin_ffts[0].original_frames;
                let mut bin_ffts =
                    Vec::with_capacity(temporal_fft.num_channels * temporal_fft.num_frequency_bins);

                let mut bin_ffts_index = 0;

                // Process each channel separately
                for ch_idx in 0..temporal_fft.num_channels {
                    log::info!(
                        "Preparing temporal FFT split for channel {}/{}",
                        ch_idx + 1,
                        temporal_fft.num_channels
                    );

                    // Process each frequency bin in this channel
                    for bin_idx in 0..temporal_fft.num_frequency_bins {
                        log::debug!(
                            "Channel {}, preparing temporal FFT split for bin {}/{}",
                            ch_idx,
                            bin_idx + 1,
                            temporal_fft.num_frequency_bins
                        );
                        let mut temporal_spectrum = Vec::with_capacity(temporal_config.fft_size);

                        for bin in 0..bin_assignments.len() {
                            if bin_assignments[bin] == part_index {
                                temporal_spectrum.push(
                                    temporal_fft.bin_ffts[bin_ffts_index].temporal_spectrum[bin],
                                );
                            } else {
                                temporal_spectrum.push(Complex64::new(0.0, 0.0));
                            }
                        }

                        bin_ffts.push(TemporalBinFFT {
                            bin_index: bin_idx,
                            channel_index: ch_idx,
                            temporal_spectrum,
                            original_frames:temporal_config.fft_size,
                        });
                        bin_ffts_index += 1;
                    }
                }

                let temporal_fft_split = TemporalFFTAnalysis {
                    bin_ffts,
                    config: temporal_config,
                    num_frequency_bins: temporal_fft.num_frequency_bins,
                    num_channels: temporal_fft.num_channels,
                };
                let split_processor = Self {
                    config,
                    audio_info,
                    channel_data: None,
                    stft_frames: None,
                    temporal_fft: Some(temporal_fft_split),
                    temporal_config,
                };
                Ok(split_processor)
            }
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &STFTConfig {
        &self.config
    }

    /// Set a new configuration (clears any existing analysis)
    pub fn set_config(&mut self, config: STFTConfig) {
        self.config = config;
        self.stft_frames = None; // Clear analysis data
        self.temporal_fft = None; // Clear temporal analysis
    }

    /// Get the current temporal FFT configuration
    pub fn temporal_config(&self) -> &TemporalFFTConfig {
        &self.temporal_config
    }

    /// Set temporal FFT configuration
    pub fn set_temporal_config(&mut self, config: TemporalFFTConfig) {
        self.temporal_config = config;
        self.temporal_fft = None; // Clear existing temporal analysis
    }

    /// Load audio data into the processor
    pub fn load_audio(&mut self, audio_info: AudioInfo, channel_data: Vec<Vec<f64>>) -> Result<()> {
        if channel_data.len() != audio_info.channels {
            return Err(format!(
                "Channel count mismatch: expected {}, got {}",
                audio_info.channels,
                channel_data.len()
            ));
        }

        // Validate that all channels have the same length
        if let Some(first_channel) = channel_data.first() {
            let expected_length = first_channel.len();
            for (i, channel) in channel_data.iter().enumerate() {
                if channel.len() != expected_length {
                    return Err(format!(
                        "Channel {} has length {}, expected {}",
                        i,
                        channel.len(),
                        expected_length
                    ));
                }
            }

            if expected_length != audio_info.duration_samples {
                return Err(format!(
                    "Audio duration mismatch: expected {} samples, got {}",
                    audio_info.duration_samples, expected_length
                ));
            }
        }

        self.audio_info = Some(audio_info);
        self.channel_data = Some(channel_data);
        self.stft_frames = None; // Clear any existing analysis
        self.temporal_fft = None; // Clear temporal analysis

        Ok(())
    }

    /// Get audio information
    pub fn audio_info(&self) -> Option<&AudioInfo> {
        self.audio_info.as_ref()
    }

    /// Get reference to channel data
    pub fn channel_data(&self) -> Option<&Vec<Vec<f64>>> {
        self.channel_data.as_ref()
    }

    /// Get reference to STFT frames
    pub fn stft_frames(&self) -> Option<&Vec<Vec<STFTFrame>>> {
        self.stft_frames.as_ref()
    }

    /// Get reference to temporal FFT analysis
    pub fn temporal_fft(&self) -> Option<&TemporalFFTAnalysis> {
        self.temporal_fft.as_ref()
    }

    /// Perform STFT analysis on all channels
    pub fn analyze(&mut self) -> Result<()> {
        let channel_data = self.channel_data.as_ref().ok_or("No audio data loaded")?;

        let analyzer = STFTAnalyzer::new(self.config)?;
        let mut all_frames = Vec::with_capacity(channel_data.len());

        for (channel_idx, channel) in channel_data.iter().enumerate() {
            log::info!(
                "Analyzing channel {} ({} samples)",
                channel_idx,
                channel.len()
            );

            let frames = analyzer.analyze(channel)?;
            log::info!(
                "Generated {} STFT frames for channel {}",
                frames.len(),
                channel_idx
            );

            all_frames.push(frames);
        }

        self.stft_frames = Some(all_frames);
        Ok(())
    }

    /// Perform temporal FFT analysis on STFT frames with optional length multiplier
    pub fn analyze_temporal(&mut self) -> Result<()> {
        let stft_frames = self
            .stft_frames
            .as_ref()
            .ok_or("No STFT analysis available. Call analyze() first.")?;

        if stft_frames.is_empty() || stft_frames[0].is_empty() {
            return Err("No STFT frames available for temporal analysis".to_string());
        }

        let num_frames = stft_frames[0].len();
        let num_channels = stft_frames.len();

        // Auto-configure temporal FFT size if current config is too small
        let required_size = if self.temporal_config.repeat_data {
            num_frames * self.temporal_config.length_multiplier
        } else {
            // For zero padding, we need FFT size that accommodates the analysis resolution
            num_frames * self.temporal_config.length_multiplier
        };

        if self.temporal_config.fft_size < required_size {
            let auto_config = TemporalFFTConfig::auto_size_with_options(
                num_frames,
                self.temporal_config.length_multiplier,
                self.temporal_config.repeat_data,
            );
            log::info!(
                "Auto-sizing temporal FFT: {} required size -> {} FFT size (multiplier: {}x, mode: {})",
                required_size,
                auto_config.fft_size,
                auto_config.length_multiplier,
                if auto_config.repeat_data { "repeat" } else { "zero padding" }
            );
            self.temporal_config = auto_config;
        }

        log::info!(
            "Performing temporal FFT analysis: {} channels, {} time frames, FFT size {}, multiplier {}x, mode: {}",
            num_channels,
            num_frames,
            self.temporal_config.fft_size,
            self.temporal_config.length_multiplier,
            if self.temporal_config.repeat_data { "repeat" } else { "zero padding" }
        );

        let analyzer = TemporalFFTAnalyzer::new(self.temporal_config);
        let temporal_analysis = analyzer.analyze(stft_frames)?;

        log::info!(
            "Temporal FFT complete: {} channels, {} frequency bins per channel, {} temporal bins each",
            temporal_analysis.num_channels,
            temporal_analysis.num_frequency_bins,
            temporal_analysis.num_temporal_bins()
        );

        if self.temporal_config.length_multiplier > 1 {
            log::info!(
                "Length multiplier {}x applied with {} mode: {} temporal frequency resolution",
                self.temporal_config.length_multiplier,
                if self.temporal_config.repeat_data {
                    "repeat"
                } else {
                    "zero padding"
                },
                if self.temporal_config.repeat_data {
                    "extended"
                } else {
                    "enhanced"
                }
            );
        }

        self.temporal_fft = Some(temporal_analysis);
        Ok(())
    }

    /// Perform temporal FFT analysis with explicit length multiplier and repeat option
    pub fn analyze_temporal_with_options(
        &mut self,
        length_multiplier: usize,
        repeat_data: bool,
    ) -> Result<()> {
        let stft_frames = self
            .stft_frames
            .as_ref()
            .ok_or("No STFT analysis available. Call analyze() first.")?;

        if stft_frames.is_empty() || stft_frames[0].is_empty() {
            return Err("No STFT frames available for temporal analysis".to_string());
        }

        let num_frames = stft_frames[0].len();

        // Create config with specified options
        let config =
            TemporalFFTConfig::auto_size_with_options(num_frames, length_multiplier, repeat_data);
        self.temporal_config = config;

        // Perform analysis
        self.analyze_temporal()
    }

    /// Synthesize STFT frames from temporal FFT analysis
    pub fn synthesize_from_temporal(&mut self) -> Result<()> {
        let temporal_analysis = self
            .temporal_fft
            .as_ref()
            .ok_or("No temporal FFT analysis available. Call analyze_temporal() first.")?;

        log::info!(
            "Synthesizing STFT frames from temporal FFT analysis: {} channels",
            temporal_analysis.num_channels
        );

        let synthesizer = TemporalFFTSynthesizer::new(self.temporal_config);
        let reconstructed_frames =
            synthesizer.synthesize(temporal_analysis, self.config.hop_size())?;

        log::info!(
            "Temporal synthesis complete: {} channels reconstructed",
            reconstructed_frames.len()
        );

        self.stft_frames = Some(reconstructed_frames);
        Ok(())
    }

    /// Process temporal FFT data with a custom function
    pub fn process_temporal_fft<F>(&mut self, mut processor: F) -> Result<()>
    where
        F: FnMut(&mut TemporalFFTAnalysis),
    {
        let temporal_analysis = self
            .temporal_fft
            .as_mut()
            .ok_or("No temporal FFT analysis available")?;

        processor(temporal_analysis);
        Ok(())
    }

    /// Synthesize audio from STFT frames
    pub fn synthesize(&self) -> Result<Vec<Vec<f64>>> {
        let frames = self
            .stft_frames
            .as_ref()
            .ok_or("No STFT analysis available. Call analyze() first.")?;

        let synthesizer = STFTSynthesizer::new(self.config)?;
        let mut synthesized_channels = Vec::with_capacity(frames.len());

        for (channel_idx, channel_frames) in frames.iter().enumerate() {
            log::info!(
                "Synthesizing channel {} ({} frames)",
                channel_idx,
                channel_frames.len()
            );

            let synthesized = synthesizer.synthesize(channel_frames)?;
            synthesized_channels.push(synthesized);
        }

        Ok(synthesized_channels)
    }

    /// Get the number of frames per channel (if analysis has been performed)
    pub fn num_frames(&self) -> Option<usize> {
        self.stft_frames
            .as_ref()
            .and_then(|frames| frames.first())
            .map(|first_channel| first_channel.len())
    }

    /// Get the expected number of frames for the current audio data
    pub fn expected_num_frames(&self) -> Option<usize> {
        self.channel_data.as_ref().map(|channels| {
            if let Some(first_channel) = channels.first() {
                let analyzer = STFTAnalyzer::new(self.config).unwrap();
                analyzer.num_frames(first_channel.len())
            } else {
                0
            }
        })
    }

    /// Process a single frame across all channels with a custom function
    pub fn process_frame<F>(&mut self, frame_index: usize, mut processor: F) -> Result<()>
    where
        F: FnMut(usize, &mut STFTFrame), // channel_index, frame
    {
        let frames = self
            .stft_frames
            .as_mut()
            .ok_or("No STFT analysis available")?;

        if frames.is_empty() {
            return Err("No channels available".to_string());
        }

        let num_frames = frames[0].len();
        if frame_index >= num_frames {
            return Err(format!(
                "Frame index {} out of range (0..{})",
                frame_index, num_frames
            ));
        }

        for (channel_idx, channel_frames) in frames.iter_mut().enumerate() {
            if frame_index < channel_frames.len() {
                processor(channel_idx, &mut channel_frames[frame_index]);
            }
        }

        Ok(())
    }

    /// Process all frames with a custom function
    pub fn process_all_frames<F>(&mut self, mut processor: F) -> Result<()>
    where
        F: FnMut(usize, usize, &mut STFTFrame), // channel_index, frame_index, frame
    {
        let frames = self
            .stft_frames
            .as_mut()
            .ok_or("No STFT analysis available")?;

        for (channel_idx, channel_frames) in frames.iter_mut().enumerate() {
            for (frame_idx, frame) in channel_frames.iter_mut().enumerate() {
                processor(channel_idx, frame_idx, frame);
            }
        }

        Ok(())
    }

    /// Clear all data and reset processor
    pub fn clear(&mut self) {
        self.audio_info = None;
        self.channel_data = None;
        self.stft_frames = None;
        self.temporal_fft = None;
    }

    /// Check if processor has audio data loaded
    pub fn has_audio(&self) -> bool {
        self.audio_info.is_some() && self.channel_data.is_some()
    }

    /// Check if STFT analysis has been performed
    pub fn has_analysis(&self) -> bool {
        self.stft_frames.is_some()
    }

    /// Check if temporal FFT analysis has been performed
    pub fn has_temporal_analysis(&self) -> bool {
        self.temporal_fft.is_some()
    }

    /// Synthesize STFT frames from temporal FFT using oscillator bank with parameter control
    ///
    /// This method provides a convenient interface to the oscillator bank synthesis
    /// directly from the processor, handling the temporal analysis access and frame storage.
    pub fn synthesize_temporal_with_oscillator_bank(
        &mut self,
        stretch_start: f64,
        shift_start: Option<f64>,
        dispersion_start: Option<f64>,
        stretch_end: Option<f64>,
        shift_end: Option<f64>,
        dispersion_end: Option<f64>,
        output_frames: Option<usize>,
    ) -> Result<()> {
        let temporal_analysis = self
            .temporal_fft
            .as_ref()
            .ok_or("No temporal FFT analysis available. Call analyze_temporal() first.")?;

        let synthesizer = TemporalFFTSynthesizer::new(self.temporal_config);

        let hop_size = self.config.hop_size();

        log::info!(
            "Performing oscillator bank synthesis: {} output frames, hop size {}",
            output_frames.unwrap_or(0),
            hop_size
        );

        // Perform the synthesis
        let synthesized_frames = synthesizer.synthesize_with_oscillator_bank(
            temporal_analysis,
            hop_size,
            stretch_start,
            shift_start,
            dispersion_start,
            stretch_end,
            shift_end,
            dispersion_end,
            output_frames,
        )?;

        // Store the synthesized frames
        self.stft_frames = Some(synthesized_frames);

        log::info!("Oscillator bank synthesis complete, frames stored in processor");
        Ok(())
    }

    /// Get oscillator bank synthesis parameters with validation
    ///
    /// This helper method validates and provides defaults for oscillator bank parameters
    pub fn validate_oscillator_params(
        &mut self,
        stretch_start: f64,
        shift_start: Option<f64>,
        dispersion_start: Option<f64>,
        stretch_end: Option<f64>,
        shift_end: Option<f64>,
        dispersion_end: Option<f64>,
    ) -> Result<(f64, f64, f64, f64, f64, f64)> {
        // Validate stretch parameters
        if stretch_start <= 0.0 {
            return Err(format!(
                "stretch_start must be positive, got {}",
                stretch_start
            ));
        }

        let stretch_end_val = stretch_end.unwrap_or(stretch_start);
        if stretch_end_val <= 0.0 {
            return Err(format!(
                "stretch_end must be positive, got {}",
                stretch_end_val
            ));
        }

        // Validate shift parameters
        let shift_start_val = shift_start.unwrap_or(0.0);
        let shift_end_val = shift_end.unwrap_or(shift_start_val);

        if !(-1.0..=1.0).contains(&shift_start_val) {
            return Err(format!(
                "shift_start must be between -1.0 and 1.0, got {}",
                shift_start_val
            ));
        }
        if !(-1.0..=1.0).contains(&shift_end_val) {
            return Err(format!(
                "shift_end must be between -1.0 and 1.0, got {}",
                shift_end_val
            ));
        }

        // Dispersion parameters (no specific validation needed)
        let dispersion_start_val = dispersion_start.unwrap_or(1.0);
        let dispersion_end_val = dispersion_end.unwrap_or(dispersion_start_val);

        Ok((
            stretch_start,
            shift_start_val,
            dispersion_start_val,
            stretch_end_val,
            shift_end_val,
            dispersion_end_val,
        ))
    }

    /// Get a summary of oscillator bank synthesis parameters
    pub fn describe_oscillator_synthesis(
        &mut self,
        stretch_start: f64,
        shift_start: f64,
        dispersion_start: f64,
        stretch_end: f64,
        shift_end: f64,
        dispersion_end: f64,
        output_frames: usize,
    ) -> String {
        let mut description = String::new();

        description.push_str("Oscillator Bank Synthesis Parameters:\n");

        // Stretch description
        if (stretch_start - stretch_end).abs() < 1e-6 {
            if (stretch_start - 1.0).abs() < 1e-6 {
                description.push_str("  Temporal stretch: None (1.0)\n");
            } else {
                description.push_str(&format!(
                    "  Temporal stretch: {:.3}x (constant)\n",
                    stretch_start
                ));
            }
        } else {
            description.push_str(&format!(
                "  Temporal stretch: {:.3}x → {:.3}x (linear sweep)\n",
                stretch_start, stretch_end
            ));
        }

        // Shift description
        if (shift_start - shift_end).abs() < 1e-6 {
            if shift_start.abs() < 1e-6 {
                description.push_str("  Frequency shift: None (0.0)\n");
            } else {
                description.push_str(&format!(
                    "  Frequency shift: {:.3} (constant)\n",
                    shift_start
                ));
            }
        } else {
            description.push_str(&format!(
                "  Frequency shift: {:.3} → {:.3} (linear sweep)\n",
                shift_start, shift_end
            ));
        }

        // Dispersion description
        if (dispersion_start - dispersion_end).abs() < 1e-6 {
            if (dispersion_start - 1.0).abs() < 1e-6 {
                description.push_str("  Dispersion: None (1.0)\n");
            } else {
                description.push_str(&format!(
                    "  Dispersion: {:.3} (constant)\n",
                    dispersion_start
                ));
            }
        } else {
            description.push_str(&format!(
                "  Dispersion: {:.3} → {:.3} (linear sweep)\n",
                dispersion_start, dispersion_end
            ));
        }

        description.push_str(&format!("  Output frames: {}\n", output_frames));
        description.push_str("  Method: Complex oscillator bank synthesis\n");

        // Add interpretation notes
        if stretch_start != 1.0 || stretch_end != 1.0 {
            description.push_str(
                "  Note: Stretch < 1.0 compresses temporal frequencies, > 1.0 expands them\n",
            );
        }
        if shift_start != 0.0 || shift_end != 0.0 {
            description
                .push_str("  Note: Shift -1.0 moves all frequencies down, +1.0 moves them up\n");
        }
        if dispersion_start != 1.0 || dispersion_end != 1.0 {
            description
                .push_str("  Note: Dispersion creates frequency-dependent stretching effects\n");
        }

        description
    }

    /// Check if the processor has the capability for oscillator bank synthesis
    pub fn can_oscillator_synthesize(&self) -> bool {
        self.temporal_fft.is_some()
    }

    /// Get recommended oscillator bank parameters for common effects
    pub fn get_oscillator_presets(
        &mut self,
    ) -> Vec<(
        &'static str,
        (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>),
    )> {
        vec![
            // (name, (stretch_start, shift_start, dispersion_start, stretch_end, shift_end, dispersion_end))
            ("Identity", (1.0, 0.0, 1.0, None, None, None)),
            ("Time Stretch", (0.5, 0.0, 1.0, None, None, None)),
            ("Time Compress", (2.0, 0.0, 1.0, None, None, None)),
            ("Shift Up", (1.0, 0.2, 1.0, None, None, None)),
            ("Shift Down", (1.0, -0.2, 1.0, None, None, None)),
            ("Dispersive", (1.0, 0.0, 1.5, None, None, None)),
            (
                "Time Stretch to Compress",
                (0.5, 0.0, 1.0, Some(2.0), None, None),
            ),
            (
                "Frequency Sweep",
                (1.0, -0.3, 1.0, Some(1.0), Some(0.3), None),
            ),
            (
                "Complex Morph",
                (0.7, -0.1, 0.8, Some(1.8), Some(0.1), Some(1.3)),
            ),
            ("Extreme Stretch", (0.25, 0.0, 1.0, Some(4.0), None, None)),
        ]
    }

    /// Validate phase multiply parameters
    ///
    /// Checks if the given parameters are reasonable and logs warnings for
    /// potentially problematic values.
    ///
    /// # Parameters
    ///
    /// * `factor` - Phase multiplication factor to validate
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if valid, warning logged for edge cases
    pub fn validate_phase_multiply_params(&self, factor: f64) -> Result<()> {
        if factor.is_nan() || factor.is_infinite() {
            return Err("Phase multiplication factor must be a finite number".to_string());
        }

        if factor.abs() > 10.0 {
            log::warn!(
                "Large phase multiplication factor ({}) may produce extreme distortion effects",
                factor
            );
        }

        if factor == 0.0 {
            log::warn!(
                "Phase multiplication factor of 0.0 will zero all phases (may produce silence)"
            );
        }

        if factor == 1.0 {
            log::info!(
                "Phase multiplication factor of 1.0 produces no change (identity operation)"
            );
        } else if factor == -1.0 {
            log::info!("Phase multiplication factor of -1.0 produces phase reversal (time-reversal-like effect)");
        }

        Ok(())
    }

    /// Apply temporal phase multiplication to all channels and frequency bins
    pub fn apply_temporal_phase_multiply(&mut self, factor: f64) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            temporal_fft.phase_multiply(factor);
            log::info!(
                "Applied temporal phase multiply with factor {} to {} channels, {} frequency bins",
                factor,
                temporal_fft.num_channels,
                temporal_fft.num_frequency_bins
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal_fft() first.".to_string())
        }
    }

    /// Apply frequency-dependent temporal phase multiplication
    pub fn apply_dispersive_temporal_phase_multiply(
        &mut self,
        base_factor: f64,
        freq_scaling: f64,
    ) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            temporal_fft.phase_multiply_dispersive(base_factor, freq_scaling);
            log::info!(
                "Applied dispersive temporal phase multiply with base factor {} and frequency scaling {}",
                base_factor,
                freq_scaling
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal_fft() first.".to_string())
        }
    }

    /// Apply temporal phase reversal
    pub fn apply_temporal_phase_reversal(&mut self) -> Result<()> {
        self.apply_temporal_phase_multiply(-1.0)
    }

    /// Apply temporal phase scrambling
    pub fn apply_temporal_phase_scrambling(&mut self, intensity: f64, seed: u64) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            temporal_fft.phase_scramble(intensity, seed);
            log::info!(
                "Applied temporal phase scrambling with intensity {} and seed {}",
                intensity,
                seed
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal_fft() first.".to_string())
        }
    }

    /// Get phase statistics from the current temporal FFT data
    pub fn get_temporal_phase_statistics(&self) -> Result<crate::temporal_fft::PhaseStatistics> {
        if let Some(temporal_fft) = &self.temporal_fft {
            Ok(temporal_fft.get_phase_statistics())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal_fft() first.".to_string())
        }
    }

    /// Apply temporal blur to smooth temporal frequency evolution
    ///
    /// # Parameters
    ///
    /// * `radius` - Number of adjacent bins to average (e.g., 1, 2, 3)
    /// * `strength` - Blur strength (0.0 = no blur, 1.0 = full averaging)
    ///
    /// # Example
    ///
    /// ```no_run
    /// processor.apply_temporal_blur(2, 0.5)?; // Moderate smoothing
    /// processor.apply_temporal_blur(3, 0.8)?; // Strong smoothing
    /// ```
    pub fn apply_temporal_blur(&mut self, radius: usize, strength: f64) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            // Validate parameters
            if !(0.0..=1.0).contains(&strength) {
                return Err(format!(
                    "Blur strength must be between 0.0 and 1.0, got {}",
                    strength
                ));
            }

            temporal_fft.blur(radius, strength);
            log::info!(
                "Applied temporal blur with radius {} and strength {}",
                radius,
                strength
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal() first.".to_string())
        }
    }

    // Apply temporal convolution with an impulse response audio file
    ///
    /// This method:
    /// 1. Loads the impulse audio file
    /// 2. Performs STFT analysis using the same configuration
    /// 3. Performs temporal FFT analysis
    /// 4. Matches frame counts (pads with zeros or truncates)
    /// 5. Multiplies temporal spectra (convolution in frequency domain)
    ///
    /// # Parameters
    ///
    /// * `impulse_path` - Path to the impulse response audio file
    ///
    /// # Example
    ///
    /// ```no_run
    /// processor.apply_temporal_convolve("impulse.wav")?;
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_temporal_convolve<P: AsRef<std::path::Path>>(
        &mut self,
        impulse_path: P,
    ) -> Result<()> {
        use crate::audio_io::read_audio_file;
        use crate::stft::{STFTAnalyzer, STFTFrame};
        use crate::temporal_fft::{TemporalConvolve, TemporalFFTAnalyzer};

        // Check that we have temporal FFT data
        let main_temporal = self
            .temporal_fft
            .as_ref()
            .ok_or("No temporal FFT data available. Run analyze_temporal() first.")?;

        // Get the number of original frames from the main temporal FFT
        let target_frame_count = if !main_temporal.bin_ffts.is_empty() {
            main_temporal.bin_ffts[0].original_frames
        } else {
            return Err("Main temporal FFT has no data".to_string());
        };

        log::info!("Loading impulse response from: {:?}", impulse_path.as_ref());

        // Step 1: Load the impulse audio file
        let (impulse_info, impulse_channels) = read_audio_file(impulse_path.as_ref())
            .map_err(|e| format!("Failed to load impulse file: {}", e))?;

        log::info!(
            "Loaded impulse: {} channels, {} Hz, {:.2}s",
            impulse_info.channels,
            impulse_info.sample_rate,
            impulse_info.duration_seconds
        );

        // Check sample rate compatibility
        if let Some(main_info) = &self.audio_info {
            if main_info.sample_rate != impulse_info.sample_rate {
                log::warn!(
                    "Sample rate mismatch: main {} Hz, impulse {} Hz. Results may be unexpected.",
                    main_info.sample_rate,
                    impulse_info.sample_rate
                );
            }
        }

        // Step 2: Perform STFT analysis with the same configuration
        let analyzer = STFTAnalyzer::new(self.config)?;
        let mut impulse_stft_frames = Vec::with_capacity(impulse_channels.len());

        for (ch_idx, channel_data) in impulse_channels.iter().enumerate() {
            log::info!("Analyzing impulse channel {}", ch_idx);
            let mut frames = analyzer.analyze(channel_data)?;

            // Step 3: Match frame counts
            if frames.len() < target_frame_count {
                // Pad with empty frames
                log::info!(
                    "Padding impulse frames from {} to {}",
                    frames.len(),
                    target_frame_count
                );

                let num_bins = if !frames.is_empty() {
                    frames[0].spectrum.len()
                } else {
                    self.config.fft_bins()
                };

                while frames.len() < target_frame_count {
                    frames.push(STFTFrame {
                        spectrum: vec![Complex64::new(0.0, 0.0); num_bins],
                        frame_index: frames.len(),
                        time_position: frames.len() * self.config.hop_size(),
                    });
                }
            } else if frames.len() > target_frame_count {
                // Truncate frames
                log::info!(
                    "Truncating impulse frames from {} to {}",
                    frames.len(),
                    target_frame_count
                );
                frames.truncate(target_frame_count);
            }

            impulse_stft_frames.push(frames);
        }

        // Step 4: Perform temporal FFT analysis on impulse
        log::info!("Performing temporal FFT analysis on impulse response");
        let temporal_analyzer = TemporalFFTAnalyzer::new(self.temporal_config);
        let impulse_temporal = temporal_analyzer.analyze(&impulse_stft_frames)?;

        log::info!(
            "Impulse temporal FFT complete: {} channels, {} frequency bins",
            impulse_temporal.num_channels,
            impulse_temporal.num_frequency_bins
        );

        // Step 5: Apply convolution
        if let Some(main_temporal_mut) = &mut self.temporal_fft {
            let convolve_op = TemporalConvolve::new(impulse_temporal);
            convolve_op.apply(main_temporal_mut);

            log::info!("Temporal convolution complete");
            Ok(())
        } else {
            Err("Lost temporal FFT data during processing".to_string())
        }
    }

    /// Apply temporal power amplitude scaling
    ///
    /// This operation modifies the dynamic range of temporal frequency amplitudes.
    ///
    /// # Parameters
    ///
    /// * `factor` - Power factor to apply:
    ///   - < 1.0: Expands dynamic range (makes quiet parts quieter)
    ///   - = 1.0: No change
    ///   - > 1.0: Compresses dynamic range (makes quiet parts louder)
    ///
    /// # Example
    ///
    /// ```no_run
    /// processor.apply_temporal_power_amplitude(0.5)?;  // Expand dynamic range
    /// processor.apply_temporal_power_amplitude(2.0)?;  // Compress dynamic range
    /// ```
    pub fn apply_temporal_power_amplitude(&mut self, factor: f64) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            // Validate parameter
            if factor <= 0.0 {
                return Err(format!(
                    "Power amplitude factor must be positive, got {}",
                    factor
                ));
            }

            if factor.is_nan() || factor.is_infinite() {
                return Err("Power amplitude factor must be a finite number".to_string());
            }

            temporal_fft.power_amplitude(factor);

            let effect_type = if factor < 1.0 {
                "expansion"
            } else if factor > 1.0 {
                "compression"
            } else {
                "identity"
            };

            log::info!(
                "Applied temporal power amplitude with factor {} (dynamic range {})",
                factor,
                effect_type
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal() first.".to_string())
        }
    }

    /// Apply temporal power amplitude scaling per frequency bin
    ///
    /// This operation modifies the dynamic range of temporal frequency amplitudes
    /// independently for each frequency bin, preserving relative amplitude differences
    /// between bins while modifying dynamics within each bin.
    ///
    /// # Parameters
    ///
    /// * `factor` - Power factor to apply:
    ///   - < 1.0: Expands dynamic range within each bin
    ///   - = 1.0: No change
    ///   - > 1.0: Compresses dynamic range within each bin
    ///
    /// # Example
    ///
    /// ```no_run
    /// processor.apply_temporal_power_amplitude_per_bin(0.5)?;  // Expand per-bin dynamics
    /// processor.apply_temporal_power_amplitude_per_bin(2.0)?;  // Compress per-bin dynamics
    /// ```
    pub fn apply_temporal_power_amplitude_per_bin(&mut self, factor: f64) -> Result<()> {
        if let Some(temporal_fft) = &mut self.temporal_fft {
            // Validate parameter
            if factor <= 0.0 {
                return Err(format!(
                    "Power amplitude factor must be positive, got {}",
                    factor
                ));
            }

            if factor.is_nan() || factor.is_infinite() {
                return Err("Power amplitude factor must be a finite number".to_string());
            }

            temporal_fft.power_amplitude_per_bin(factor);

            let effect_type = if factor < 1.0 {
                "expansion"
            } else if factor > 1.0 {
                "compression"
            } else {
                "identity"
            };

            log::info!(
                "Applied temporal power amplitude per-bin with factor {} (dynamic range {} per frequency bin)",
                factor,
                effect_type
            );
            Ok(())
        } else {
            Err("No temporal FFT data available. Run analyze_temporal() first.".to_string())
        }
    }

    /// Apply temporal cross-synthesis using current magnitude and phase from external file
    ///
    /// This method:
    /// 1. Loads the phase source audio file
    /// 2. Performs STFT analysis using the same configuration
    /// 3. Performs temporal FFT analysis
    /// 4. Keeps magnitude from current temporal FFT, replaces phase with source
    ///
    /// # Parameters
    ///
    /// * `phase_path` - Path to audio file providing phase information
    ///
    /// # Example
    ///
    /// ```no_run
    /// processor.apply_temporal_cross_synthesize("drums.wav")?;
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_temporal_cross_synthesize<P: AsRef<std::path::Path>>(
        &mut self,
        phase_path: P,
    ) -> Result<()> {
        use crate::audio_io::read_audio_file;
        use crate::stft::{STFTAnalyzer, STFTFrame};
        use crate::temporal_fft::{TemporalCrossSynthesize, TemporalFFTAnalyzer};

        // Check that we have temporal FFT data
        let main_temporal = self
            .temporal_fft
            .as_ref()
            .ok_or("No temporal FFT data available. Run analyze_temporal() first.")?;

        // Get the number of original frames from the main temporal FFT
        let target_frame_count = if !main_temporal.bin_ffts.is_empty() {
            main_temporal.bin_ffts[0].original_frames
        } else {
            return Err("Main temporal FFT has no data".to_string());
        };

        log::info!("Loading phase source from: {:?}", phase_path.as_ref());

        // Load the phase source audio file
        let (phase_info, phase_channels) = read_audio_file(phase_path.as_ref())
            .map_err(|e| format!("Failed to load phase file: {}", e))?;

        log::info!(
            "Phase source: {} channels, {} Hz, {:.2}s",
            phase_info.channels,
            phase_info.sample_rate,
            phase_info.duration_seconds
        );

        // Check sample rate compatibility
        if let Some(main_info) = &self.audio_info {
            if main_info.sample_rate != phase_info.sample_rate {
                log::warn!(
                    "Sample rate mismatch: current {} Hz, phase {} Hz. Results may be unexpected.",
                    main_info.sample_rate,
                    phase_info.sample_rate
                );
            }
        }

        // Analyze phase source with STFT
        let analyzer = STFTAnalyzer::new(self.config)?;
        let mut phase_stft_frames = Vec::with_capacity(phase_channels.len());

        for (ch_idx, channel_data) in phase_channels.iter().enumerate() {
            log::info!("Analyzing phase source channel {}", ch_idx);
            let mut frames = analyzer.analyze(channel_data)?;

            // Match frame counts
            if frames.len() < target_frame_count {
                // Pad with empty frames
                log::info!(
                    "Padding phase frames from {} to {}",
                    frames.len(),
                    target_frame_count
                );

                let num_bins = if !frames.is_empty() {
                    frames[0].spectrum.len()
                } else {
                    self.config.fft_bins()
                };

                while frames.len() < target_frame_count {
                    frames.push(STFTFrame {
                        spectrum: vec![Complex64::new(0.0, 0.0); num_bins],
                        frame_index: frames.len(),
                        time_position: frames.len() * self.config.hop_size(),
                    });
                }
            } else if frames.len() > target_frame_count {
                // Truncate frames
                log::info!(
                    "Truncating phase frames from {} to {}",
                    frames.len(),
                    target_frame_count
                );
                frames.truncate(target_frame_count);
            }

            phase_stft_frames.push(frames);
        }

        // Perform temporal FFT analysis on phase source
        log::info!("Performing temporal FFT analysis on phase source");
        let temporal_analyzer = TemporalFFTAnalyzer::new(self.temporal_config);
        let phase_temporal = temporal_analyzer.analyze(&phase_stft_frames)?;

        log::info!(
            "Phase temporal FFT complete: {} channels, {} frequency bins",
            phase_temporal.num_channels,
            phase_temporal.num_frequency_bins
        );

        // Apply in-place cross-synthesis
        if let Some(main_temporal_mut) = &mut self.temporal_fft {
            let cross_synth = TemporalCrossSynthesize::new(phase_temporal);
            cross_synth.apply(main_temporal_mut);

            log::info!("Temporal cross-synthesis complete");
            log::info!("Current temporal FFT now has:");
            log::info!("  - Magnitude: from original analysis");
            log::info!("  - Phase: from '{}'", phase_path.as_ref().display());
            Ok(())
        } else {
            Err("Lost temporal FFT data during processing".to_string())
        }
    }
}
impl Default for STFTProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::WindowType;
    use std::f64::consts::PI;

    fn generate_test_audio(
        sample_rate: u32,
        duration_seconds: f64,
        num_channels: usize,
    ) -> (AudioInfo, Vec<Vec<f64>>) {
        let samples = (sample_rate as f64 * duration_seconds) as usize;
        let mut channels = Vec::with_capacity(num_channels);

        for ch in 0..num_channels {
            let mut channel_data = Vec::with_capacity(samples);
            let frequency = 440.0 * (ch + 1) as f64; // Different frequency per channel

            for i in 0..samples {
                let t = i as f64 / sample_rate as f64;
                let sample = (2.0 * PI * frequency * t).sin() * 0.5;
                channel_data.push(sample);
            }
            channels.push(channel_data);
        }

        let info = AudioInfo::new(sample_rate, num_channels, samples);
        (info, channels)
    }

    #[test]
    fn test_processor_creation() {
        let processor = STFTProcessor::new();
        assert_eq!(processor.config().window_size, 1024);
        assert_eq!(processor.config().overlap_factor, 8);
        assert!(!processor.has_audio());
        assert!(!processor.has_analysis());
    }

    #[test]
    fn test_processor_with_config() {
        let config = STFTConfig::new(512, 4, WindowType::Hamming).unwrap();
        let processor = STFTProcessor::with_config(config);
        assert_eq!(processor.config().window_size, 512);
        assert_eq!(processor.config().overlap_factor, 4);
        assert_eq!(processor.config().window_type, WindowType::Hamming);
    }

    #[test]
    fn test_load_audio() {
        let mut processor = STFTProcessor::new();
        let (info, channels) = generate_test_audio(44100, 0.5, 2);

        assert!(processor.load_audio(info.clone(), channels).is_ok());
        assert!(processor.has_audio());
        assert_eq!(processor.audio_info().unwrap().channels, 2);
        assert_eq!(processor.audio_info().unwrap().sample_rate, 44100);
    }

    #[test]
    fn test_analyze_and_synthesize() {
        let mut processor = STFTProcessor::new();
        let (info, channels) = generate_test_audio(44100, 0.5, 1);

        // Load audio
        processor.load_audio(info, channels.clone()).unwrap();

        // Analyze
        processor.analyze().unwrap();
        assert!(processor.has_analysis());
        assert!(processor.num_frames().is_some());

        // Synthesize
        let synthesized = processor.synthesize().unwrap();
        assert_eq!(synthesized.len(), 1); // One channel

        // Check that synthesis preserves approximate signal length
        // (allowing for some variation due to windowing)
        let original_len = channels[0].len();
        let synthesized_len = synthesized[0].len();
        let len_diff = (original_len as i64 - synthesized_len as i64).abs();
        assert!(len_diff < processor.config().window_size as i64);

        // Check that the signal is reasonably preserved
        let min_len = std::cmp::min(original_len, synthesized_len);
        let start_check = processor.config().window_size; // Skip edge effects
        let end_check = min_len - processor.config().window_size;

        if end_check > start_check {
            let mut error_sum = 0.0;
            for i in start_check..end_check {
                let error = (channels[0][i] - synthesized[0][i]).abs();
                error_sum += error;
            }
            let mean_error = error_sum / (end_check - start_check) as f64;

            println!("Mean reconstruction error: {:.6}", mean_error);
            assert!(
                mean_error < 0.1,
                "Reconstruction error too large: {}",
                mean_error
            );
        }
    }

    #[test]
    fn test_frame_processing() {
        let mut processor = STFTProcessor::new();
        let (info, channels) = generate_test_audio(44100, 0.1, 2);

        processor.load_audio(info, channels).unwrap();
        processor.analyze().unwrap();

        let num_frames = processor.num_frames().unwrap();
        assert!(num_frames > 0);

        // Test processing a single frame
        processor
            .process_frame(0, |channel_idx, frame| {
                // Simple test: zero out the first bin
                frame.spectrum[0] = num_complex::Complex64::new(0.0, 0.0);
                println!("Processed frame 0, channel {}", channel_idx);
            })
            .unwrap();

        // Test processing all frames
        let mut frame_count = 0;
        processor
            .process_all_frames(|_channel_idx, _frame_idx, _frame| {
                frame_count += 1;
            })
            .unwrap();

        // Should have processed frames for all channels
        assert_eq!(frame_count, num_frames * 2); // 2 channels
    }

    #[test]
    fn test_error_conditions() {
        let mut processor = STFTProcessor::new();

        // Try to analyze without data
        assert!(processor.analyze().is_err());

        // Try to synthesize without analysis
        assert!(processor.synthesize().is_err());

        // Try to process frames without analysis
        assert!(processor.process_frame(0, |_, _| {}).is_err());

        // Load mismatched data
        let info = AudioInfo::new(44100, 2, 1000);
        let channels = vec![vec![0.0; 1000]]; // Only 1 channel, but info says 2
        assert!(processor.load_audio(info, channels).is_err());
    }
}
