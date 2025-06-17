use anakroni_lib::{
    audio_io::{read_audio_bytes, write_audio_bytes, AudioInfo},
    processor::STFTProcessor,
    stft::STFTConfig,
    utils,
    window::WindowType,
    Complex64, Result,
};
use js_sys::Float32Array;
use serde::Serialize;
use wasm_bindgen::prelude::*;

// Set up panic hook for better error messages
fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
pub fn start() {
    init_panic_hook();
    anakroni_lib::init();
}

// Serde-compatible info structs for passing to JavaScript
#[derive(Serialize)]
struct AudioInfoJs {
    sample_rate: u32,
    channels: usize,
    duration_samples: usize,
    duration_seconds: f64,
}

impl From<&AudioInfo> for AudioInfoJs {
    fn from(info: &AudioInfo) -> Self {
        Self {
            sample_rate: info.sample_rate,
            channels: info.channels,
            duration_samples: info.duration_samples,
            duration_seconds: info.duration_seconds,
        }
    }
}

#[derive(Serialize)]
struct STFTConfigJs {
    window_size: usize,
    overlap_factor: usize,
    window_type: String,
    hop_size: usize,
    overlap_percent: f64,
    fft_bins: usize,
}

impl From<&STFTConfig> for STFTConfigJs {
    fn from(config: &STFTConfig) -> Self {
        Self {
            window_size: config.window_size,
            overlap_factor: config.overlap_factor,
            window_type: config.window_type.name().to_string(),
            hop_size: config.hop_size(),
            overlap_percent: config.overlap_percent(),
            fft_bins: config.fft_bins(),
        }
    }
}

#[derive(Serialize)]
struct ProcessorInfoJs {
    audio_info: Option<AudioInfoJs>,
    config: STFTConfigJs,
    has_audio: bool,
    has_analysis: bool,
    num_frames: Option<usize>,
}

#[wasm_bindgen]
pub struct WasmSTFTProcessor {
    processor: STFTProcessor,
    original_audio: Option<(AudioInfo, Vec<Vec<f64>>)>,
}

#[wasm_bindgen]
impl WasmSTFTProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        init_panic_hook();

        Self {
            processor: STFTProcessor::new(),
            original_audio: None,
        }
    }

    /// Load audio data from a Float32Array (interleaved)
    #[wasm_bindgen]
    pub fn load_audio_data(
        &mut self,
        channels: u16,
        sample_rate: u32,
        audio_data: &Float32Array,
    ) -> Result<()> {
        let mut samples = Vec::new();
        let length = audio_data.length() as usize;
        let channel_length = length / channels as usize;

        // Prepare to deinterleave the audio samples
        for _ in 0..channels {
            samples.push(vec![0.0; channel_length]);
        }

        // Copy audio data to Rust
        let mut buffer = vec![0.0; length];
        audio_data.copy_to(&mut buffer[..]);

        // Deinterleave
        for i in 0..channel_length {
            for ch in 0..channels as usize {
                samples[ch][i] = buffer[i * channels as usize + ch] as f64;
            }
        }

        let audio_info = AudioInfo::new(sample_rate, channels as usize, channel_length);

        // Store original audio for reset functionality
        self.original_audio = Some((audio_info.clone(), samples.clone()));

        // Load into processor
        self.processor.load_audio(audio_info, samples)?;

        Ok(())
    }

    /// Read audio from byte data (e.g., uploaded file)
    #[wasm_bindgen]
    pub fn read_audio_bytes(&mut self, data: js_sys::Uint8Array) -> Result<()> {
        let data_vec = data.to_vec();

        match read_audio_bytes(data_vec) {
            Ok((audio_info, channel_data)) => {
                // Store original audio
                self.original_audio = Some((audio_info.clone(), channel_data.clone()));

                // Load into processor
                self.processor.load_audio(audio_info, channel_data)?;
                Ok(())
            }
            Err(err) => Err(err.to_string()),
        }
    }

    /// Perform STFT analysis
    #[wasm_bindgen]
    pub fn analyze(&mut self) -> Result<()> {
        self.processor.analyze()
    }

    /// Get processor information
    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        let info = ProcessorInfoJs {
            audio_info: self.processor.audio_info().map(AudioInfoJs::from),
            config: STFTConfigJs::from(self.processor.config()),
            has_audio: self.processor.has_audio(),
            has_analysis: self.processor.has_analysis(),
            num_frames: self.processor.num_frames(),
        };

        serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::null())
    }

    /// Set STFT configuration
    #[wasm_bindgen]
    pub fn set_config(
        &mut self,
        window_size: usize,
        overlap_factor: usize,
        window_type: &str,
    ) -> Result<()> {
        let wt = match window_type.to_lowercase().as_str() {
            "hanning" => WindowType::Hanning,
            "hamming" => WindowType::Hamming,
            "rectangular" => WindowType::Rectangular,
            "bartlett" => WindowType::Bartlett,
            _ => return Err(format!("Unknown window type: {}", window_type)),
        };

        let config = STFTConfig::new(window_size, overlap_factor, wt)?;
        self.processor.set_config(config);
        Ok(())
    }

    /// Load a configuration preset
    #[wasm_bindgen]
    pub fn load_preset(&mut self, preset_name: &str) -> Result<()> {
        let preset_name_formatted = preset_name.to_lowercase().replace("_", " ");
        let presets = utils::presets::all_presets();

        if let Some((_, config)) = presets
            .iter()
            .find(|(name, _)| name.to_lowercase() == preset_name_formatted)
        {
            self.processor.set_config(config.clone());
            Ok(())
        } else {
            Err(format!("Unknown preset: {}", preset_name))
        }
    }

    /// Get available presets
    #[wasm_bindgen]
    pub fn get_presets(&self) -> JsValue {
        let presets = utils::presets::all_presets();
        let preset_info: Vec<_> = presets
            .iter()
            .map(|(name, config)| {
                serde_json::json!({
                    "name": name,
                    "key": name.to_lowercase().replace(" ", "_"),
                    "window_size": config.window_size,
                    "overlap_factor": config.overlap_factor,
                    "window_type": config.window_type.name()
                })
            })
            .collect();

        serde_wasm_bindgen::to_value(&preset_info).unwrap_or(JsValue::null())
    }

    /// Get processed audio data as Float32Array (interleaved)
    #[wasm_bindgen]
    pub fn get_processed_audio(&self) -> Result<Float32Array> {
        let synthesized = self.processor.synthesize()?;

        if synthesized.is_empty() {
            return Ok(Float32Array::new_with_length(0));
        }

        let channels = synthesized.len();
        let samples_per_channel = synthesized[0].len();
        let total_samples = channels * samples_per_channel;

        // Find max amplitude for normalization
        let mut max_amp = 0.0f64;
        for channel in &synthesized {
            for &sample in channel {
                max_amp = max_amp.max(sample.abs());
            }
        }

        if max_amp == 0.0 {
            max_amp = 1.0;
        }

        // Create interleaved output
        let result = Float32Array::new_with_length(total_samples as u32);

        for i in 0..samples_per_channel {
            for ch in 0..channels {
                let index = (i * channels + ch) as u32;
                let normalized_sample = (synthesized[ch][i] / max_amp) as f32;
                result.set_index(index, normalized_sample);
            }
        }

        Ok(result)
    }

    /// Save processed audio as WAV bytes
    #[wasm_bindgen]
    pub fn save_audio_bytes(&self) -> std::result::Result<js_sys::Uint8Array, JsValue> {
        let synthesized = match self.processor.synthesize() {
            Ok(data) => data,
            Err(e) => return Err(JsValue::from_str(&e)),
        };

        let audio_info = match self.processor.audio_info() {
            Some(info) => info,
            None => return Err(JsValue::from_str("No audio info available")),
        };

        // Update info with new length
        let new_length = if !synthesized.is_empty() {
            synthesized[0].len()
        } else {
            0
        };

        let updated_info = AudioInfo::new(audio_info.sample_rate, audio_info.channels, new_length);

        match write_audio_bytes(&updated_info, &synthesized) {
            Ok(bytes) => {
                let array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(&bytes);
                Ok(array)
            }
            Err(e) => Err(JsValue::from_str(&format!("Error: {}", e))),
        }
    }

    /// Reset to original audio data
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Result<()> {
        if let Some((audio_info, channel_data)) = self.original_audio.clone() {
            self.processor.load_audio(audio_info, channel_data)?;
            Ok(())
        } else {
            Err("No original audio data available".to_string())
        }
    }

    /// Apply simple highpass filter
    #[wasm_bindgen]
    pub fn apply_highpass(&mut self, frequency_hz: f64) -> Result<()> {
        if !self.processor.has_analysis() {
            return Err("No STFT analysis available".to_string());
        }

        if let Some(info) = self.processor.audio_info() {
            let config = self.processor.config();
            let cutoff_bin =
                utils::frequency_to_bin(frequency_hz, info.sample_rate, config.window_size);

            self.processor.process_all_frames(|_ch, _frame, frame| {
                for bin in 0..cutoff_bin.min(frame.spectrum.len()) {
                    frame.spectrum[bin] = Complex64::new(0.0, 0.0);
                }
            })?;
        }

        Ok(())
    }

    /// Apply simple lowpass filter
    #[wasm_bindgen]
    pub fn apply_lowpass(&mut self, frequency_hz: f64) -> Result<()> {
        if !self.processor.has_analysis() {
            return Err("No STFT analysis available".to_string());
        }

        if let Some(info) = self.processor.audio_info() {
            let config = self.processor.config();
            let cutoff_bin =
                utils::frequency_to_bin(frequency_hz, info.sample_rate, config.window_size);

            self.processor.process_all_frames(|_ch, _frame, frame| {
                for bin in cutoff_bin..frame.spectrum.len() {
                    frame.spectrum[bin] = Complex64::new(0.0, 0.0);
                }
            })?;
        }

        Ok(())
    }

    /// Zero out DC component
    #[wasm_bindgen]
    pub fn zero_dc(&mut self) -> Result<()> {
        if !self.processor.has_analysis() {
            return Err("No STFT analysis available".to_string());
        }

        self.processor.process_all_frames(|_ch, _frame, frame| {
            frame.spectrum[0] = Complex64::new(0.0, 0.0);
        })?;

        Ok(())
    }

    /// Get spectrum data for visualization
    #[wasm_bindgen]
    pub fn get_spectrum_data(
        &self,
        channel: usize,
        frame: usize,
        max_points: usize,
    ) -> Result<Float32Array> {
        if let Some(frames) = self.processor.stft_frames() {
            if channel >= frames.len() {
                return Err(format!("Channel {} out of range", channel));
            }

            if frame >= frames[channel].len() {
                return Err(format!("Frame {} out of range", frame));
            }

            let spectrum = &frames[channel][frame].spectrum;
            let mut magnitudes = Vec::with_capacity(spectrum.len());

            for complex_val in spectrum {
                magnitudes.push(complex_val.norm() as f32);
            }

            // Downsample if needed
            let output_size = max_points.min(magnitudes.len());
            let result = Float32Array::new_with_length(output_size as u32);

            if output_size == magnitudes.len() {
                // No downsampling needed
                for (i, &mag) in magnitudes.iter().enumerate() {
                    result.set_index(i as u32, mag);
                }
            } else {
                // Downsample
                for i in 0..output_size {
                    let src_idx = (i * (magnitudes.len() - 1)) / (output_size - 1);
                    result.set_index(i as u32, magnitudes[src_idx]);
                }
            }

            Ok(result)
        } else {
            Err("No STFT analysis available".to_string())
        }
    }

    /// Get frequency bin information
    #[wasm_bindgen]
    pub fn get_bin_frequency(&self, bin: usize) -> f64 {
        if let Some(info) = self.processor.audio_info() {
            let config = self.processor.config();
            utils::bin_to_frequency(bin, info.sample_rate, config.window_size)
        } else {
            0.0
        }
    }

    /// Get analysis summary as string
    #[wasm_bindgen]
    pub fn get_analysis_summary(&self) -> String {
        utils::analysis_summary(&self.processor)
    }
}
