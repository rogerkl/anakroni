//! Audio I/O functionality using Symphonia
//!
//! Handles reading and writing audio files in various formats,
//! similar to the mammut-fft project but adapted for multi-channel STFT processing.

use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io::Cursor;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::{i24, u24};

/// Audio metadata information
#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_samples: usize,
    pub duration_seconds: f64,
}

impl AudioInfo {
    pub fn new(sample_rate: u32, channels: usize, duration_samples: usize) -> Self {
        let duration_seconds = duration_samples as f64 / sample_rate as f64;
        Self {
            sample_rate,
            channels,
            duration_samples,
            duration_seconds,
        }
    }
}

/// Read audio from a MediaSourceStream and return channel data and metadata
fn read_audio_stream(mss: MediaSourceStream) -> Result<(AudioInfo, Vec<Vec<f64>>), Box<dyn Error>> {
    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

    let track = probed
        .format
        .default_track()
        .ok_or("No default track found")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|_| "Unsupported codec")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("Sample rate not specified")?;

    let channels = track
        .codec_params
        .channels
        .ok_or("Channels not specified")?
        .count();

    let mut channel_buffers: Vec<Vec<f64>> = vec![Vec::new(); channels];
    let mut format = probed.format;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(err)) => {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(Box::new(err));
            }
            Err(err) => return Err(Box::new(err)),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::IoError(err)) => {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(Box::new(err));
            }
            Err(err) => return Err(Box::new(err)),
        };

        match decoded {
            AudioBufferRef::F32(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push(sample as f64);
                    }
                }
            }
            AudioBufferRef::F64(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push(sample);
                    }
                }
            }
            AudioBufferRef::U32(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push((sample as f64 / u32::MAX as f64) * 2.0 - 1.0);
                    }
                }
            }
            AudioBufferRef::U24(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c]
                            .push((sample.inner() as f64 / u24::MAX.inner() as f64) * 2.0 - 1.0);
                    }
                }
            }
            AudioBufferRef::U16(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push((sample as f64 / u16::MAX as f64) * 2.0 - 1.0);
                    }
                }
            }
            AudioBufferRef::U8(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push((sample as f64 / u8::MAX as f64) * 2.0 - 1.0);
                    }
                }
            }
            AudioBufferRef::S32(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push(sample as f64 / i32::MAX as f64);
                    }
                }
            }
            AudioBufferRef::S24(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push(sample.inner() as f64 / i24::MAX.inner() as f64);
                    }
                }
            }
            AudioBufferRef::S16(buffer) => {
                for c in 0..channels {
                    let samples = buffer.chan(c);
                    for &sample in samples {
                        channel_buffers[c].push(sample as f64 / i16::MAX as f64);
                    }
                }
            }
            _ => return Err("Unsupported audio format".into()),
        }
    }

    let duration_samples = if !channel_buffers.is_empty() {
        channel_buffers[0].len()
    } else {
        0
    };

    let info = AudioInfo::new(sample_rate, channels, duration_samples);

    Ok((info, channel_buffers))
}

/// Read audio file from filesystem path
#[cfg(not(target_arch = "wasm32"))]
pub fn read_audio_file<P: AsRef<Path>>(
    path: P,
) -> Result<(AudioInfo, Vec<Vec<f64>>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    read_audio_stream(mss)
}

/// Read audio data from byte buffer
pub fn read_audio_bytes(data: Vec<u8>) -> Result<(AudioInfo, Vec<Vec<f64>>), Box<dyn Error>> {
    let cursor = Cursor::new(data);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    read_audio_stream(mss)
}

/// Write audio data to WAV file
#[cfg(not(target_arch = "wasm32"))]
pub fn write_audio_file<P: AsRef<Path>>(
    path: P,
    audio_info: &AudioInfo,
    channel_data: &[Vec<f64>],
) -> Result<(), Box<dyn Error>> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::BufWriter;

    if channel_data.len() != audio_info.channels {
        return Err("Channel count mismatch".into());
    }

    let spec = WavSpec {
        channels: audio_info.channels as u16,
        sample_rate: audio_info.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let file = File::create(path)?;
    let buf_writer = BufWriter::new(file);
    let mut writer = WavWriter::new(buf_writer, spec)?;

    // Interleave the channel data for writing
    let num_samples = if !channel_data.is_empty() {
        channel_data[0].len()
    } else {
        0
    };

    for sample_idx in 0..num_samples {
        for channel_idx in 0..audio_info.channels {
            let sample = channel_data[channel_idx][sample_idx] as f32;
            writer.write_sample(sample)?;
        }
    }

    writer.finalize()?;
    Ok(())
}

/// Write audio data to WAV format in memory and return bytes
pub fn write_audio_bytes(
    audio_info: &AudioInfo,
    channel_data: &[Vec<f64>],
) -> Result<Vec<u8>, Box<dyn Error>> {
    use hound::{SampleFormat, WavSpec, WavWriter};

    if channel_data.len() != audio_info.channels {
        return Err("Channel count mismatch".into());
    }

    let spec = WavSpec {
        channels: audio_info.channels as u16,
        sample_rate: audio_info.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;

        let num_samples = if !channel_data.is_empty() {
            channel_data[0].len()
        } else {
            0
        };

        for sample_idx in 0..num_samples {
            for channel_idx in 0..audio_info.channels {
                let sample = channel_data[channel_idx][sample_idx] as f32;
                writer.write_sample(sample)?;
            }
        }

        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_info() {
        let info = AudioInfo::new(44100, 2, 44100);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.duration_samples, 44100);
        assert!((info.duration_seconds - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_write_read_audio_bytes() {
        // Create test audio data
        let sample_rate = 44100;
        let channels = 2;
        let duration_samples = 1000;

        let mut test_data = vec![Vec::new(); channels];
        for i in 0..duration_samples {
            let t = i as f64 / sample_rate as f64;
            let sample_left = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            let sample_right = (2.0 * std::f64::consts::PI * 880.0 * t).sin() * 0.5;

            test_data[0].push(sample_left);
            test_data[1].push(sample_right);
        }

        let info = AudioInfo::new(sample_rate, channels, duration_samples);

        // Write to bytes
        let wav_bytes = write_audio_bytes(&info, &test_data).unwrap();
        assert!(!wav_bytes.is_empty());

        // Read back from bytes
        let (read_info, read_data) = read_audio_bytes(wav_bytes).unwrap();

        assert_eq!(read_info.sample_rate, info.sample_rate);
        assert_eq!(read_info.channels, info.channels);
        assert_eq!(read_data.len(), channels);

        // Check that data is approximately the same (allowing for floating point precision)
        for ch in 0..channels {
            assert_eq!(read_data[ch].len(), test_data[ch].len());
            for (i, (&original, &read)) in
                test_data[ch].iter().zip(read_data[ch].iter()).enumerate()
            {
                let diff = (original - read).abs();
                assert!(
                    diff < 1e-5,
                    "Sample {}, channel {}: {} != {} (diff: {})",
                    i,
                    ch,
                    original,
                    read,
                    diff
                );
            }
        }
    }
}
