# Anakroni

A two-dimensional Fourier transform audio processor implementing Short-Time Fourier Transform (STFT) followed by secondary FFT analysis on the temporal evolution of individual frequency bins.


This application is an experimental audio processing application made due to a thought experiment.

"What if we analyzed an audio file with into frames with STFT and then did a single FFT transform on all the 
frames corresponding to each bin, and then does operations on the FFTs?" 

This is similar to the Mammoth FFT approach where we just do one FFT frame for a whole audio file, but instead we do it on the temporal evolution of each 
frequency bin. 

The results may not be very usable, but could be fun...

## Overview

Anakroni performs spectral analysis in two stages:
1. **Spatial Domain**: STFT decomposes audio into time-frequency representation
2. **Temporal Domain**: FFT analysis of each frequency bin's magnitude and phase evolution over time

This creates a representation where both the spectral content and its rate of change can be independently manipulated through complex-domain operations.

## Technical Architecture

### Signal Flow
```
Input Audio → STFT Analysis → Temporal FFT → Operations → IFFT → ISTFT → Output Audio
```

### STFT Parameters
- **Window sizes**: 256, 512, 1024, 2048, 4096 samples
- **Overlap factors**: 1-32 (hop_size = window_size / overlap_factor)
- **Window functions**: Hanning, Hamming, Rectangular, Bartlett
- **FFT implementation**: Real-to-complex FFT via realfft crate

### Temporal FFT Analysis
- Performs complex FFT on temporal evolution vectors of each STFT bin
- FFT size automatically adjusted to next power of 2 ≥ frame count
- Optional length multiplier for increased length due to stretch and/or increased temporal frequency resolution
- Maintains Hermitian symmetry for real-valued reconstruction

## Implemented Operations

### Frequency Domain Operations (Temporal FFT)

#### Linear Operations
- **DC Removal**: Zero bin 0 of temporal spectrum
- **High-pass Filter**: Zero bins below cutoff frequency (normalized 0-1)
- **Low-pass Filter**: Zero bins above cutoff frequency (normalized 0-1)

#### Shift operations on temporal FFT bins
- **Temporal Shift**: Shift bins up or down
- **Circular Shift**: Shift bins up or down with wrap around
- **Dispersive Shift**: Frequency-dependent shift

#### Non-Linear Operations
- **Temporal Stretch**: Scale bin placement, <1 time stretch, >1 time compress
- **Phase Multiplication**: Multiply phases of bins in the temporal FFT
- **Amplitude Power Scaling**: Scale amplitudes of bins  via power law

#### Multi-Input Operations
- **Convolution**: Complex multiplication with secondary temporal FFT
- **Cross-synthesis (TODO)**: Magnitude/phase exchange between signals

### Synthesis Methods

#### Standard IFFT Path
- Inverse temporal FFT → ISTFT with overlap-add
- Maintains phase coherence across frames

#### Oscillator Bank Synthesis (WARNING: very computationally intensive, use on short files only)
- Direct additive synthesis from temporal FFT bins
- Parameters:
  - Frequency scaling factor (stretch)
  - Frequency offset (shift)
  - Frequency-dependent scaling (dispersion)
- Supports time-varying parameter interpolation

## Implementation Details

### Data Structures
```rust
TemporalFFTAnalysis {
    bin_ffts: Vec<TemporalBinFFT>,  // Per-bin temporal spectra
    config: TemporalFFTConfig,       // FFT parameters
    num_frequency_bins: usize,       // STFT bin count
    num_channels: usize,             // Audio channel count
}
```

### Numerical Precision
- 64-bit floating point throughout signal path
- Complex64 (f64 real + f64 imaginary) for spectral data
- Automatic gain compensation for window overlap

### Memory Layout
- Non-interleaved channel storage
- Row-major temporal spectrum storage
- Pre-allocated buffers for real-time operation

## Build Configuration

### Dependencies
- `rustfft`: Complex FFT operations
- `realfft`: Optimized real-to-complex transforms
- `symphonia`: Audio codec support
- `hound`: WAV file I/O

### Compilation
```bash
cargo build --release --features image
```

### Feature Flags
- `image`: Spectrogram generation support
- `wasm`: WebAssembly target support (for future web client use)

## Usage Examples

### Basic Processing Pipeline
```bash
anakroni
> set window_size 2048
> set overlap 8
> load input.wav
> temporal_analyze
> temporal_process temporal_stretch 0.5
> temporal_synthesize
> save output.wav
```

## License

MIT License. See LICENSE file for details.