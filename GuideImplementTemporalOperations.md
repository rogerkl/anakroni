Guide to Implementing New Temporal FFT Operations
=================================================

Overview
--------

The temporal FFT processing system uses a trait-based architecture that makes it easy to add new operations. This guide will walk you through creating a new temporal operation from scratch.

Architecture Overview
---------------------

The temporal FFT operations are organized as follows:

    lib/src/temporal_fft/operations/
    +-- mod.rs              # Module root and trait definition
    +-- basic.rs            # Basic operations (DC, filters)
    +-- shift.rs            # Shift operations
    +-- stretch.rs          # Stretch operations
    +-- phase_multiply.rs   # Phase operations
    +-- utils.rs            # Shared utilities

Each operation implements the `TemporalOperation` trait:

rust

    pub trait TemporalOperation {
        fn apply(&self, analysis: &mut TemporalFFTAnalysis);
        fn name(&self) -> &'static str;
        fn description(&self) -> String;
    }

Step-by-Step Guide: Creating a New Operation
--------------------------------------------

Let's create a new operation called "Temporal Blur" that smooths the temporal evolution by averaging adjacent temporal frequency bins.

### Step 1: Create the Operation File

Create a new file `lib/src/temporal_fft/operations/blur.rs`:

rust

    //! Temporal blur operation - smooths temporal frequency spectrum
    
    use super::{TemporalOperation, process_all_temporal_spectra};
    use crate::temporal_fft::TemporalFFTAnalysis;
    use num_complex::Complex64;
    
    /// Temporal blur operation
    pub struct TemporalBlur {
        /// Blur radius (number of adjacent bins to average)
        radius: usize,
        /// Blur strength (0.0 = no blur, 1.0 = full averaging)
        strength: f64,
    }
    
    impl TemporalBlur {
        pub fn new(radius: usize, strength: f64) -> Self {
            Self { radius, strength }
        }
    }
    
    impl TemporalOperation for TemporalBlur {
        fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
            let fft_size = analysis.config.fft_size;
            
            log::info!(
                "Applying temporal blur with radius {} and strength {}",
                self.radius,
                self.strength
            );
    
            process_all_temporal_spectra(analysis, |_ch_idx, _bin_idx, temporal_spectrum| {
                blur_spectrum(temporal_spectrum, self.radius, self.strength, fft_size);
            });
        }
    
        fn name(&self) -> &'static str {
            "Temporal Blur"
        }
    
        fn description(&self) -> String {
            format!(
                "Blur temporal frequencies with radius {} and strength {:.2}",
                self.radius, self.strength
            )
        }
    }
    
    /// Apply blur to a single spectrum
    fn blur_spectrum(
        spectrum: &mut Vec<Complex64>,
        radius: usize,
        strength: f64,
        fft_size: usize,
    ) {
        if radius == 0 || strength == 0.0 {
            return; // No blur needed
        }
    
        // Create a copy for reading original values
        let original = spectrum.clone();
        
        // Apply blur to each bin
        for i in 0..fft_size {
            let mut sum = Complex64::new(0.0, 0.0);
            let mut count = 0;
            
            // Average with neighboring bins
            for offset in 0..=radius {
                // Add bins on both sides
                if offset == 0 {
                    sum += original[i];
                    count += 1;
                } else {
                    // Positive offset
                    if i + offset < fft_size {
                        sum += original[i + offset];
                        count += 1;
                    }
                    // Negative offset
                    if i >= offset {
                        sum += original[i - offset];
                        count += 1;
                    }
                }
            }
            
            // Apply averaged value with strength
            if count > 0 {
                let averaged = sum / count as f64;
                spectrum[i] = original[i] * (1.0 - strength) + averaged * strength;
            }
        }
        
        // Ensure DC and Nyquist remain real
        spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
        if fft_size % 2 == 0 {
            spectrum[fft_size / 2] = Complex64::new(spectrum[fft_size / 2].re, 0.0);
        }
    }

### Step 2: Add to Module Exports

Update `lib/src/temporal_fft/operations/mod.rs`:

rust

    //! Temporal FFT operations module
    
    mod basic;
    mod shift;
    mod stretch;
    mod phase_multiply;
    mod blur;  // Add this
    mod utils;
    
    // Re-export operations
    pub use basic::{ZeroTemporalDC, TemporalHighpass, TemporalLowpass};
    pub use shift::{TemporalShift, CircularTemporalShift, DispersiveTemporalShift};
    pub use stretch::TemporalStretch;
    pub use phase_multiply::{
        TemporalPhaseMultiply, DispersiveTemporalPhaseMultiply,
        TemporalPhaseReversal, TemporalPhaseScrambling,
    };
    pub use blur::TemporalBlur;  // Add this
    
    // ... rest of the file ...
    
    // Add convenience method to TemporalFFTAnalysis
    impl TemporalFFTAnalysis {
        // ... existing methods ...
        
        /// Apply temporal blur
        pub fn blur(&mut self, radius: usize, strength: f64) {
            TemporalBlur::new(radius, strength).apply(self);
        }
    }

### Step 3: Export from Temporal FFT Module

Update `lib/src/temporal_fft/mod.rs`:

rust

    // Re-export operations for convenience
    pub use operations::{
        TemporalOperation,
        // Basic operations
        ZeroTemporalDC, TemporalHighpass, TemporalLowpass,
        // Shift operations  
        TemporalShift, CircularTemporalShift, DispersiveTemporalShift,
        // Stretch operations
        TemporalStretch,
        // Phase operations
        TemporalPhaseMultiply, DispersiveTemporalPhaseMultiply,
        TemporalPhaseReversal, TemporalPhaseScrambling,
        // Blur operation
        TemporalBlur,  // Add this
    };

### Step 4: Add Processor Integration

Update `lib/src/processor.rs` to add a convenience method:

rust

    impl STFTProcessor {
        // ... existing methods ...
        
        /// Apply temporal blur
        pub fn apply_temporal_blur(&mut self, radius: usize, strength: f64) -> Result<()> {
            if let Some(temporal_fft) = &mut self.temporal_fft {
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
    }

### Step 5: Add CLI Support

Update the temporal\_process match statement in `cli/src/main.rs`:

rust

    "temporal_blur" => {
        if parts.len() < 4 {
            println!("Usage: temporal_process temporal_blur <radius> <strength>");
            println!("  radius: Number of adjacent bins to average (e.g., 1, 2, 3)");
            println!("  strength: Blur strength (0.0 = none, 1.0 = full)");
            println!("Examples:");
            println!("  temporal_process temporal_blur 1 0.5    # Subtle smoothing");
            println!("  temporal_process temporal_blur 3 0.8    # Strong smoothing");
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
                println!("Applied temporal blur with radius {} and strength {}", radius, strength);
            }
            Err(e) => println!("Error applying temporal blur: {}", e),
        }
    }

Also update the help text in the `temporal_process` section:

rust

    println!("  temporal_blur <radius> <strength> - Smooth temporal frequencies");

Best Practices
--------------

### 1\. Parameter Validation

Always validate parameters in your operation:

rust

    impl TemporalOperation for MyOperation {
        fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
            // Validate parameters
            if self.parameter < 0.0 {
                log::warn!("Parameter is negative, clamping to 0.0");
                let parameter = 0.0;
            }
            
            // Proceed with operation
        }
    }

### 2\. Preserve FFT Properties

Always maintain the mathematical properties of the FFT:

rust

    // DC component must be real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);
    
    // Nyquist component must be real (for even-sized FFT)
    if fft_size % 2 == 0 {
        spectrum[fft_size / 2] = Complex64::new(spectrum[fft_size / 2].re, 0.0);
    }

### 3\. Use Utility Functions

Leverage the provided utility functions:

rust

    use super::{process_all_temporal_spectra, enforce_hermitian_symmetry};
    
    // Process all spectra easily
    process_all_temporal_spectra(analysis, |ch_idx, bin_idx, spectrum| {
        // Your processing here
    });
    
    // Ensure proper symmetry after modifications
    enforce_hermitian_symmetry(spectrum);

### 4\. Logging

Add appropriate logging for debugging:

rust

    log::info!("Starting operation with parameters: {}", params);
    log::debug!("Processing bin {} of channel {}", bin_idx, ch_idx);
    log::warn!("Unusual condition detected: {}", condition);

### 5\. Documentation

Document your operation thoroughly:

rust

    /// Temporal echo effect
    /// 
    /// This operation adds delayed copies of the temporal spectrum to create
    /// an echo effect in the temporal domain. The echo decays exponentially
    /// with each repetition.
    /// 
    /// # Parameters
    /// 
    /// * `delay` - Delay in temporal frequency bins
    /// * `decay` - Decay factor for each echo (0.0 to 1.0)
    /// * `repeats` - Number of echo repetitions
    pub struct TemporalEcho {
        delay: usize,
        decay: f64,
        repeats: usize,
    }

Advanced Example: Frequency-Dependent Operation
-----------------------------------------------

Here's a more complex example that processes different frequency bins differently:

rust

    //! Temporal formant shift - shifts formant structure in temporal domain
    
    use super::{TemporalOperation, enforce_hermitian_symmetry};
    use crate::temporal_fft::TemporalFFTAnalysis;
    use num_complex::Complex64;
    
    pub struct TemporalFormantShift {
        shift_factor: f64,
        preserve_pitch: bool,
    }
    
    impl TemporalFormantShift {
        pub fn new(shift_factor: f64, preserve_pitch: bool) -> Self {
            Self { shift_factor, preserve_pitch }
        }
    }
    
    impl TemporalOperation for TemporalFormantShift {
        fn apply(&self, analysis: &mut TemporalFFTAnalysis) {
            let num_freq_bins = analysis.num_frequency_bins;
            
            // Process each frequency bin differently based on its position
            for bin_fft in analysis.bin_ffts.iter_mut() {
                // Calculate frequency-dependent shift
                let freq_ratio = bin_fft.bin_index as f64 / num_freq_bins as f64;
                
                // Apply different processing to different frequency ranges
                if freq_ratio < 0.1 {
                    // Low frequencies: minimal shift
                    apply_formant_shift(&mut bin_fft.temporal_spectrum, self.shift_factor * 0.2);
                } else if freq_ratio < 0.5 {
                    // Mid frequencies: full shift
                    apply_formant_shift(&mut bin_fft.temporal_spectrum, self.shift_factor);
                } else {
                    // High frequencies: reduced shift
                    apply_formant_shift(&mut bin_fft.temporal_spectrum, self.shift_factor * 0.5);
                }
                
                if self.preserve_pitch {
                    // Additional processing to preserve pitch
                    preserve_fundamental(&mut bin_fft.temporal_spectrum, bin_fft.bin_index);
                }
            }
        }
        
        fn name(&self) -> &'static str {
            "Temporal Formant Shift"
        }
        
        fn description(&self) -> String {
            format!(
                "Shift formants by factor {} {}",
                self.shift_factor,
                if self.preserve_pitch { "with pitch preservation" } else { "" }
            )
        }
    }

Testing Your Operation
----------------------

### Unit Tests

Add tests in your operation file:

rust

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::temporal_fft::{TemporalFFTConfig, TemporalFFTAnalyzer};
        use crate::stft::STFTFrame;
        
        fn create_test_analysis() -> TemporalFFTAnalysis {
            // Create test STFT frames
            let frames = vec![vec![STFTFrame {
                spectrum: vec![Complex64::new(1.0, 0.0); 32],
                frame_index: 0,
                time_position: 0,
            }; 16]];
            
            let config = TemporalFFTConfig::new(16).unwrap();
            let analyzer = TemporalFFTAnalyzer::new(config);
            analyzer.analyze(&frames).unwrap()
        }
        
        #[test]
        fn test_temporal_blur() {
            let mut analysis = create_test_analysis();
            let blur = TemporalBlur::new(1, 0.5);
            
            // Store original for comparison
            let original_spectrum = analysis.bin_ffts[0].temporal_spectrum.clone();
            
            // Apply blur
            blur.apply(&mut analysis);
            
            // Verify changes were made
            let blurred_spectrum = &analysis.bin_ffts[0].temporal_spectrum;
            assert_ne!(original_spectrum, *blurred_spectrum);
            
            // Verify DC is still real
            assert_eq!(blurred_spectrum[0].im, 0.0);
        }
    }

### Integration Tests

Test the full pipeline:

rust

    #[test]
    fn test_blur_integration() {
        // Create processor
        let mut processor = STFTProcessor::new();
        
        // Generate test audio
        let (info, audio) = generate_test_sine(44100, 1.0, 440.0);
        
        // Full pipeline
        processor.load_audio(info, audio).unwrap();
        processor.analyze().unwrap();
        processor.analyze_temporal().unwrap();
        processor.apply_temporal_blur(2, 0.7).unwrap();
        processor.synthesize_from_temporal().unwrap();
        
        let result = processor.synthesize().unwrap();
        assert!(!result.is_empty());
    }

Performance Considerations
--------------------------

### 1\. Avoid Repeated Allocations

rust

    // Bad: Allocates on every call
    fn process_spectrum(spectrum: &mut Vec<Complex64>) {
        let temp = vec![Complex64::new(0.0, 0.0); spectrum.len()]; // Avoid this
    }
    
    // Good: Reuse existing memory
    fn process_spectrum(spectrum: &mut Vec<Complex64>) {
        // Work in-place when possible
        for val in spectrum.iter_mut() {
            *val *= 0.5;
        }
    }

### 2\. Use Iterator Methods

rust

    // More idiomatic and often faster
    let sum: Complex64 = spectrum.iter()
        .skip(1)
        .take(10)
        .sum();

### 3\. Consider Parallelization

For operations that process bins independently:

rust

    use rayon::prelude::*;
    
    // Process bins in parallel
    analysis.bin_ffts.par_iter_mut()
        .for_each(|bin_fft| {
            process_bin(&mut bin_fft.temporal_spectrum);
        });

Common Patterns
---------------

### 1\. Windowed Operations

rust

    /// Apply a window function to temporal spectrum
    fn apply_temporal_window(spectrum: &mut Vec<Complex64>, window_type: WindowType) {
        let window = generate_window(window_type, spectrum.len());
        for (i, val) in spectrum.iter_mut().enumerate() {
            *val *= window[i];
        }
    }

### 2\. Threshold Operations

rust

    /// Zero out components below threshold
    fn apply_threshold(spectrum: &mut Vec<Complex64>, threshold: f64) {
        for val in spectrum.iter_mut() {
            if val.norm() < threshold {
                *val = Complex64::new(0.0, 0.0);
            }
        }
    }

### 3\. Morphological Operations

rust

    /// Find peaks in temporal spectrum
    fn find_temporal_peaks(spectrum: &[Complex64], window_size: usize) -> Vec<usize> {
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let mut peaks = Vec::new();
        
        for i in window_size..magnitudes.len() - window_size {
            let is_peak = magnitudes[i-window_size..=i+window_size]
                .iter()
                .all(|&m| m <= magnitudes[i]);
                
            if is_peak {
                peaks.push(i);
            }
        }
        
        peaks
    }

Debugging Tips
--------------

1.  **Visualize Your Operation**: Use the temporal spectrogram visualization to see the effects:
    
    bash
    
        temporal_spectrogram before.png
        temporal_process your_operation
        temporal_spectrogram after.png
    
2.  **Add Debug Output**: Temporarily add debug prints:
    
    rust
    
        log::debug!("Before: {:?}", spectrum[0..5].to_vec());
        // Your operation
        log::debug!("After: {:?}", spectrum[0..5].to_vec());
    
3.  **Test with Simple Signals**: Start with sine waves or impulses to understand the effect.
4.  **Check Phase Statistics**: Use the built-in phase statistics:
    
    bash
    
        temporal_process temporal_phase_stats
    

Summary
-------

Creating a new temporal FFT operation involves:

1.  Creating a struct that implements `TemporalOperation`
2.  Adding it to the module exports
3.  Creating a convenience method on `TemporalFFTAnalysis`
4.  (Optional) Adding a processor method
5.  Adding CLI support
6.  Writing tests

The trait-based system makes it easy to add new operations while maintaining consistency across the codebase. Always remember to preserve FFT properties and provide good documentation for your operations.