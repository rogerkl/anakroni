//! Statistical analysis for temporal FFT data

use crate::temporal_fft::TemporalFFTAnalysis;

/// Statistics about phase distribution in temporal FFT data
#[derive(Debug, Clone)]
pub struct PhaseStatistics {
    pub mean_phase: f64,
    pub phase_variance: f64,
    pub min_phase: f64,
    pub max_phase: f64,
    pub num_samples: usize,
}

impl std::fmt::Display for PhaseStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Phase Statistics: mean={:.3}rad, variance={:.3}, range=[{:.3}, {:.3}]rad, samples={}",
            self.mean_phase, self.phase_variance, self.min_phase, self.max_phase, self.num_samples
        )
    }
}

/// General statistics interface for temporal FFT analysis
pub trait TemporalStatistics {
    /// Get phase statistics across all temporal spectra
    fn get_phase_statistics(&self) -> PhaseStatistics;
}

impl TemporalStatistics for TemporalFFTAnalysis {
    /// Get phase statistics for debugging and analysis
    ///
    /// Returns information about phase distribution across all temporal spectra.
    fn get_phase_statistics(&self) -> PhaseStatistics {
        let mut all_phases = Vec::new();

        for bin_fft in &self.bin_ffts {
            for &complex_val in &bin_fft.temporal_spectrum {
                if complex_val.norm() > 1e-12 {
                    // Skip near-zero components
                    all_phases.push(complex_val.arg());
                }
            }
        }

        if all_phases.is_empty() {
            return PhaseStatistics {
                mean_phase: 0.0,
                phase_variance: 0.0,
                min_phase: 0.0,
                max_phase: 0.0,
                num_samples: 0,
            };
        }

        all_phases.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = all_phases.iter().sum::<f64>() / all_phases.len() as f64;
        let variance = all_phases
            .iter()
            .map(|&phase| (phase - mean).powi(2))
            .sum::<f64>()
            / all_phases.len() as f64;

        PhaseStatistics {
            mean_phase: mean,
            phase_variance: variance,
            min_phase: all_phases[0],
            max_phase: all_phases[all_phases.len() - 1],
            num_samples: all_phases.len(),
        }
    }
}
