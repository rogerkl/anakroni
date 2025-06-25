//! Crossfade functionality for temporal_split operations
//!
//! Provides smooth transitions between parts by applying linear crossfade
//! weights to overlapping bins instead of hard boundaries.

use num_complex::Complex64;
use std::collections::HashMap;

/// Represents the amplitude for a bin in a part
#[derive(Debug, Clone)]
pub struct BinAmplitude {
    pub bin: usize,
    pub amplitude: f64,
}

/// Represents the amplitude for a bin in a part
#[derive(Debug, Clone)]
pub struct PartBinAmplitudes {
    pub part: usize,
    pub bin_amplitudes: Vec<BinAmplitude>,
}

/// Represents the boundaries for a fade in our out
#[derive(Debug, Clone)]
struct BinBoundaries {
    bin_start: usize,
    bin_end: usize,
}

impl BinBoundaries {
    fn bin_from_end(&self, bin_count: usize, factor: f64) -> usize {
        1 + self.bin_end - (bin_count as f64 * factor) as usize
    }

    fn bin_from_start(&self, bin_count: usize, factor: f64) -> usize {
        self.bin_start + (bin_count as f64 * factor) as usize - 1
    }
}

/// Represents the boundaries for the fades for a part
#[derive(Debug, Clone)]
struct FadeBoundariesPart {
    //part
    part: usize,
    // bins belonging to this part
    part_bins: BinBoundaries,

    // TODO names are probably not self-explanatory here ...

    //bins from previous part that will be faded out 0->0.5
    fade_out_start: Option<BinBoundaries>,
    //bins from our part that will be faded in 0.5->1
    fade_in_start: Option<BinBoundaries>,
    //bins from our part that will be faded out 1->0.5
    fade_out_end: Option<BinBoundaries>,
    //bins from next part that will be faded in 0.5->0
    fade_in_end: Option<BinBoundaries>,
}

impl FadeBoundariesPart {
    fn bin_count(&self) -> usize {
        (self.part_bins.bin_end - self.part_bins.bin_start) + 1
    }
    fn bin_amplitudes(&self) -> Vec<BinAmplitude> {
        let mut bin_amplitudes = Vec::new();
        let mut current_bin_start = None;
        match &self.fade_out_start {
            Some(fos) => {
                //0->0.5
                let count = ((fos.bin_end + 1) - fos.bin_start);
                let factor = 0.5 / count as f64;
                for (counter, bin) in (fos.bin_start..=fos.bin_end).enumerate() {
                    bin_amplitudes.push(BinAmplitude {
                        bin,
                        amplitude: counter as f64 * factor,
                    });
                    current_bin_start = Some(bin);
                }
                match &self.fade_in_start {
                    Some(fis) => {
                        //0.5->1
                        let count = ((fis.bin_end + 1) - fis.bin_start);
                        let factor = 0.5 / count as f64;
                        for (counter, bin) in (fis.bin_start..=fis.bin_end).enumerate() {
                            bin_amplitudes.push(BinAmplitude {
                                bin,
                                amplitude: 0.5 + (counter as f64 * factor),
                            });
                            current_bin_start = Some(bin);
                        }
                    }
                    None => {}
                }
            }
            None => {}
        }
        let middle_start_bin = match &current_bin_start {
            Some(s) => *s + 1,
            None => self.part_bins.bin_start,
        };

        match &self.fade_out_end {
            Some(foe) => {
                //1->0.5
                for bin in middle_start_bin..foe.bin_start {
                    bin_amplitudes.push(BinAmplitude { bin, amplitude: 1. });
                }

                let count = ((foe.bin_end + 1) - foe.bin_start);
                let factor = 0.5 / count as f64;
                for (counter, bin) in (foe.bin_start..=foe.bin_end).enumerate() {
                    bin_amplitudes.push(BinAmplitude {
                        bin,
                        amplitude: 0.5 + ((count - counter) as f64 * factor),
                    });
                    current_bin_start = Some(bin);
                }
                match &self.fade_in_end {
                    //0.5->0
                    Some(fie) => {
                        //0->0.5
                        let count = ((fie.bin_end + 1) - fie.bin_start);
                        let factor = 0.5 / count as f64;
                        for (counter, bin) in (fie.bin_start..=fie.bin_end).enumerate() {
                            bin_amplitudes.push(BinAmplitude {
                                bin,
                                amplitude: (count - counter) as f64 * factor,
                            });
                            current_bin_start = Some(bin);
                        }
                    }
                    None => {}
                }
            }
            None => {
                for bin in middle_start_bin..=self.part_bins.bin_end {
                    bin_amplitudes.push(BinAmplitude { bin, amplitude: 1. });
                }
            }
        }
        bin_amplitudes
    }
}

/// Applies crossfade to a hard distribution, creating smooth transitions between parts
pub fn apply_crossfade_to_distribution(
    hard_assignments: Vec<usize>,
    crossfade_factor: f64,
    num_parts: usize,
) -> Vec<PartBinAmplitudes> {
    let num_bins = hard_assignments.len();

    // Validate crossfade factor
    let crossfade_factor = crossfade_factor.clamp(0.0, 0.5);

    let mut part_boundaries: Vec<FadeBoundariesPart> = Vec::new();
    let mut vec_index: Option<usize> = None;

    for (bin_idx, &part_idx) in hard_assignments.iter().enumerate() {
        if let Some(mut index) = &vec_index {
            if part_boundaries.get(index).unwrap().part != part_idx {
                part_boundaries.push(FadeBoundariesPart {
                    part: part_idx,
                    part_bins: BinBoundaries {
                        bin_start: bin_idx,
                        bin_end: bin_idx,
                    },
                    fade_in_start: None,
                    fade_out_start: None,
                    fade_in_end: None,
                    fade_out_end: None,
                });
                vec_index = Some(index + 1);
            } else {
                part_boundaries.get_mut(index).unwrap().part_bins.bin_end = bin_idx;
            }
        } else {
            part_boundaries.push(FadeBoundariesPart {
                part: part_idx,
                part_bins: BinBoundaries {
                    bin_start: bin_idx,
                    bin_end: bin_idx,
                },
                fade_in_start: None,
                fade_out_start: None,
                fade_in_end: None,
                fade_out_end: None,
            });
            vec_index = Some(0);
        }
    }
    for index in 0..part_boundaries.len() {
        // First, collect data from adjacent parts
        let prev_part_data = if index > 0 {
            part_boundaries.get(index - 1).map(|p| {
                let fade_out_start_start =
                    p.part_bins.bin_from_end(p.bin_count(), crossfade_factor);
                (fade_out_start_start, p.part_bins.bin_end)
            })
        } else {
            None
        };

        let next_part_data = part_boundaries.get(index + 1).map(|p| {
            let fade_in_end_end = p.part_bins.bin_from_start(p.bin_count(), crossfade_factor);
            (p.part_bins.bin_start, fade_in_end_end)
        });

        // Now get mutable access and update
        if (crossfade_factor > 0.0) {
            if let Some(fade_boundaries_part) = part_boundaries.get_mut(index) {
                // Process previous part fade
                if let Some((fade_out_start_start, prev_bin_end)) = prev_part_data {
                    // Fade in for current part
                    let fade_in_start_end = fade_boundaries_part
                        .part_bins
                        .bin_from_start(fade_boundaries_part.bin_count(), crossfade_factor);
                    if fade_in_start_end > fade_boundaries_part.part_bins.bin_start {
                        fade_boundaries_part.fade_in_start = Some(BinBoundaries {
                            bin_start: fade_boundaries_part.part_bins.bin_start,
                            bin_end: fade_in_start_end,
                        });
                    }

                    // Fade out from previous part
                    if fade_out_start_start < prev_bin_end {
                        fade_boundaries_part.fade_out_start = Some(BinBoundaries {
                            bin_start: fade_out_start_start,
                            bin_end: prev_bin_end,
                        });
                    }
                }

                // Process next part fade
                if let Some((next_bin_start, fade_in_end_end)) = next_part_data {
                    // Fade in from next part
                    if fade_in_end_end > next_bin_start {
                        fade_boundaries_part.fade_in_end = Some(BinBoundaries {
                            bin_start: next_bin_start,
                            bin_end: fade_in_end_end,
                        });
                    }

                    // Fade out for current part
                    let fade_out_end_start = fade_boundaries_part
                        .part_bins
                        .bin_from_end(fade_boundaries_part.bin_count(), crossfade_factor);
                    if fade_out_end_start < fade_boundaries_part.part_bins.bin_end {
                        fade_boundaries_part.fade_out_end = Some(BinBoundaries {
                            bin_start: fade_out_end_start,
                            bin_end: fade_boundaries_part.part_bins.bin_end,
                        });
                    }
                }
            }
        }
    }
    log::info!("part_boundaries: {:?}", part_boundaries);
    part_boundaries
        .into_iter()
        .map(|fbp| PartBinAmplitudes {
            part: fbp.part,
            bin_amplitudes: fbp.bin_amplitudes(),
        })
        .collect()
}
