//! Scorers, such as language models, fused into `prefix_beam_search`.
//!
//! Set [`FusionConfig`] as the `fusion` field of an offline or online
//! recognizer config and choose `decoding_method = "prefix_beam_search"`.
//!
//! ```no_run
//! use sherpa_onnx::{OfflineRecognizerConfig, ScorerConfig};
//!
//! let mut config = OfflineRecognizerConfig::default();
//! config.decoding_method = Some("prefix_beam_search".into());
//! config.fusion.scorers.push(ScorerConfig {
//!     r#type: Some("kenlm".into()),
//!     model: Some("./lm.bin".into()),
//!     weight: 0.6,
//!     ..Default::default()
//! });
//! ```

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::CString;

#[derive(Clone, Debug)]
/// One scorer, such as a language model, fused into `prefix_beam_search`.
///
/// For a value this cannot express, such as a weight of 0, use the
/// recognizer config's `options_json`.
pub struct ScorerConfig {
    /// Registered scorer type: `kenlm` or `onnx-rnn`.
    pub r#type: Option<String>,
    /// Path to the scorer's model.
    pub model: Option<String>,
    /// What the scorer reads: `char` or `token`.
    pub unit: Option<String>,
    /// The scorer's word-boundary unit for character scorers.
    pub space: Option<String>,
    /// Threads for the scorer's own model.
    pub num_threads: i32,
    /// Execution provider for the scorer's model.
    pub provider: Option<String>,
    /// Search weight of the scorer's log score. 0 means 1.
    pub weight: f32,
    /// JSON object of type-specific options.
    pub extra: Option<String>,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            r#type: None,
            model: None,
            unit: Some("char".into()),
            space: Some("▁".into()),
            num_threads: 1,
            provider: Some("cpu".into()),
            weight: 1.0,
            extra: Some("{}".into()),
        }
    }
}

impl ScorerConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::ScorerConfig {
        sys::ScorerConfig {
            type_: to_c_ptr(&self.r#type, cstrings),
            model: to_c_ptr(&self.model, cstrings),
            unit: to_c_ptr(&self.unit, cstrings),
            space: to_c_ptr(&self.space, cstrings),
            num_threads: self.num_threads,
            provider: to_c_ptr(&self.provider, cstrings),
            weight: self.weight,
            extra: to_c_ptr(&self.extra, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// Scorers and search settings for `prefix_beam_search`.
///
/// The default searches with no scorer.
pub struct FusionConfig {
    /// Scorers fused with independent weights.
    pub scorers: Vec<ScorerConfig>,
    /// Added to a hypothesis's score per `length_bonus_unit` of output.
    pub length_bonus: f32,
    /// What `length_bonus` counts: `token` or `word`.
    pub length_bonus_unit: Option<String>,
    /// CTC models: score an output by `sum` or `max` alignment.
    pub ctc_alignment: Option<String>,
    /// Most probable tokens per frame that may extend a hypothesis.
    pub token_candidates: i32,
    /// A hypothesis survives a frame only within this score of the best.
    /// Negative: no limit.
    pub score_beam: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            scorers: Vec::new(),
            length_bonus: 0.0,
            length_bonus_unit: Some("token".into()),
            ctc_alignment: Some("sum".into()),
            token_candidates: 8,
            score_beam: 12.0,
        }
    }
}

impl FusionConfig {
    /// Convert to the sys struct. Its scorer array is pushed onto `scorers`,
    /// which, like `cstrings`, must outlive the returned struct.
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
        scorers: &mut Vec<Vec<sys::ScorerConfig>>,
    ) -> sys::FusionConfig {
        let array: Vec<sys::ScorerConfig> = self
            .scorers
            .iter()
            .map(|s| s.to_sys(cstrings))
            .collect();
        let ptr = if array.is_empty() {
            std::ptr::null()
        } else {
            array.as_ptr()
        };
        let num_scorers = array.len() as i32;
        // Moving the Vec into `scorers` does not move its heap buffer.
        scorers.push(array);
        sys::FusionConfig {
            scorers: ptr,
            num_scorers,
            length_bonus: self.length_bonus,
            length_bonus_unit: to_c_ptr(&self.length_bonus_unit, cstrings),
            ctc_alignment: to_c_ptr(&self.ctc_alignment, cstrings),
            token_candidates: self.token_candidates,
            score_beam: self.score_beam,
        }
    }
}
