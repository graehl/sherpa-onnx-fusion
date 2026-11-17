#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_float};

/// `SherpaOnnxScorerConfig`: one scorer fused into `prefix_beam_search`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ScorerConfig {
    pub type_: *const c_char,
    pub model: *const c_char,
    pub unit: *const c_char,
    pub space: *const c_char,
    pub num_threads: i32,
    pub provider: *const c_char,
    pub weight: c_float,
    pub extra: *const c_char,
    pub rescore: i32,
}

/// `SherpaOnnxFusionConfig`: settings for `prefix_beam_search`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FusionConfig {
    pub scorers: *const ScorerConfig,
    pub num_scorers: i32,
    pub length_bonus: c_float,
    pub length_bonus_unit: *const c_char,
    pub ctc_alignment: *const c_char,
    pub token_candidates: i32,
    pub score_beam: c_float,
    pub scorer_text_case: *const c_char,
    pub length_normalize: i32,
    pub modified_beam_search_rules: i32,
    pub ctc_prior: *const c_char,
    pub ctc_prior_weight: c_float,
}
