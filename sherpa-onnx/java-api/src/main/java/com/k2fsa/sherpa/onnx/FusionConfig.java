// Copyright 2026 graehl

package com.k2fsa.sherpa.onnx;

// Scorers and search settings for decodingMethod "prefix_beam_search".
public class FusionConfig {
    private final ScorerConfig[] scorers;
    private final float lengthBonus;
    private final String lengthBonusUnit;  // "token" or "word"
    private final String ctcAlignment;  // "sum" or "max"
    private final int tokenCandidates;
    // A hypothesis survives a frame only within this score of the best;
    // 0 or less: no limit.
    private final float scoreBeam;
    // Case of the text character scorers are given: "keep" or "lower".
    private final String scorerTextCase;

    private FusionConfig(Builder builder) {
        this.scorers = builder.scorers;
        this.lengthBonus = builder.lengthBonus;
        this.lengthBonusUnit = builder.lengthBonusUnit;
        this.ctcAlignment = builder.ctcAlignment;
        this.tokenCandidates = builder.tokenCandidates;
        this.scoreBeam = builder.scoreBeam;
        this.scorerTextCase = builder.scorerTextCase;
    }

    public static Builder builder() {
        return new Builder();
    }

    public ScorerConfig[] getScorers() {
        return scorers;
    }

    public float getLengthBonus() {
        return lengthBonus;
    }

    public String getLengthBonusUnit() {
        return lengthBonusUnit;
    }

    public String getCtcAlignment() {
        return ctcAlignment;
    }

    public int getTokenCandidates() {
        return tokenCandidates;
    }

    public float getScoreBeam() {
        return scoreBeam;
    }

    public String getScorerTextCase() {
        return scorerTextCase;
    }

    public static class Builder {
        private ScorerConfig[] scorers = new ScorerConfig[0];
        private float lengthBonus = 0.0f;
        private String lengthBonusUnit = "token";
        private String ctcAlignment = "sum";
        private int tokenCandidates = 8;
        private float scoreBeam = 12.0f;
        private String scorerTextCase = "keep";

        public FusionConfig build() {
            return new FusionConfig(this);
        }

        public Builder setScorers(ScorerConfig... scorers) {
            this.scorers = scorers;
            return this;
        }

        public Builder setLengthBonus(float lengthBonus) {
            this.lengthBonus = lengthBonus;
            return this;
        }

        public Builder setLengthBonusUnit(String lengthBonusUnit) {
            this.lengthBonusUnit = lengthBonusUnit;
            return this;
        }

        public Builder setCtcAlignment(String ctcAlignment) {
            this.ctcAlignment = ctcAlignment;
            return this;
        }

        public Builder setTokenCandidates(int tokenCandidates) {
            this.tokenCandidates = tokenCandidates;
            return this;
        }

        public Builder setScoreBeam(float scoreBeam) {
            this.scoreBeam = scoreBeam;
            return this;
        }

        public Builder setScorerTextCase(String scorerTextCase) {
            this.scorerTextCase = scorerTextCase;
            return this;
        }
    }
}
