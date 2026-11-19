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
    // Transducers: rank results by score divided by output length, as
    // modified_beam_search does.
    private final boolean lengthNormalize;
    // Stateless-decoder transducers: search as modified_beam_search does, so
    // that its results are reproduced, with or without scorers.
    private final boolean modifiedBeamSearchRules;
    // CTC models: a text file of "token-id log-prior" lines. Each listed
    // token other than blank has ctcPriorWeight times its log prior
    // subtracted from its log probability in every frame.
    private final String ctcPrior;
    private final float ctcPriorWeight;
    // Offline transducers: log at the end what pruning each result needed
    // and what cutting the lists waiting for later frames would cost.
    private final boolean traceSurvival;
    // TDT: at every round's end also cut each list of arrivals
    // waiting past the next frame to its best waitingMaxActive within
    // waitingScoreBeam of its best (0: twice the max active paths, half of
    // scoreBeam).
    private final boolean pruneWaiting;
    private final int waitingMaxActive;
    private final float waitingScoreBeam;

    private FusionConfig(Builder builder) {
        this.scorers = builder.scorers;
        this.lengthBonus = builder.lengthBonus;
        this.lengthBonusUnit = builder.lengthBonusUnit;
        this.ctcAlignment = builder.ctcAlignment;
        this.tokenCandidates = builder.tokenCandidates;
        this.scoreBeam = builder.scoreBeam;
        this.scorerTextCase = builder.scorerTextCase;
        this.lengthNormalize = builder.lengthNormalize;
        this.modifiedBeamSearchRules = builder.modifiedBeamSearchRules;
        this.ctcPrior = builder.ctcPrior;
        this.ctcPriorWeight = builder.ctcPriorWeight;
        this.traceSurvival = builder.traceSurvival;
        this.pruneWaiting = builder.pruneWaiting;
        this.waitingMaxActive = builder.waitingMaxActive;
        this.waitingScoreBeam = builder.waitingScoreBeam;
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

    public boolean getLengthNormalize() {
        return lengthNormalize;
    }

    public boolean getModifiedBeamSearchRules() {
        return modifiedBeamSearchRules;
    }

    public String getCtcPrior() {
        return ctcPrior;
    }

    public float getCtcPriorWeight() {
        return ctcPriorWeight;
    }

    public boolean getTraceSurvival() {
        return traceSurvival;
    }

    public boolean getPruneWaiting() {
        return pruneWaiting;
    }

    public int getWaitingMaxActive() {
        return waitingMaxActive;
    }

    public float getWaitingScoreBeam() {
        return waitingScoreBeam;
    }

    public static class Builder {
        private ScorerConfig[] scorers = new ScorerConfig[0];
        private float lengthBonus = 0.0f;
        private String lengthBonusUnit = "token";
        private String ctcAlignment = "sum";
        private int tokenCandidates = 8;
        private float scoreBeam = 12.0f;
        private String scorerTextCase = "keep";
        private boolean lengthNormalize = false;
        private boolean modifiedBeamSearchRules = false;
        private String ctcPrior = "";
        private float ctcPriorWeight = 1.0f;
        private boolean traceSurvival = false;
        private boolean pruneWaiting = true;
        private int waitingMaxActive = 0;
        private float waitingScoreBeam = 0.0f;

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

        public Builder setLengthNormalize(boolean lengthNormalize) {
            this.lengthNormalize = lengthNormalize;
            return this;
        }

        public Builder setModifiedBeamSearchRules(boolean modifiedBeamSearchRules) {
            this.modifiedBeamSearchRules = modifiedBeamSearchRules;
            return this;
        }

        public Builder setCtcPrior(String ctcPrior) {
            this.ctcPrior = ctcPrior;
            return this;
        }

        public Builder setCtcPriorWeight(float ctcPriorWeight) {
            this.ctcPriorWeight = ctcPriorWeight;
            return this;
        }

        public Builder setTraceSurvival(boolean traceSurvival) {
            this.traceSurvival = traceSurvival;
            return this;
        }

        public Builder setPruneWaiting(boolean pruneWaiting) {
            this.pruneWaiting = pruneWaiting;
            return this;
        }

        public Builder setWaitingMaxActive(int waitingMaxActive) {
            this.waitingMaxActive = waitingMaxActive;
            return this;
        }

        public Builder setWaitingScoreBeam(float waitingScoreBeam) {
            this.waitingScoreBeam = waitingScoreBeam;
            return this;
        }
    }
}
