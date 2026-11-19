// Copyright 2026 graehl

package com.k2fsa.sherpa.onnx;

// One scorer, such as a language model, fused into
// decodingMethod "prefix_beam_search".
public class ScorerConfig {
    private final String type;  // "kenlm" or "onnx-rnn"
    private final String model;
    private final String unit;  // "char" or "token"
    private final String space;
    private final int numThreads;
    private final String provider;
    private final float weight;
    private final String extra;  // JSON object of type-specific options
    // true: apply the scorer only to the hypotheses that survive the search,
    // as n-best rescoring does, instead of during the search.
    private final boolean rescore;
    // When the scorer is destroyed, log what it measured of its own work and
    // memory (onnx-rnn: its widest model step and peak memory).
    private final boolean stats;

    private ScorerConfig(Builder builder) {
        this.type = builder.type;
        this.model = builder.model;
        this.unit = builder.unit;
        this.space = builder.space;
        this.numThreads = builder.numThreads;
        this.provider = builder.provider;
        this.weight = builder.weight;
        this.extra = builder.extra;
        this.rescore = builder.rescore;
        this.stats = builder.stats;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getType() {
        return type;
    }

    public String getModel() {
        return model;
    }

    public String getUnit() {
        return unit;
    }

    public String getSpace() {
        return space;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public String getProvider() {
        return provider;
    }

    public float getWeight() {
        return weight;
    }

    public String getExtra() {
        return extra;
    }

    public boolean getRescore() {
        return rescore;
    }

    public boolean getStats() {
        return stats;
    }

    public static class Builder {
        private String type = "";
        private String model = "";
        private String unit = "char";
        private String space = "▁";
        private int numThreads = 1;
        private String provider = "cpu";
        private float weight = 1.0f;
        private String extra = "{}";
        private boolean rescore = false;
        private boolean stats = true;

        public ScorerConfig build() {
            return new ScorerConfig(this);
        }

        public Builder setType(String type) {
            this.type = type;
            return this;
        }

        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setUnit(String unit) {
            this.unit = unit;
            return this;
        }

        public Builder setSpace(String space) {
            this.space = space;
            return this;
        }

        public Builder setNumThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Builder setProvider(String provider) {
            this.provider = provider;
            return this;
        }

        public Builder setWeight(float weight) {
            this.weight = weight;
            return this;
        }

        public Builder setExtra(String extra) {
            this.extra = extra;
            return this;
        }

        public Builder setRescore(boolean rescore) {
            this.rescore = rescore;
            return this;
        }

        public Builder setStats(boolean stats) {
            this.stats = stats;
            return this;
        }
    }
}
