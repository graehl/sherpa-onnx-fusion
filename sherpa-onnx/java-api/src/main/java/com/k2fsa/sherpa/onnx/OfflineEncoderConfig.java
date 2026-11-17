// Copyright 2026 graehl

package com.k2fsa.sherpa.onnx;

// How an offline model's encoder runs.
public class OfflineEncoderConfig {
    private final OfflineEncoderWindowConfig window;
    // Comma-separated input lengths in seconds an encoder call may have: an
    // input is padded to the shortest that holds it. Needs a model that
    // takes a padded batch.
    private final String lengths;
    // Else pad an input to a multiple of this many seconds. 0: no padding.
    private final float padTo;

    private OfflineEncoderConfig(Builder builder) {
        this.window = builder.window;
        this.lengths = builder.lengths;
        this.padTo = builder.padTo;
    }

    public static Builder builder() {
        return new Builder();
    }

    public OfflineEncoderWindowConfig getWindow() {
        return window;
    }

    public String getLengths() {
        return lengths;
    }

    public float getPadTo() {
        return padTo;
    }

    public static class Builder {
        private OfflineEncoderWindowConfig window = OfflineEncoderWindowConfig.builder().build();
        private String lengths = "";
        private float padTo = 0.0f;

        public OfflineEncoderConfig build() {
            return new OfflineEncoderConfig(this);
        }

        public Builder setWindow(OfflineEncoderWindowConfig window) {
            this.window = window;
            return this;
        }

        public Builder setLengths(String lengths) {
            this.lengths = lengths;
            return this;
        }

        public Builder setPadTo(float padTo) {
            this.padTo = padTo;
            return this;
        }
    }
}
