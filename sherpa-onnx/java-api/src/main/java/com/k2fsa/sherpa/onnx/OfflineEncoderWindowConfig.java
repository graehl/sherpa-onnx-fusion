// Copyright 2026 graehl

package com.k2fsa.sherpa.onnx;

// How an offline encoder runs an utterance longer than a window: in
// overlapping windows whose outputs are blended per output frame. Supported
// for the offline CTC models, the zipformer transducer, and NeMo RNN-T and
// TDT.
public class OfflineEncoderWindowConfig {
    private final float seconds;  // window length; 0: encode whole
    private final float overlap;  // least overlap of neighbouring windows
    // How far an interior window may move, in seconds, to put its edges
    // where the audio is quietest.
    private final float slack;
    private final String normalize;  // "window" or "utterance"
    // "center": a frame is the weighted average of the windows that cover
    // it; "cut": it is taken from the window in which it is farthest from
    // an edge.
    private final String weight;
    // CTC models: what "center" averages, "log" probabilities or "prob".
    private final String blend;

    private OfflineEncoderWindowConfig(Builder builder) {
        this.seconds = builder.seconds;
        this.overlap = builder.overlap;
        this.slack = builder.slack;
        this.normalize = builder.normalize;
        this.weight = builder.weight;
        this.blend = builder.blend;
    }

    public static Builder builder() {
        return new Builder();
    }

    public float getSeconds() {
        return seconds;
    }

    public float getOverlap() {
        return overlap;
    }

    public float getSlack() {
        return slack;
    }

    public String getNormalize() {
        return normalize;
    }

    public String getWeight() {
        return weight;
    }

    public String getBlend() {
        return blend;
    }

    public static class Builder {
        private float seconds = 0.0f;
        private float overlap = 3.0f;
        private float slack = 1.0f;
        private String normalize = "window";
        private String weight = "center";
        private String blend = "log";

        public OfflineEncoderWindowConfig build() {
            return new OfflineEncoderWindowConfig(this);
        }

        public Builder setSeconds(float seconds) {
            this.seconds = seconds;
            return this;
        }

        public Builder setOverlap(float overlap) {
            this.overlap = overlap;
            return this;
        }

        public Builder setSlack(float slack) {
            this.slack = slack;
            return this;
        }

        public Builder setNormalize(String normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder setWeight(String weight) {
            this.weight = weight;
            return this;
        }

        public Builder setBlend(String blend) {
            this.blend = blend;
            return this;
        }
    }
}
