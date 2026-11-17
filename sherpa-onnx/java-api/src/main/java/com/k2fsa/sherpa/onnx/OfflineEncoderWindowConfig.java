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
    // Exponent of the "center" weight, (1 + distance from the nearer
    // retained edge)^centerPower: 1 is linear, 0 uniform.
    private final float centerPower;
    // Context-only output frames per interior window edge; they inform the
    // window's other frames but are not themselves kept.
    private final int contextFrames;
    // The same as a fraction of each window's output length. At most one of
    // contextFrames and contextFraction may be non-zero.
    private final float contextFraction;

    private OfflineEncoderWindowConfig(Builder builder) {
        this.seconds = builder.seconds;
        this.overlap = builder.overlap;
        this.slack = builder.slack;
        this.normalize = builder.normalize;
        this.weight = builder.weight;
        this.blend = builder.blend;
        this.centerPower = builder.centerPower;
        this.contextFrames = builder.contextFrames;
        this.contextFraction = builder.contextFraction;
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

    public float getCenterPower() {
        return centerPower;
    }

    public int getContextFrames() {
        return contextFrames;
    }

    public float getContextFraction() {
        return contextFraction;
    }

    public static class Builder {
        private float seconds = 0.0f;
        private float overlap = 3.0f;
        private float slack = 1.0f;
        private String normalize = "window";
        private String weight = "center";
        private String blend = "log";
        private float centerPower = 1.0f;
        private int contextFrames = 0;
        private float contextFraction = 0.0f;

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

        public Builder setCenterPower(float centerPower) {
            this.centerPower = centerPower;
            return this;
        }

        public Builder setContextFrames(int contextFrames) {
            this.contextFrames = contextFrames;
            return this;
        }

        public Builder setContextFraction(float contextFraction) {
            this.contextFraction = contextFraction;
            return this;
        }
    }
}
