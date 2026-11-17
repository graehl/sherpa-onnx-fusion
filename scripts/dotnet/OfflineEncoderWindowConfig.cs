/// Copyright (c)  2026  graehl

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// How an offline encoder runs an utterance longer than a window: in
    /// overlapping windows whose outputs are blended per output frame.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineEncoderWindowConfig
    {
        public OfflineEncoderWindowConfig()
        {
            Seconds = 0.0F;
            Overlap = 3.0F;
            Slack = 1.0F;
            Normalize = "window";
            Weight = "center";
            Blend = "log";
        }

        /// <summary>
        /// Window length in seconds. 0: every utterance is encoded whole.
        /// </summary>
        public float Seconds;

        /// <summary>
        /// The least overlap of neighbouring windows, in seconds.
        /// </summary>
        public float Overlap;

        /// <summary>
        /// How far an interior window may move, in seconds, to put its edges
        /// where the audio is quietest.
        /// </summary>
        public float Slack;

        /// <summary>
        /// Scope of input normalization: "window" or "utterance".
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Normalize;

        /// <summary>
        /// "center": a frame is the weighted average of the windows that
        /// cover it; "cut": it is taken from the window in which it is
        /// farthest from an edge.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Weight;

        /// <summary>
        /// CTC models: what "center" averages, "log" probabilities or
        /// "prob".
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Blend;
    }
}
