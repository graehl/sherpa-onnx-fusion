/// Copyright (c)  2026  graehl

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// One scorer, such as a language model, fused into
    /// DecodingMethod = "prefix_beam_search". Pass an array of them to the
    /// OfflineRecognizer or OnlineRecognizer constructor.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ScorerConfig
    {
        public ScorerConfig()
        {
            Type = "";
            Model = "";
            Unit = "char";
            // Empty means "▁". LPStr is not UTF-8 on Windows, so the
            // default is left to the native side.
            Space = "";
            NumThreads = 1;
            Provider = "cpu";
            Weight = 1.0F;
            Extra = "{}";
            Rescore = 0;
        }

        /// <summary>Registered scorer type: "kenlm" or "onnx-rnn".</summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Type;

        /// <summary>Path to the scorer's model.</summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        /// <summary>What the scorer reads: "char" or "token".</summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Unit;

        /// <summary>
        /// The word-boundary unit for character scorers. Empty: "▁".
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Space;

        public int NumThreads;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;

        /// <summary>
        /// Search weight of the scorer's log score. 0 means 1; set a weight
        /// of 0 through OptionsJson.
        /// </summary>
        public float Weight;

        /// <summary>JSON object of type-specific options.</summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Extra;

        /// <summary>
        /// Non-zero: apply the scorer only to the hypotheses that survive the
        /// search, as n-best rescoring does, instead of during the search.
        /// </summary>
        public int Rescore;
    }
}
