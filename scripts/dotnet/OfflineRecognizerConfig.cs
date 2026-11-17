/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineRecognizerConfig
    {
        public OfflineRecognizerConfig()
        {
            FeatConfig = new FeatureConfig();
            ModelConfig = new OfflineModelConfig();
            LmConfig = new OfflineLMConfig();

            DecodingMethod = "greedy_search";
            MaxActivePaths = 4;
            HotwordsFile = "";
            HotwordsScore = 1.5F;
            RuleFsts = "";
            RuleFars = "";
            BlankPenalty = 0.0F;
            Hr = new HomophoneReplacerConfig();
            CtcFstDecoderConfig = new OfflineCtcFstDecoderConfig();
            Fusion = new FusionConfig();
            OptionsJson = "";
        }
        public FeatureConfig FeatConfig;
        public OfflineModelConfig ModelConfig;
        public OfflineLMConfig LmConfig;

        [MarshalAs(UnmanagedType.LPStr)]
        public string DecodingMethod;

        public int MaxActivePaths;

        [MarshalAs(UnmanagedType.LPStr)]
        public string HotwordsFile;

        public float HotwordsScore;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFsts;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFars;

        public float BlankPenalty;

        public HomophoneReplacerConfig Hr;

        public OfflineCtcFstDecoderConfig CtcFstDecoderConfig;

        /// Scorers and search settings for prefix_beam_search.
        public FusionConfig Fusion;

        /// Optional JSON object of command-line options applied after every
        /// other field, e.g. {"scorer": [{"weight": 40}], "score-beam": 0}.
        /// An error in it makes recognizer creation fail.
        [MarshalAs(UnmanagedType.LPStr)]
        public string OptionsJson;
    }
}
