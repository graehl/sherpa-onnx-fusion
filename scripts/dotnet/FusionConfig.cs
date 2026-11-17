/// Copyright (c)  2026  graehl

using System;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// Settings for DecodingMethod = "prefix_beam_search".
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct FusionConfig
    {
        public FusionConfig()
        {
            Scorers = IntPtr.Zero;
            NumScorers = 0;
            LengthBonus = 0.0F;
            LengthBonusUnit = "token";
            CtcAlignment = "sum";
            TokenCandidates = 8;
            ScoreBeam = 12.0F;
            ScorerTextCase = "keep";
            LengthNormalize = 0;
            ModifiedBeamSearchRules = 0;
            CtcPrior = "";
            CtcPriorWeight = 1.0F;
            LiteralZeros = 1;
        }

        /// <summary>
        /// Native array of NumScorers SherpaOnnxScorerConfig. Leave it zero
        /// and pass a ScorerConfig[] to the recognizer's constructor, which
        /// fills it for the duration of creation.
        /// </summary>
        public IntPtr Scorers;

        public int NumScorers;

        /// <summary>
        /// Added to a hypothesis's score per LengthBonusUnit of output.
        /// </summary>
        public float LengthBonus;

        /// <summary>What LengthBonus counts: "token" or "word".</summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string LengthBonusUnit;

        /// <summary>
        /// CTC models: score an output by "sum" or "max" alignment.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string CtcAlignment;

        /// <summary>
        /// Most probable tokens per frame that may extend a hypothesis.
        /// </summary>
        public int TokenCandidates;

        /// <summary>
        /// A hypothesis survives a frame only within this score of the best.
        /// 0 or less: no limit.
        /// </summary>
        public float ScoreBeam;

        /// <summary>
        /// Case of the text character scorers are given: "keep" or "lower".
        /// The recognizer's output is unchanged.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string ScorerTextCase;

        /// <summary>
        /// Transducers, non-zero: rank results by score divided by output
        /// length, as modified_beam_search does.
        /// </summary>
        public int LengthNormalize;

        /// <summary>
        /// Stateless-decoder transducers, non-zero: search as
        /// modified_beam_search does, so that its results are reproduced.
        /// </summary>
        public int ModifiedBeamSearchRules;

        /// <summary>
        /// CTC models: a text file of "token-id log-prior" lines. Each listed
        /// token other than blank has CtcPriorWeight times its log prior
        /// subtracted from its log probability in every frame.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string CtcPrior;

        public float CtcPriorWeight;

        /// <summary>
        /// 1 (set by the constructor, which fills in every default): a 0 in
        /// a numeric field here or in a ScorerConfig is taken as 0. 0: as in
        /// the C API, a numeric 0 means the default, which is what a
        /// zero-initialized default(FusionConfig) gets. Build every
        /// ScorerConfig with new ScorerConfig(), not default(ScorerConfig) or
        /// new ScorerConfig[n], or its Weight and NumThreads are 0.
        /// </summary>
        public int LiteralZeros;
    }

    /// Copies a ScorerConfig[] into native memory for FusionConfig.Scorers.
    internal sealed class NativeScorers : IDisposable
    {
        public NativeScorers(ScorerConfig[] scorers)
        {
            if (scorers == null || scorers.Length == 0)
            {
                return;
            }

            _size = Marshal.SizeOf(typeof(ScorerConfig));
            Ptr = Marshal.AllocHGlobal(_size * scorers.Length);
            for (int i = 0; i < scorers.Length; ++i)
            {
                Marshal.StructureToPtr(scorers[i], At(i), false);
                Count = i + 1;
            }
        }

        public IntPtr Ptr = IntPtr.Zero;
        public int Count = 0;

        public void Dispose()
        {
            if (Ptr == IntPtr.Zero)
            {
                return;
            }

            for (int i = 0; i < Count; ++i)
            {
                Marshal.DestroyStructure(At(i), typeof(ScorerConfig));
            }
            Marshal.FreeHGlobal(Ptr);
            Ptr = IntPtr.Zero;
            Count = 0;
        }

        private IntPtr At(int i)
        {
            return new IntPtr(Ptr.ToInt64() + (long)i * _size);
        }

        private int _size;
    }
}
