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
        /// Negative: no limit.
        /// </summary>
        public float ScoreBeam;
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
