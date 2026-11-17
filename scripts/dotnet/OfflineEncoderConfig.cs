/// Copyright (c)  2026  graehl

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>How an offline model's encoder runs.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineEncoderConfig
    {
        public OfflineEncoderConfig()
        {
            Window = new OfflineEncoderWindowConfig();
            Lengths = "";
            PadTo = 0.0F;
        }

        /// <summary>Encoding a long utterance in windows.</summary>
        public OfflineEncoderWindowConfig Window;

        /// <summary>
        /// Comma-separated input lengths in seconds an encoder call may have:
        /// an input is padded to the shortest that holds it. Needs a model
        /// that takes a padded batch.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string Lengths;

        /// <summary>
        /// Else pad an input to a multiple of this many seconds. 0: no
        /// padding.
        /// </summary>
        public float PadTo;
    }
}
