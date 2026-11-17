/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineModelConfig
    {
        public OfflineModelConfig()
        {
            Transducer = new OfflineTransducerModelConfig();
            Paraformer = new OfflineParaformerModelConfig();
            NeMoCtc = new OfflineNemoEncDecCtcModelConfig();
            Whisper = new OfflineWhisperModelConfig();
            Tdnn = new OfflineTdnnModelConfig();
            Tokens = "";
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
            ModelType = "";
            ModelingUnit = "cjkchar";
            BpeVocab = "";
            TeleSpeechCtc = "";
            SenseVoice = new OfflineSenseVoiceModelConfig();
            Moonshine = new OfflineMoonshineModelConfig();
            FireRedAsr = new OfflineFireRedAsrModelConfig();
            Dolphin = new OfflineDolphinModelConfig();
            ZipformerCtc = new OfflineZipformerCtcModelConfig();
            Canary = new OfflineCanaryModelConfig();
            WenetCtc = new OfflineWenetCtcModelConfig();
            Omnilingual = new OfflineOmnilingualAsrCtcModelConfig();
            MedAsr = new OfflineMedAsrCtcModelConfig();
            FunAsrNano = new OfflineFunAsrNanoModelConfig();
            FireRedAsrCtc = new OfflineFireRedAsrCtcModelConfig();
            Qwen3Asr = new OfflineQwen3AsrModelConfig();
            CohereTranscribe = new OfflineCohereTranscribeModelConfig();
            W2vBertCtc = "";
            LitertDispatchDir = "";
            LitertWindow = "";
            Encoder = new OfflineEncoderConfig();
        }
        public OfflineTransducerModelConfig Transducer;
        public OfflineParaformerModelConfig Paraformer;
        public OfflineNemoEncDecCtcModelConfig NeMoCtc;
        public OfflineWhisperModelConfig Whisper;
        public OfflineTdnnModelConfig Tdnn;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Tokens;

        public int NumThreads;

        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;

        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelType;

        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelingUnit;

        [MarshalAs(UnmanagedType.LPStr)]
        public string BpeVocab;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TeleSpeechCtc;

        public OfflineSenseVoiceModelConfig SenseVoice;
        public OfflineMoonshineModelConfig Moonshine;
        public OfflineFireRedAsrModelConfig FireRedAsr;
        public OfflineDolphinModelConfig Dolphin;
        public OfflineZipformerCtcModelConfig ZipformerCtc;
        public OfflineCanaryModelConfig Canary;
        public OfflineWenetCtcModelConfig WenetCtc;
        public OfflineOmnilingualAsrCtcModelConfig Omnilingual;
        public OfflineMedAsrCtcModelConfig MedAsr;
        public OfflineFunAsrNanoModelConfig FunAsrNano;
        public OfflineFireRedAsrCtcModelConfig FireRedAsrCtc;
        public OfflineQwen3AsrModelConfig Qwen3Asr;
        public OfflineCohereTranscribeModelConfig CohereTranscribe;

        /// Path to a w2v-BERT 2.0 CTC model exported from transformers.
        [MarshalAs(UnmanagedType.LPStr)]
        public string W2vBertCtc;

        /// Where LiteRT finds its accelerator library, used with a .tflite
        /// model and provider "npu".
        [MarshalAs(UnmanagedType.LPStr)]
        public string LitertDispatchDir;

        /// "WINDOW_S:OVERLAP_S:SLACK_S": how a .tflite model runs an
        /// utterance longer than its window length. Empty: windows of its
        /// longest length.
        [MarshalAs(UnmanagedType.LPStr)]
        public string LitertWindow;

        /// How the encoder runs, for example a long utterance in windows.
        public OfflineEncoderConfig Encoder;
    }
}
