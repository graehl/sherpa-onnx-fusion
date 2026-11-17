// Decode a file with prefix_beam_search and a KenLM language model fused
// into the search.
package main

import (
	"log"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	lm := sherpa.ScorerConfig{Type: "kenlm"}

	flag.StringVar(&config.ModelConfig.ZipformerCtc.Model, "zipformer-ctc", "", "Path to the Zipformer CTC model")
	flag.StringVar(&config.ModelConfig.Tokens, "tokens", "", "Path to the tokens file")
	flag.IntVar(&config.ModelConfig.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.ModelConfig.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.DecodingMethod, "decoding-method", "prefix_beam_search", "Decoding method: greedy_search or prefix_beam_search")
	flag.StringVar(&lm.Model, "lm", "", "Optional. Path to a KenLM model fused into prefix_beam_search")
	flag.Float32Var(&lm.Weight, "lm-weight", 0.6, "Search weight of the LM's log score")
	flag.StringVar(&config.OptionsJson, "options-json", "", `Optional. JSON object of further options, e.g. '{"score-beam": 0}'`)

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide one wave file")
	}

	if lm.Model != "" {
		config.Fusion.Scorers = []sherpa.ScorerConfig{lm}
	}

	waveFilename := flag.Arg(0)
	wave := sherpa.ReadWave(waveFilename)
	if wave == nil {
		log.Fatalf("Failed to read %v", waveFilename)
	}

	recognizer := sherpa.NewOfflineRecognizer(&config)
	if recognizer == nil {
		log.Fatalf("Failed to create the recognizer")
	}
	defer sherpa.DeleteOfflineRecognizer(recognizer)

	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(wave.SampleRate, wave.Samples)
	recognizer.Decode(stream)

	log.Println("Text: " + stream.GetResult().Text)
}
