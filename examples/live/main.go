// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package main demonstrates RunLive with audio output + transcription.
// Usage: GOOGLE_API_KEY=... go run ./examples/live/
// After quit, play the audio: play -t raw -r 24000 -e signed -b 16 -c 1 output.pcm
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type weatherArgs struct {
	City string `json:"city" jsonschema:"city name"`
}

type weatherResult struct {
	Temp      string `json:"temp"`
	Condition string `json:"condition"`
}

var (
	audioOut        *os.File
	totalAudioBytes int
)

func main() {
	ctx := context.Background()
	r, queue := setup(ctx)

	var err error
	audioOut, err = os.Create("output.pcm")
	if err != nil {
		log.Fatalf("create output.pcm: %v", err)
	}
	defer func() { _ = audioOut.Close() }()

	fmt.Println("Connected to Gemini Live (audio + transcription).")
	fmt.Println("---")

	go readInput(ctx, queue)

	cfg := agent.RunConfig{
		ResponseModalities:       []genai.Modality{genai.ModalityAudio},
		OutputAudioTranscription: true,
	}
	for ev, err := range r.RunLiveQueue(ctx, "user1", "sess1", queue, cfg) {
		if err != nil {
			fmt.Printf("[error] %v\n", err)
			break
		}
		if ev == nil {
			continue
		}
		printEvent(ev)
	}

	fmt.Printf("\nSession ended. Audio: %d bytes saved to output.pcm\n", totalAudioBytes)
	if totalAudioBytes > 0 {
		fmt.Println("Play with: play -t raw -r 24000 -e signed -b 16 -c 1 output.pcm")
		fmt.Println("  (install sox: brew install sox)")
	}
}

func setup(ctx context.Context) (*runner.Runner, *agent.LiveRequestQueue) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("Set GOOGLE_API_KEY")
	}

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash-native-audio-latest", &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatalf("model: %v", err)
	}

	getWeather, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Get the weather for a city",
	}, func(_ tool.Context, args weatherArgs) (weatherResult, error) {
		fmt.Printf("  [tool] get_weather(%s)\n", args.City)
		return weatherResult{Temp: "72F", Condition: "sunny"}, nil
	})
	if err != nil {
		log.Fatalf("tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "live_agent",
		Model:       m,
		Description: "A live voice agent",
		Instruction: "You are a helpful assistant. Be concise.",
		Tools:       []tool.Tool{getWeather},
	})
	if err != nil {
		log.Fatalf("agent: %v", err)
	}

	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{
		AppName: "live-test", UserID: "user1", SessionID: "sess1",
	}); err != nil {
		log.Fatalf("session: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        "live-test",
		Agent:          a,
		SessionService: svc,
	})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	return r, agent.NewLiveRequestQueue(100)
}

func readInput(ctx context.Context, queue *agent.LiveRequestQueue) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("Type a message (or 'quit' to exit):")
	fmt.Print("> ")
	for scanner.Scan() {
		text := scanner.Text()
		if text == "quit" {
			queue.Close()
			return
		}
		if err := queue.Send(ctx, &model.LiveRequest{
			Content: genai.NewContentFromText(text, "user"),
		}); err != nil {
			log.Printf("send: %v", err)
			return
		}
	}
	queue.Close()
}

func printEvent(ev *session.Event) {
	if ev.Content == nil {
		return
	}
	for _, part := range ev.Content.Parts {
		if part.Text != "" {
			fmt.Printf("[%s] %s\n", ev.Author, part.Text)
		}
		if part.FunctionCall != nil {
			fmt.Printf("[%s] calling %s(%v)\n", ev.Author, part.FunctionCall.Name, part.FunctionCall.Args)
		}
		if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
			n, err := audioOut.Write(part.InlineData.Data)
			if err != nil {
				log.Printf("write audio: %v", err)
			}
			totalAudioBytes += n
			fmt.Printf("\r  [audio] %d bytes received", totalAudioBytes)
		}
	}
	if ev.TurnComplete {
		if totalAudioBytes > 0 {
			fmt.Println()
		}
		fmt.Print("> ")
	}
}
