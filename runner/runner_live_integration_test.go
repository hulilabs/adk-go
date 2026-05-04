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

//go:build integration

package runner_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// TestIntegration_GeminiLive_3x_AcceptsBatchedHistoryWithoutPolicyClose is the
// acceptance gate for the 1008 fix. It opens a real Gemini Live session with
// non-empty history on a 3.x model and verifies that the server does NOT
// respond with a "1008 policy violation" close.
func TestIntegration_GeminiLive_3x_AcceptsBatchedHistoryWithoutPolicyClose(t *testing.T) {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	runLiveAcceptanceCheck(t, "gemini-3.1-flash-live-preview")
}

// TestIntegration_GeminiLive_25_BaselineWithHistory pins the regression
// baseline: the per-turn replay path must keep working against the real API
// for the 2.5 model that this fork has historically supported.
func TestIntegration_GeminiLive_25_BaselineWithHistory(t *testing.T) {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	runLiveAcceptanceCheck(t, "gemini-2.5-flash-native-audio-latest")
}

func runLiveAcceptanceCheck(t *testing.T, modelName string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	llm, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "integration_agent",
		Model:       llm,
		Instruction: "You are a concise assistant. Answer in one short sentence.",
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	innerSvc := session.InMemoryService()
	createResp, err := innerSvc.Create(ctx, &session.CreateRequest{
		AppName:   "integration",
		UserID:    "user1",
		SessionID: "sess1",
	})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}

	priorEvents := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", "user")}, Author: "user"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hi! How can I help?", "model")}, Author: "integration_agent"},
	}
	for _, ev := range priorEvents {
		if err := innerSvc.AppendEvent(ctx, createResp.Session, ev); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}

	r, err := runner.New(runner.Config{
		AppName:        "integration",
		Agent:          a,
		SessionService: innerSvc,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	queue := agent.NewLiveRequestQueue(100)
	if err := queue.Send(ctx, &model.LiveRequest{
		Content: genai.NewContentFromText("what's 2+2?", "user"),
	}); err != nil {
		t.Fatalf("queue.Send: %v", err)
	}

	gotResponse := false
	go func() {
		// Close the queue once we've collected a model response so the
		// senderLoop exits cleanly. We don't expect to be the only writer.
		<-time.After(15 * time.Second)
		queue.Close()
	}()

	for ev, ierr := range r.RunLive(ctx, "user1", "sess1", queue, agent.RunConfig{}) {
		if ierr != nil {
			if isPolicyClose(ierr) {
				t.Fatalf("server rejected our payload with a policy close: %v", ierr)
			}
			// Any other error (timeout, cancelled, EOF) is acceptable as
			// long as we already saw a model response.
			break
		}
		if ev != nil && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text != "" {
					gotResponse = true
				}
			}
		}
		if ev != nil && ev.TurnComplete {
			queue.Close()
		}
	}

	if !gotResponse {
		t.Errorf("expected at least one model text response from %s within timeout", modelName)
	}
}

// isPolicyClose reports whether the error appears to be a 1008 policy
// violation close from the Live API. We match on string fragments since the
// underlying genai SDK surfaces the websocket close as a wrapped error.
func isPolicyClose(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "1008") || strings.Contains(msg, "policy violation")
}
