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

package runner

import (
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil/fakegemini"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestFakeGemini3_RoutesHistoryThroughBatchedPath_AndTextThroughRealtime
// validates the runner-to-connection routing decisions end-to-end without a
// real network: history flows through SendBatchedHistory, mid-session text
// flows through Send as RealtimeInput, and Close fires once.
func TestFakeGemini3_RoutesHistoryThroughBatchedPath_AndTextThroughRealtime(t *testing.T) {
	llm, rec := fakegemini.New(
		"gemini-3.1-flash-live-preview",
		fakegemini.WithReceiveSequence(
			&model.LLMResponse{TurnComplete: true},
		),
	)

	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
	})
	if err != nil {
		t.Fatal(err)
	}

	innerSvc := session.InMemoryService()
	createResp, err := innerSvc.Create(context.Background(), &session.CreateRequest{
		AppName:   "test",
		UserID:    "user1",
		SessionID: "sess1",
	})
	if err != nil {
		t.Fatal(err)
	}

	priorEvents := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hi", "user")}, Author: "user"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", "model")}, Author: "test_agent"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("any updates?", "user")}, Author: "user"},
	}
	for _, ev := range priorEvents {
		if err := innerSvc.AppendEvent(context.Background(), createResp.Session, ev); err != nil {
			t.Fatal(err)
		}
	}

	r, err := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: innerSvc,
	})
	if err != nil {
		t.Fatal(err)
	}

	queue := agent.NewLiveRequestQueue(100)
	_ = queue.Send(context.Background(), &model.LiveRequest{
		Content: genai.NewContentFromText("ping", "user"),
	})
	queue.Close()

	for ev, ierr := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{}) {
		_ = ev
		if ierr != nil {
			t.Fatalf("unexpected error: %v", ierr)
		}
	}

	batched := rec.BatchedHistory()
	if len(batched) != 1 {
		t.Fatalf("expected exactly 1 batched-history call, got %d", len(batched))
	}
	if got := len(batched[0]); got != 3 {
		t.Errorf("expected 3 history turns batched, got %d", got)
	}

	sends := rec.Sends()
	if len(sends) != 1 {
		t.Fatalf("expected exactly 1 mid-session Send, got %d (%+v)", len(sends), sends)
	}
	got := sends[0]
	if got.Content != nil {
		t.Errorf("mid-session text should have Content==nil after rewrite, got %+v", got.Content)
	}
	if got.RealtimeInput == nil || got.RealtimeInput.Text != "ping" {
		t.Errorf("mid-session text should arrive as RealtimeInput.Text=%q, got %+v", "ping", got.RealtimeInput)
	}

	for i, req := range sends {
		if req.Content != nil {
			t.Errorf("rec.Sends[%d] should not contain history Content, got %+v", i, req.Content)
		}
	}

	if !rec.Closed() {
		t.Error("expected fake connection to be closed after the run")
	}
}
