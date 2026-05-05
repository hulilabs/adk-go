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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil/fakegemini"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestFakeGemini3_RoutesHistoryThroughBatchedPath_AndTextThroughRealtime
// validates the runner-to-connection routing decisions end-to-end without
// a real network: history flows through SendBatchedHistory, mid-session
// text flows through Send as RealtimeInput, and Close fires once.
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

	if got := rec.ConnectCount(); got != 1 {
		t.Errorf("expected exactly 1 ConnectLive call (no reconnect), got %d", got)
	}
}

// TestFakeGemini3_ResumeReconnectSkipsBatchedHistory verifies the critical
// composition contract with #36: when a 3.x session reconnects with a saved
// resumption handle, history MUST NOT be batched-sent again. The if handle
// == "" gate at base_flow_live.go:329 enforces this for both per-turn (2.5)
// and batched (3.x) paths.
func TestFakeGemini3_ResumeReconnectSkipsBatchedHistory(t *testing.T) {
	// First connect: yields a resumption update + GoAway so the runner
	// records the handle and reconnects. Second connect: turn-complete
	// (then Close blocks Receive until shutdown).
	llm, rec := fakegemini.New(
		"gemini-3.1-flash-live-preview",
		fakegemini.WithReceiveSequences(
			[]*model.LLMResponse{
				{
					SessionResumptionUpdate: &genai.LiveServerSessionResumptionUpdate{
						NewHandle: "h-fake-resume",
						Resumable: true,
					},
				},
				{GoAway: &genai.LiveServerGoAway{}},
			},
			[]*model.LLMResponse{
				{TurnComplete: true},
			},
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
	if err := innerSvc.AppendEvent(context.Background(), createResp.Session, priorUserEvent("hello")); err != nil {
		t.Fatal(err)
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

	// Close the queue once we observe the second connect; this lets the
	// reconnected session terminate cleanly.
	closerDone := make(chan struct{})
	go func() {
		defer close(closerDone)
		waitForCond(5*time.Second, func() bool { return rec.ConnectCount() >= 2 })
		queue.Close()
	}()

	for ev, ierr := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{
		SessionResumption: &genai.SessionResumptionConfig{},
	}) {
		_ = ev
		_ = ierr
	}
	<-closerDone

	if got := rec.ConnectCount(); got != 2 {
		t.Fatalf("expected exactly 2 ConnectLive calls (initial + reconnect), got %d", got)
	}
	handles := rec.ConnectHandles()
	if handles[0] != "" {
		t.Errorf("connect[0] handle = %q, want empty (initial fresh connect)", handles[0])
	}
	if handles[1] != "h-fake-resume" {
		t.Errorf("connect[1] handle = %q, want %q (resumed)", handles[1], "h-fake-resume")
	}

	// The batched-history send happens ONCE on the initial connect (handle
	// == ""). The reconnect (handle != "") must NOT produce another
	// batched-history call — that's the composition contract with #36.
	batched := rec.BatchedHistory()
	if len(batched) != 1 {
		t.Errorf("expected exactly 1 batched-history call (initial connect only), got %d", len(batched))
	}
}
