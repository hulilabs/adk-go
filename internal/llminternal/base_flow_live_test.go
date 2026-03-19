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

package llminternal

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
)

func TestSenderLoop_FanOutAndCleanup(t *testing.T) {
	t.Parallel()

	toolQueue := agent.NewLiveRequestQueue(100)
	var closed atomic.Bool
	toolCtx, toolCancel := context.WithCancel(context.Background())
	st := &agent.ActiveStreamingTool{
		Name:   "test-stream",
		Stream: toolQueue,
		Cancel: func() {
			closed.Store(true)
			toolCancel()
		},
	}

	// Create invocation context with the streaming tool registered.
	invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
	invCtx.AddActiveStreamingTool(st)

	// Mock connection that also counts sends for fan-out verification.
	conn := newFanOutMockConn()

	// Create a queue with some messages.
	queue := agent.NewLiveRequestQueue(10)
	_ = queue.Send(context.Background(), &model.LiveRequest{Content: genai.NewContentFromText("hello", "user")})
	_ = queue.Send(context.Background(), &model.LiveRequest{Content: genai.NewContentFromText("world", "user")})
	queue.Close()

	eventCh := make(chan eventOrError, 10)
	lf := &LiveFlow{}

	// Run senderLoop — it will process both messages, fan out, then
	// the defer will close the streaming tool.
	lf.senderLoop(context.Background(), invCtx, conn, queue, eventCh)

	// Verify messages were sent to the connection.
	if got := conn.sendCount.Load(); got != 2 {
		t.Errorf("connection got %d sends, want 2", got)
	}

	// Verify the tool was closed on exit (defer fired).
	if !closed.Load() {
		t.Error("expected streaming tool to be closed after senderLoop exits")
	}

	// Ensure the tool's cancel was called.
	select {
	case <-toolCtx.Done():
		// ok
	default:
		t.Error("expected tool context to be cancelled")
	}
}

func TestSenderLoop_CleanupOnSendError(t *testing.T) {
	t.Parallel()

	toolQueue := agent.NewLiveRequestQueue(100)
	var closed atomic.Bool
	st := &agent.ActiveStreamingTool{
		Name:   "test-stream",
		Stream: toolQueue,
		Cancel: func() { closed.Store(true) },
	}

	invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
	invCtx.AddActiveStreamingTool(st)

	// Connection that fails on first send.
	conn := &fanOutMockConn{failAt: 0} //nolint:govet

	queue := agent.NewLiveRequestQueue(10)
	_ = queue.Send(context.Background(), &model.LiveRequest{Content: genai.NewContentFromText("fail", "user")})
	queue.Close()

	eventCh := make(chan eventOrError, 10)
	lf := &LiveFlow{}

	lf.senderLoop(context.Background(), invCtx, conn, queue, eventCh)

	// Tool should still be closed even when conn.Send errors.
	if !closed.Load() {
		t.Error("expected streaming tool to be closed on send error exit")
	}

	// Should have an error in eventCh.
	select {
	case msg := <-eventCh:
		if msg.err == nil {
			t.Error("expected error event from failed send")
		}
	default:
		t.Error("expected error event in eventCh")
	}
}

type fanOutMockConn struct {
	sendCount atomic.Int32
	failAt    int32 // index at which Send returns an error; -1 = never fail
}

func newFanOutMockConn() *fanOutMockConn {
	return &fanOutMockConn{failAt: -1}
}

func (c *fanOutMockConn) Send(_ context.Context, _ *model.LiveRequest) error {
	idx := c.sendCount.Add(1) - 1
	if c.failAt >= 0 && idx == c.failAt {
		return fmt.Errorf("mock send error")
	}
	return nil
}

func (c *fanOutMockConn) Receive(_ context.Context) (*model.LLMResponse, error) {
	return nil, nil
}

func (c *fanOutMockConn) Close() error { return nil }
