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
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestRunLive_RetryBudgetExhaustion verifies that when every connect attempt
// fails, RunLive yields a "failed to connect live after N retries" error and
// stops after exactly maxReconnectRetries+1 attempts (one initial attempt
// plus maxReconnectRetries retries).
func TestRunLive_RetryBudgetExhaustion(t *testing.T) {
	withFastReconnectVars(t, 2, 10*time.Millisecond, 10*time.Millisecond)

	var calls atomic.Int32
	connectFn := func(_ string) (model.LiveConnection, error) {
		calls.Add(1)
		return nil, errors.New("synthetic connect failure")
	}

	ctx := newTestInvocationContext(t, context.Background())
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
		}
	}

	if lastErr == nil || !strings.Contains(lastErr.Error(), "failed to connect live after") {
		t.Fatalf("expected retry-exhaustion error, got %v", lastErr)
	}
	if got, want := calls.Load(), int32(maxReconnectRetries+1); got != want {
		t.Errorf("connectFn called %d times, want %d", got, want)
	}
}

// TestRunLive_ContextCancelDuringReconnectBackoff verifies that cancelling
// the parent context during retry backoff propagates promptly and RunLive
// returns a context.Canceled chain — not the retry-exhaustion message —
// well before the next backoff timer would have fired.
//
// We use a 1s base backoff and a 50ms cancel deadline so a non-interruptible
// sleep would yield wall time ≥ 1s, while a select-on-Done backoff yields
// well under 100ms. The 200ms threshold sits comfortably between the two
// modes, so a regression to bare time.Sleep will trip this assertion.
func TestRunLive_ContextCancelDuringReconnectBackoff(t *testing.T) {
	withFastReconnectVars(t, 5, time.Second, 10*time.Millisecond)

	parent, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	var calls atomic.Int32
	connectFn := func(_ string) (model.LiveConnection, error) {
		n := calls.Add(1)
		if n == 1 {
			// Schedule a cancel well before the 1s backoff timer fires.
			time.AfterFunc(50*time.Millisecond, cancel)
		}
		return nil, errors.New("synthetic")
	}

	ctx := newTestInvocationContext(t, parent)
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}
	start := time.Now()

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
		}
	}
	elapsed := time.Since(start)

	if !errors.Is(lastErr, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", lastErr)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("expected fast cancel, elapsed = %s (likely waited out a non-interruptible sleep)", elapsed)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("expected exactly 1 connect attempt before cancel, got %d", got)
	}
}

// withFastReconnectVars overrides the package-level reconnect knobs for the
// duration of a test and restores them via t.Cleanup. Tests use small values
// so they run in tens of milliseconds rather than seconds.
func withFastReconnectVars(t *testing.T, retries int, baseBackoff, goAwaySleep time.Duration) {
	t.Helper()
	prevR, prevB, prevG := maxReconnectRetries, reconnectBaseBackoff, reconnectGoAwaySleep
	maxReconnectRetries = retries
	reconnectBaseBackoff = baseBackoff
	reconnectGoAwaySleep = goAwaySleep
	t.Cleanup(func() {
		maxReconnectRetries = prevR
		reconnectBaseBackoff = prevB
		reconnectGoAwaySleep = prevG
	})
}

// newTestInvocationContext builds a minimal agent.InvocationContext suitable
// for driving LiveFlow.RunLive in unit tests. Only the embedded context.Context
// (for cancellation) and Session (for sendHistory) are exercised by tests
// where connectFn fails before the session loops start.
func newTestInvocationContext(t *testing.T, parent context.Context) agent.InvocationContext {
	t.Helper()
	resp, err := session.InMemoryService().Create(parent, &session.CreateRequest{
		AppName:   "test",
		UserID:    "u",
		SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	return icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Session: resp.Session,
	})
}
