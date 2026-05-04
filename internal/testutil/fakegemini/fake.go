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

// Package fakegemini provides a model.LiveCapableLLM fake suitable for
// routing-level (tier-2) tests. It records calls and replays canned receive
// sequences. Not a wire-protocol fake — for that, run tier-3 integration
// tests against the real API.
package fakegemini

import (
	"context"
	"io"
	"iter"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// Recorder captures observable calls a runner makes against the fake. Read
// recorded data via the accessor methods after the run to assert routing
// decisions; do not access the fields directly.
type Recorder struct {
	mu sync.Mutex

	batchedHistory [][]*genai.Content
	sends          []*model.LiveRequest
	closed         bool
}

// BatchedHistory returns a defensive copy of all SendBatchedHistory calls
// captured so far. Each entry is one call with the slice of turns.
func (r *Recorder) BatchedHistory() [][]*genai.Content {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([][]*genai.Content, len(r.batchedHistory))
	for i, t := range r.batchedHistory {
		cp := make([]*genai.Content, len(t))
		copy(cp, t)
		out[i] = cp
	}
	return out
}

// Sends returns a defensive copy of all Send calls captured so far.
// Mid-session text rewrites land here as RealtimeInput entries; per-turn
// history (when batching is disabled) lands here as Content entries.
func (r *Recorder) Sends() []*model.LiveRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]*model.LiveRequest, len(r.sends))
	copy(out, r.sends)
	return out
}

// Closed reports whether Close was invoked at least once.
func (r *Recorder) Closed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closed
}

// FakeLiveLLM implements model.LiveCapableLLM. ConnectLive returns a
// recording connection that drains the configured receive sequence.
type FakeLiveLLM struct {
	name     string
	recorder *Recorder
	recvSeq  []*model.LLMResponse
}

// Option configures a FakeLiveLLM.
type Option func(*FakeLiveLLM)

// WithReceiveSequence sets the canned response sequence that ConnectLive's
// Receive will drain in order. After the sequence is exhausted, Receive
// returns io.EOF.
func WithReceiveSequence(resps ...*model.LLMResponse) Option {
	return func(f *FakeLiveLLM) { f.recvSeq = resps }
}

// New constructs a FakeLiveLLM with the given model name and options.
// The returned Recorder is wired to the LLM's connection and is the test's
// single source of truth for the routing assertions.
func New(modelName string, opts ...Option) (*FakeLiveLLM, *Recorder) {
	rec := &Recorder{}
	f := &FakeLiveLLM{name: modelName, recorder: rec}
	for _, opt := range opts {
		opt(f)
	}
	return f, rec
}

// Name returns the model name passed to New.
func (f *FakeLiveLLM) Name() string { return f.name }

// GenerateContent is a no-op for live-only tests; it returns an empty
// iterator so the type satisfies model.LLM.
func (f *FakeLiveLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(_ func(*model.LLMResponse, error) bool) {}
}

// ConnectLive returns a recording connection that satisfies both
// model.LiveConnection and model.BatchedHistorySender.
func (f *FakeLiveLLM) ConnectLive(_ context.Context, _ *model.LLMRequest) (model.LiveConnection, error) {
	return &fakeConn{
		rec:     f.recorder,
		recv:    f.recvSeq,
		closeCh: make(chan struct{}),
	}, nil
}

var _ model.LiveCapableLLM = (*FakeLiveLLM)(nil)

type fakeConn struct {
	rec *Recorder

	mu      sync.Mutex
	recv    []*model.LLMResponse
	recvAt  int
	closed  bool
	closeCh chan struct{}
}

func (c *fakeConn) Send(_ context.Context, req *model.LiveRequest) error {
	c.rec.mu.Lock()
	c.rec.sends = append(c.rec.sends, req)
	c.rec.mu.Unlock()
	return nil
}

func (c *fakeConn) SendBatchedHistory(_ context.Context, turns []*genai.Content) error {
	cp := make([]*genai.Content, len(turns))
	copy(cp, turns)
	c.rec.mu.Lock()
	c.rec.batchedHistory = append(c.rec.batchedHistory, cp)
	c.rec.mu.Unlock()
	return nil
}

func (c *fakeConn) Receive(ctx context.Context) (*model.LLMResponse, error) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, io.EOF
	}
	if c.recvAt >= len(c.recv) {
		c.mu.Unlock()
		// Block until close or cancellation so we don't busy-loop EOF
		// before pending sends have been recorded by the runner. Mirrors
		// the real Live API's long-polling shape for routing tests.
		select {
		case <-c.closeCh:
			return nil, io.EOF
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	resp := c.recv[c.recvAt]
	c.recvAt++
	c.mu.Unlock()
	if resp != nil && resp.ReceivedAt.IsZero() {
		resp.ReceivedAt = time.Now()
	}
	return resp, nil
}

func (c *fakeConn) Close() error {
	c.mu.Lock()
	alreadyClosed := c.closed
	c.closed = true
	c.mu.Unlock()
	if !alreadyClosed {
		close(c.closeCh)
	}
	c.rec.mu.Lock()
	c.rec.closed = true
	c.rec.mu.Unlock()
	return nil
}

var (
	_ model.LiveConnection       = (*fakeConn)(nil)
	_ model.BatchedHistorySender = (*fakeConn)(nil)
)
