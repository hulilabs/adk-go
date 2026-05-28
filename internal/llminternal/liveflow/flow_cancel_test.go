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

package liveflow

import (
	"context"
	"errors"
	"math/rand/v2"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// cancelTrackingConn simulates the genai SDK's behavior where a blocking
// Read parked in Receive returns *net.OpError{Err: net.ErrClosed} after a
// local Close. Receive blocks on recvGate; Close closes the gate, which
// unblocks Receive and makes it return net.ErrClosed.
//
// Used to drive the #43 cancel-race regression tests deterministically.
type cancelTrackingConn struct {
	closed   atomic.Bool
	recvGate chan struct{}
	// returnedErr is the error Receive yields once unblocked. Defaults to
	// the net.OpError wrapping net.ErrClosed that the real SDK produces.
	returnedErr error
}

func newCancelTrackingConn() *cancelTrackingConn {
	return &cancelTrackingConn{
		recvGate:    make(chan struct{}),
		returnedErr: &net.OpError{Op: "read", Net: "tcp", Err: net.ErrClosed},
	}
}

func (c *cancelTrackingConn) Send(_ context.Context, _ *model.LiveRequest) error {
	return nil
}

func (c *cancelTrackingConn) Receive(ctx context.Context) (*model.LLMResponse, error) {
	select {
	case <-c.recvGate:
		return nil, c.returnedErr
	case <-ctx.Done():
		// Defensive: if the test never closes the gate, fall back to ctx
		// error so the goroutine doesn't leak past the test.
		return nil, ctx.Err()
	}
}

func (c *cancelTrackingConn) Close() error {
	if c.closed.CompareAndSwap(false, true) {
		close(c.recvGate)
	}
	return nil
}

// peerClosedConn returns net.ErrClosed from Receive immediately, without
// the caller ever invoking Close. Models a peer-initiated transport drop
// surfacing as a raw net.ErrClosed — must NOT be suppressed.
type peerClosedConn struct{}

func (peerClosedConn) Send(_ context.Context, _ *model.LiveRequest) error { return nil }
func (peerClosedConn) Receive(_ context.Context) (*model.LLMResponse, error) {
	return nil, &net.OpError{Op: "read", Net: "tcp", Err: net.ErrClosed}
}
func (peerClosedConn) Close() error { return nil }

// TestRunLive_CancelSuppressesLocalNetErrClosed is the positive control for
// issue #43 AC #1 & #4: when the caller cancels ctx while Receive is parked,
// the iterator must yield context.Canceled — and must NOT surface the raw
// *net.OpError{Err: net.ErrClosed} that the parked Read returns once we
// locally close the connection.
func TestRunLive_CancelSuppressesLocalNetErrClosed(t *testing.T) {
	withFastReconnectVars(t, 0, 10*time.Millisecond, 10*time.Millisecond)

	parent, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	conn := newCancelTrackingConn()
	connectFn := func(_ string) (model.LiveConnection, error) {
		// Schedule a cancel a hair after the receiver is parked so the
		// race is real: ctx.Done() and the local Close fire concurrently.
		time.AfterFunc(20*time.Millisecond, cancel)
		return conn, nil
	}

	ctx := newTestInvocationContext(t, parent)
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}

	var lastErr error
	var errs []error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
			errs = append(errs, err)
		}
	}

	if !errors.Is(lastErr, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", lastErr)
	}
	for _, e := range errs {
		if errors.Is(e, net.ErrClosed) {
			t.Fatalf("net.ErrClosed leaked through iterator: %v (all errors: %v)", e, errs)
		}
	}
}

// TestRunLive_NetErrClosedFromPeerStillSurfaces is the negative control for
// issue #43 AC #2: a raw net.ErrClosed that arrives without any local-close
// call must continue to surface so reconnect / error-handling logic can
// react to genuine peer-initiated transport drops.
func TestRunLive_NetErrClosedFromPeerStillSurfaces(t *testing.T) {
	withFastReconnectVars(t, 0, 10*time.Millisecond, 10*time.Millisecond)

	connectFn := func(_ string) (model.LiveConnection, error) {
		return peerClosedConn{}, nil
	}

	ctx := newTestInvocationContext(t, context.Background())
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
			break // signal iterator to tear down; sender is parked on queue otherwise
		}
	}

	if !errors.Is(lastErr, net.ErrClosed) {
		t.Fatalf("expected peer-initiated net.ErrClosed to surface, got %v", lastErr)
	}
	if errors.Is(lastErr, context.Canceled) {
		t.Fatalf("unexpected context.Canceled when there was no cancel: %v", lastErr)
	}
}

// TestRunLive_CancelRaceUnderLoad exercises the cancel-vs-close ordering
// across many iterations. With -race, any data race in the localClose flag
// handshake would surface here. Any leak of net.ErrClosed under cancel
// would also surface as a test failure.
func TestRunLive_CancelRaceUnderLoad(t *testing.T) {
	withFastReconnectVars(t, 0, 5*time.Millisecond, 5*time.Millisecond)

	const iterations = 50
	for i := 0; i < iterations; i++ {
		parent, cancel := context.WithCancel(context.Background())
		conn := newCancelTrackingConn()
		// Random jitter in [0, 500us] to land cancel in different points
		// of the receiver-loop lifecycle.
		jitter := time.Duration(rand.IntN(500)) * time.Microsecond
		connectFn := func(_ string) (model.LiveConnection, error) {
			time.AfterFunc(jitter, cancel)
			return conn, nil
		}

		ctx := newTestInvocationContext(t, parent)
		queue := agent.NewLiveRequestQueue(1)
		flow := &LiveFlow{}

		var lastErr error
		for ev, err := range flow.RunLive(ctx, connectFn, queue) {
			_ = ev
			if err != nil {
				lastErr = err
			}
		}

		if lastErr != nil && errors.Is(lastErr, net.ErrClosed) {
			cancel()
			t.Fatalf("iter %d: net.ErrClosed leaked on cancel (jitter=%s): %v", i, jitter, lastErr)
		}
		if lastErr != nil && !errors.Is(lastErr, context.Canceled) {
			cancel()
			t.Fatalf("iter %d: unexpected error class (jitter=%s): %v", i, jitter, lastErr)
		}
		cancel()
	}
}
