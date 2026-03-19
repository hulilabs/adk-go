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

package agent

import (
	"context"
	"testing"
)

func TestSendActivityStart(t *testing.T) {
	t.Parallel()

	q := NewLiveRequestQueue(10)
	defer q.Close()

	if err := q.SendActivityStart(context.Background()); err != nil {
		t.Fatalf("SendActivityStart() error = %v", err)
	}

	select {
	case req := <-q.Events():
		if !req.ActivityStart {
			t.Error("expected ActivityStart = true")
		}
		if req.ActivityEnd {
			t.Error("expected ActivityEnd = false")
		}
		if req.Content != nil || req.RealtimeInput != nil || req.ToolResponse != nil {
			t.Error("expected only ActivityStart to be set")
		}
	default:
		t.Fatal("expected a message in the queue")
	}
}

func TestSendActivityEnd(t *testing.T) {
	t.Parallel()

	q := NewLiveRequestQueue(10)
	defer q.Close()

	if err := q.SendActivityEnd(context.Background()); err != nil {
		t.Fatalf("SendActivityEnd() error = %v", err)
	}

	select {
	case req := <-q.Events():
		if !req.ActivityEnd {
			t.Error("expected ActivityEnd = true")
		}
		if req.ActivityStart {
			t.Error("expected ActivityStart = false")
		}
		if req.Content != nil || req.RealtimeInput != nil || req.ToolResponse != nil {
			t.Error("expected only ActivityEnd to be set")
		}
	default:
		t.Fatal("expected a message in the queue")
	}
}

func TestSendActivityOnClosedQueue(t *testing.T) {
	t.Parallel()

	q := NewLiveRequestQueue(10)
	q.Close()

	if err := q.SendActivityStart(context.Background()); err != ErrQueueClosed {
		t.Errorf("SendActivityStart on closed queue: got %v, want ErrQueueClosed", err)
	}
	if err := q.SendActivityEnd(context.Background()); err != ErrQueueClosed {
		t.Errorf("SendActivityEnd on closed queue: got %v, want ErrQueueClosed", err)
	}
}
