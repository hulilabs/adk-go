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

import "context"

// StreamingTool is a tool that receives a continuous stream of live requests
// (e.g. audio chunks) via its own LiveRequestQueue.
type StreamingTool interface {
	// Run starts the streaming tool. It blocks until ctx is cancelled or the
	// stream queue is closed. Implementations should select on both ctx.Done()
	// and stream.Events() / stream.Done().
	Run(ctx context.Context, stream *LiveRequestQueue)
}

// ActiveStreamingTool tracks a running StreamingTool instance. The sender loop
// fans out each incoming live request to Stream so the tool receives a copy of
// the audio/realtime data in parallel with the model connection.
type ActiveStreamingTool struct {
	// Name identifies this streaming tool (for logging/diagnostics).
	Name string
	// Stream is the tool's private LiveRequestQueue that receives duplicated
	// live requests from the sender loop.
	Stream *LiveRequestQueue
	// Cancel stops the tool's goroutine.
	Cancel context.CancelFunc
}

// Close cancels the tool's context and closes its stream queue.
func (a *ActiveStreamingTool) Close() {
	if a.Cancel != nil {
		a.Cancel()
	}
	if a.Stream != nil {
		a.Stream.Close()
	}
}
