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

// Package liveflow implements the bidirectional streaming engine for
// Gemini Live sessions.
//
// LiveFlow.RunLive is the single entry point, called by llmagent.runLive.
// Internally the package coordinates four concerns across sibling files:
//
//   - sender.go and receiver.go run the two goroutines that exchange
//     LiveRequest values with a pluggable model.LiveConnection.
//   - turn_cycle_guard.go suppresses duplicate model content that Gemini
//     Live can replay after orphaned tool results. See the suppression
//     matrix at the top of that file for the state diagram.
//   - retry.go owns the reconnect lifecycle: GoAway signals are caught,
//     the session-resumption handle is read from the invocation context,
//     and the next runSession iteration reconnects via the same ConnectFn
//     without the consumer noticing the gap.
//   - tools.go buffers parallel FunctionCall parts within a configurable
//     CoalesceWindow (default 150ms) and flushes them as a single batch,
//     deduplicating identical (name, args) pairs by hash.
//
// eventOrError and sendEvent (flow.go) are the shared channel currency
// every concern uses to publish events into RunLive's iterator output.
package liveflow
