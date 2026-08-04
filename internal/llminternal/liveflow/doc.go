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
//
// # Relationship to the upstream live engine
//
// Since the google/adk-go v1.5.0 merge, two live engines coexist in this
// repository. Runner.RunLive, the agent.LiveSession API, and the adkrest
// /run_live endpoint drive the UPSTREAM engine (llminternal Flow.RunLive
// plus the googlellm live connection) — not this package. As of v1.5.0
// that engine has known gaps:
//
//   - GoAway is not surfaced as a signal; reconnect eligibility is decided
//     by substring-matching error text ("GoAway", "EOF", "1008", ...).
//   - Reconnects retry immediately in a loop with no backoff and no
//     attempt budget.
//   - Each reconnect abandons the previous attempt's unbuffered error
//     channel, leaking a blocked reader/sender goroutine per cycle
//     (reported upstream: google/adk-go#1152).
//   - The preprocessed history is re-sent on every reconnect, even when
//     resuming with a session handle the server already has context for.
//   - ToolCallCancellation server messages are dropped, so cancelled
//     tool calls keep running.
//   - Every event is authored as the agent, so input transcriptions
//     (user speech) are misattributed to the model in session history.
//
// Runner.RunLiveQueue driving this package is the hulilabs-supported live
// path: it handles each of the above (GoAway-aware reconnects with
// resumption handles, turn-cycle replay suppression, tool cancellation,
// role-correct transcription events). Route new live features and fixes
// here; treat the upstream engine as upstream-owned code that syncs with
// google/adk-go.
package liveflow
