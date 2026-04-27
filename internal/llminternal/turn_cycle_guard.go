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

import "google.golang.org/adk/model"

// turnCycleGuard tracks turn-cycle boundaries within receiverLoop to
// suppress duplicate model content caused by orphaned tool results.
// NOT goroutine-safe — only accessed from the receiverLoop goroutine.
type turnCycleGuard struct {
	contentDelivered bool // model content emitted since last reset
	suppressActive   bool // suppress subsequent model content
}

func (g *turnCycleGuard) onModelContent() bool {
	if g.suppressActive {
		return true // suppress
	}
	g.contentDelivered = true
	return false
}

func (g *turnCycleGuard) onTurnComplete() {
	if g.contentDelivered {
		g.suppressActive = true
	}
}

func (g *turnCycleGuard) reset() {
	g.contentDelivered = false
	g.suppressActive = false
}

// applyTurnCycleGuard evaluates the guard for the given response.
// Returns nil if the message should be fully suppressed.
// Returns a (possibly modified) response otherwise.
func applyTurnCycleGuard(resp *model.LLMResponse, guard *turnCycleGuard, isAudio bool) *model.LLMResponse {
	isModelContent := resp.Content != nil && resp.Content.Role == "model" && !isAudio
	hasTranscription := resp.InputTranscription != nil || resp.OutputTranscription != nil

	// Check guard INDEPENDENTLY of transcription.
	suppressModelContent := false
	if isModelContent {
		suppressModelContent = guard.onModelContent()
	}

	// Track TurnComplete in guard regardless of suppress decision.
	if resp.TurnComplete || resp.Interrupted {
		guard.onTurnComplete()
	}

	if !suppressModelContent {
		return resp
	}

	if !hasTranscription {
		return nil // pure model content, fully suppressed
	}

	// Mixed message: strip model content, keep transcription.
	stripped := *resp
	stripped.Content = nil
	return &stripped
}
