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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// sendHistory replays the session's stored conversation to a freshly
// connected live session as a single batched client-content message,
// mirroring upstream adk-go SendHistory and adk-python send_history.
//
// Live API semantics: turnComplete=true means "start generating with the
// accumulated prompt", not "replay complete". A response is invited only
// when the history ends with an unanswered user turn; a model-final history
// (the normal case for a persisted voice thread) is absorbed silently.
func (lf *LiveFlow) sendHistory(
	cancelCtx context.Context,
	ctx agent.InvocationContext,
	conn model.LiveConnection,
	ts *liveTimingState,
) error {
	events := ctx.Session().Events()

	var turns []*genai.Content
	for i := range events.Len() {
		if c := filterAudioParts(events.At(i).Content); c != nil {
			turns = append(turns, c)
		}
	}
	if len(turns) == 0 {
		return nil
	}

	turnComplete := turns[len(turns)-1].Role == genai.RoleUser
	return trackedSend(cancelCtx, conn, &model.LiveRequest{
		Contents:     turns,
		TurnComplete: &turnComplete,
	}, ts)
}

// filterAudioParts returns a copy of content with audio parts removed — the
// Live API rejects audio delivered via client content (it can corrupt the
// session), and transcripts already carry the text. Returns nil when content
// is nil or no parts survive, so callers drop the turn entirely.
//
// Never mutates the input: session events share Content pointers with the
// persisted history, so filtering must not write through them.
func filterAudioParts(content *genai.Content) *genai.Content {
	if content == nil {
		return nil
	}
	parts := make([]*genai.Part, 0, len(content.Parts))
	for _, part := range content.Parts {
		if part == nil || isAudioPart(part) {
			continue
		}
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		return nil
	}
	return &genai.Content{Role: content.Role, Parts: parts}
}

// isAudioPart reports whether part carries audio as inline or file data.
func isAudioPart(part *genai.Part) bool {
	if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
		return true
	}
	if part.FileData != nil && strings.HasPrefix(part.FileData.MIMEType, "audio/") {
		return true
	}
	return false
}
