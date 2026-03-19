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
	"testing"
)

func TestAudioCacheManager_CacheAndFlush(t *testing.T) {
	t.Parallel()

	t.Run("empty_flush_returns_nil", func(t *testing.T) {
		acm := NewAudioCacheManager()
		if ev := acm.FlushInput(nil, nil); ev != nil {
			t.Errorf("expected nil event for empty input buffer, got %v", ev)
		}
		if ev := acm.FlushOutput(nil, nil); ev != nil {
			t.Errorf("expected nil event for empty output buffer, got %v", ev)
		}
	})

	t.Run("cache_accumulates_data", func(t *testing.T) {
		acm := NewAudioCacheManager()
		acm.CacheAudio([]byte{0x01, 0x02}, "audio/pcm", CacheInput)
		acm.CacheAudio([]byte{0x03, 0x04}, "audio/pcm", CacheInput)
		acm.CacheAudio([]byte{0x05}, "audio/pcm", CacheOutput)

		// Flush with nil invCtx — embeds inline data directly (no artifact service).
		ev := acm.FlushInput(nil, nil)
		if ev == nil {
			t.Fatal("expected non-nil event for input flush")
		}
		if ev.Content == nil || len(ev.Content.Parts) == 0 {
			t.Fatal("expected content with parts")
		}
		part := ev.Content.Parts[0]
		if part.InlineData == nil {
			t.Fatal("expected inline data (no artifact service)")
		}
		if len(part.InlineData.Data) != 4 {
			t.Errorf("expected 4 bytes, got %d", len(part.InlineData.Data))
		}
		if ev.Author != "user" {
			t.Errorf("input author = %q, want %q", ev.Author, "user")
		}
		if ev.Content.Role != "user" {
			t.Errorf("input role = %q, want %q", ev.Content.Role, "user")
		}

		// Second flush should return nil (buffer was drained).
		if ev2 := acm.FlushInput(nil, nil); ev2 != nil {
			t.Error("expected nil on second input flush")
		}

		// Output buffer should still have data.
		ev = acm.FlushOutput(nil, nil)
		if ev == nil {
			t.Fatal("expected non-nil event for output flush")
		}
		if ev.Content.Role != "model" {
			t.Errorf("output role = %q, want %q", ev.Content.Role, "model")
		}
	})

	t.Run("returns_true_when_exceeding_max", func(t *testing.T) {
		acm := NewAudioCacheManager()
		big := make([]byte, maxCacheBytes+1)
		exceeded := acm.CacheAudio(big, "audio/pcm", CacheOutput)
		if !exceeded {
			t.Error("expected CacheAudio to return true when exceeding max size")
		}
	})

	t.Run("preserves_mime_type", func(t *testing.T) {
		acm := NewAudioCacheManager()
		acm.CacheAudio([]byte{0x01}, "audio/wav", CacheOutput)
		ev := acm.FlushOutput(nil, nil)
		if ev == nil {
			t.Fatal("expected non-nil event")
		}
		if ev.Content.Parts[0].InlineData.MIMEType != "audio/wav" {
			t.Errorf("expected audio/wav, got %q", ev.Content.Parts[0].InlineData.MIMEType)
		}
	})

	t.Run("defaults_mime_to_pcm", func(t *testing.T) {
		acm := NewAudioCacheManager()
		acm.CacheAudio([]byte{0x01}, "", CacheInput)
		ev := acm.FlushInput(nil, nil)
		if ev == nil {
			t.Fatal("expected non-nil event")
		}
		if ev.Content.Parts[0].InlineData.MIMEType != "audio/pcm" {
			t.Errorf("expected audio/pcm default, got %q", ev.Content.Parts[0].InlineData.MIMEType)
		}
	})
}
