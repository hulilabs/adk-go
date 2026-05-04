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

package llmagent

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
)

func TestLiveConfigFromRunConfig(t *testing.T) {
	t.Parallel()

	t.Run("nil_returns_nil", func(t *testing.T) {
		if got := liveConfigFromRunConfig(nil); got != nil {
			t.Errorf("liveConfigFromRunConfig(nil) = %v, want nil", got)
		}
	})

	t.Run("empty_config_returns_empty", func(t *testing.T) {
		got := liveConfigFromRunConfig(&agent.RunConfig{})
		want := &genai.LiveConnectConfig{}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("all_generation_params_mapped", func(t *testing.T) {
		temp := float32(0.7)
		topP := float32(0.9)
		topK := float32(40)
		maxTokens := int32(1024)

		rc := &agent.RunConfig{
			ThinkingConfig:  &genai.ThinkingConfig{IncludeThoughts: true},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: &maxTokens,
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			ThinkingConfig:  &genai.ThinkingConfig{IncludeThoughts: true},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: 1024,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_generation_params_not_mapped", func(t *testing.T) {
		rc := &agent.RunConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("new_live_config_fields_mapped", func(t *testing.T) {
		enableAffective := true
		proactiveAudio := true
		targetTokens := int64(1024)
		rc := &agent.RunConfig{
			RealtimeInputConfig: &genai.RealtimeInputConfig{
				AutomaticActivityDetection: &genai.AutomaticActivityDetection{
					Disabled: true,
				},
			},
			Proactivity: &genai.ProactivityConfig{
				ProactiveAudio: &proactiveAudio,
			},
			EnableAffectiveDialog: &enableAffective,
			ContextWindowCompression: &genai.ContextWindowCompressionConfig{
				SlidingWindow: &genai.SlidingWindow{
					TargetTokens: &targetTokens,
				},
			},
			SessionResumption: &genai.SessionResumptionConfig{
				Handle: "prev-session-handle",
			},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			RealtimeInputConfig:      rc.RealtimeInputConfig,
			Proactivity:              rc.Proactivity,
			EnableAffectiveDialog:    &enableAffective,
			ContextWindowCompression: rc.ContextWindowCompression,
			SessionResumption:        rc.SessionResumption,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("new_live_config_fields_nil_not_mapped", func(t *testing.T) {
		rc := &agent.RunConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}
		got := liveConfigFromRunConfig(rc)
		if got.RealtimeInputConfig != nil {
			t.Error("RealtimeInputConfig should be nil")
		}
		if got.Proactivity != nil {
			t.Error("Proactivity should be nil")
		}
		if got.EnableAffectiveDialog != nil {
			t.Error("EnableAffectiveDialog should be nil")
		}
		if got.ContextWindowCompression != nil {
			t.Error("ContextWindowCompression should be nil")
		}
		if got.SessionResumption != nil {
			t.Error("SessionResumption should be nil")
		}
	})

	t.Run("mixed_old_and_new_fields", func(t *testing.T) {
		temp := float32(0.5)
		enableAffective := false
		rc := &agent.RunConfig{
			// Existing fields
			Temperature:             &temp,
			SpeechConfig:            &genai.SpeechConfig{},
			InputAudioTranscription: true,
			// New fields — only two of five set
			EnableAffectiveDialog: &enableAffective,
			SessionResumption: &genai.SessionResumptionConfig{
				Handle: "resume-token-abc",
			},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			Temperature:             &temp,
			SpeechConfig:            &genai.SpeechConfig{},
			InputAudioTranscription: &genai.AudioTranscriptionConfig{},
			EnableAffectiveDialog:   &enableAffective,
			SessionResumption:       &genai.SessionResumptionConfig{Handle: "resume-token-abc"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
		// Unset new fields must remain nil.
		if got.RealtimeInputConfig != nil {
			t.Error("RealtimeInputConfig should be nil when not set")
		}
		if got.Proactivity != nil {
			t.Error("Proactivity should be nil when not set")
		}
		if got.ContextWindowCompression != nil {
			t.Error("ContextWindowCompression should be nil when not set")
		}
	})
}

// TestLiveConfigFromRunConfig_InitialHistoryInClientContent_NotForwardedToSDK
// pins the design contract: the flag is consumed inside the live flow, not
// forwarded into genai.LiveConnectConfig. A future SDK bump that introduces a
// native HistoryConfig field should make this test fail loudly so the
// maintainer remembers to migrate the wiring.
func TestLiveConfigFromRunConfig_InitialHistoryInClientContent_NotForwardedToSDK(t *testing.T) {
	t.Parallel()

	enabled := true
	withFlag := liveConfigFromRunConfig(&agent.RunConfig{InitialHistoryInClientContent: &enabled})
	without := liveConfigFromRunConfig(&agent.RunConfig{})

	if diff := cmp.Diff(without, withFlag); diff != "" {
		t.Errorf("InitialHistoryInClientContent must not affect genai.LiveConnectConfig (-without +withFlag):\n%s", diff)
	}
}

// expectedLiveConnectConfigFieldCount pins the field count of
// genai.LiveConnectConfig as observed against the pinned SDK
// (google.golang.org/genai v1.40.0). When the SDK adds or removes a field —
// most notably if it introduces a native HistoryConfig knob — this constant
// will mismatch and TestLiveConnectConfig_FieldCount_PinsSDKShape fails.
// Treat that failure as a signal to: (a) bump this constant after auditing
// the new field, and (b) consider whether
// RunConfig.InitialHistoryInClientContent should now be forwarded into
// genai.LiveConnectConfig instead of consumed inside the live flow.
const expectedLiveConnectConfigFieldCount = 20

// TestLiveConnectConfig_FieldCount_PinsSDKShape uses reflection to detect
// SDK struct shape changes. Distinct from the diff-based test above: this
// fires even when a new field is added that we don't currently set, so it
// catches forward-compatibility risks before they become silent skips.
func TestLiveConnectConfig_FieldCount_PinsSDKShape(t *testing.T) {
	t.Parallel()

	got := reflect.TypeOf(genai.LiveConnectConfig{}).NumField()
	if got != expectedLiveConnectConfigFieldCount {
		t.Errorf(
			"genai.LiveConnectConfig has %d fields, expected %d. The SDK shape changed; "+
				"audit the new/removed field and update expectedLiveConnectConfigFieldCount. "+
				"If a HistoryConfig (or equivalent) field appeared, also wire "+
				"RunConfig.InitialHistoryInClientContent through applyLiveCapabilities "+
				"and remove the in-flow gating in internal/llminternal/base_flow_live.go.",
			got, expectedLiveConnectConfigFieldCount,
		)
	}
}

// TestLiveConfigFromRunConfig_InitialHistoryInClientContent_AllStates exercises
// nil, *true, and *false. Every state must produce the same output as omitting
// the flag entirely.
func TestLiveConfigFromRunConfig_InitialHistoryInClientContent_AllStates(t *testing.T) {
	t.Parallel()

	trueVal, falseVal := true, false
	cases := []struct {
		name string
		flag *bool
	}{
		{"nil", nil},
		{"true", &trueVal},
		{"false", &falseVal},
	}

	baseline := liveConfigFromRunConfig(&agent.RunConfig{})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := liveConfigFromRunConfig(&agent.RunConfig{InitialHistoryInClientContent: tc.flag})
			if diff := cmp.Diff(baseline, got); diff != "" {
				t.Errorf("flag=%s changed config (-baseline +got):\n%s", tc.name, diff)
			}
		})
	}
}

// TestLiveConfigFromRunConfig_PreservesExistingFields_WithFlagSet ensures the
// new flag does not disturb existing live-capability wiring.
func TestLiveConfigFromRunConfig_PreservesExistingFields_WithFlagSet(t *testing.T) {
	t.Parallel()

	enabled := true
	enableAffective := true
	rc := &agent.RunConfig{
		Proactivity: &genai.ProactivityConfig{},
		SessionResumption: &genai.SessionResumptionConfig{
			Handle: "h",
		},
		EnableAffectiveDialog:         &enableAffective,
		InitialHistoryInClientContent: &enabled,
	}

	got := liveConfigFromRunConfig(rc)
	want := &genai.LiveConnectConfig{
		Proactivity:           &genai.ProactivityConfig{},
		SessionResumption:     &genai.SessionResumptionConfig{Handle: "h"},
		EnableAffectiveDialog: &enableAffective,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
}
