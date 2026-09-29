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

package gemini

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// cannedTransport answers every request with one fixed JSON body, so a test can
// exercise response handling without a recording or a network round trip.
type cannedTransport struct {
	body string
}

func (c *cannedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewBufferString(c.body)),
		Request:    req,
	}, nil
}

// TestModel_Generate_ZeroCandidates pins the non-streaming path's handling of an
// HTTP 200 response that carries no candidates. Such a response is not a transport
// failure: the API returns it when the PROMPT was blocked (promptFeedback.blockReason),
// and gemini-3* via Vertex also emits candidate-less usage-only payloads. Both must
// reach the caller as the converter shape the streaming path already yields, never as
// a bare "empty response" error, so downstream block detection sees one surface on
// both paths.
func TestModel_Generate_ZeroCandidates(t *testing.T) {
	tests := []struct {
		name             string
		body             string
		wantErrorCode    string
		wantErrorMessage string
		wantNilContent   bool
		wantEmptyParts   bool
		wantModelVersion string
	}{
		{
			name: "prompt_blocked_with_feedback",
			body: `{
				"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "blocked by safety filters"},
				"usageMetadata": {"promptTokenCount": 11, "totalTokenCount": 11},
				"modelVersion": "gemini-2.5-flash"
			}`,
			wantErrorCode:    "SAFETY",
			wantErrorMessage: "blocked by safety filters",
			wantNilContent:   true,
			wantModelVersion: "gemini-2.5-flash",
		},
		{
			name: "prompt_blocked_prohibited_content",
			body: `{
				"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
				"usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7},
				"modelVersion": "gemini-2.5-flash"
			}`,
			wantErrorCode:    "PROHIBITED_CONTENT",
			wantNilContent:   true,
			wantModelVersion: "gemini-2.5-flash",
		},
		{
			name: "no_candidates_no_feedback",
			body: `{
				"usageMetadata": {"trafficType": "ON_DEMAND"},
				"modelVersion": "gemini-3.1-flash-lite"
			}`,
			wantEmptyParts:   true,
			wantModelVersion: "gemini-3.1-flash-lite",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &genai.ClientConfig{
				HTTPClient: &http.Client{Transport: &cannedTransport{body: tt.body}},
				APIKey:     "fakekey",
			}

			geminiModel, err := NewModel(t.Context(), "gemini-2.5-flash", cfg)
			if err != nil {
				t.Fatal(err)
			}

			var got *model.LLMResponse
			var gotErr error
			responses := 0
			for resp, err := range geminiModel.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("ping")}, false) {
				responses++
				got, gotErr = resp, err
			}

			if responses != 1 {
				t.Fatalf("got %d responses, want exactly 1", responses)
			}
			if gotErr != nil {
				t.Fatalf("GenerateContent() error = %v, want nil (a candidate-less 200 is not an error)", gotErr)
			}
			if got == nil {
				t.Fatal("GenerateContent() returned a nil response with a nil error")
			}
			if got.ErrorCode != tt.wantErrorCode {
				t.Errorf("ErrorCode = %q, want %q", got.ErrorCode, tt.wantErrorCode)
			}
			if got.ErrorMessage != tt.wantErrorMessage {
				t.Errorf("ErrorMessage = %q, want %q", got.ErrorMessage, tt.wantErrorMessage)
			}
			if got.ModelVersion != tt.wantModelVersion {
				t.Errorf("ModelVersion = %q, want %q", got.ModelVersion, tt.wantModelVersion)
			}
			if got.FinishReason != "" {
				t.Errorf("FinishReason = %q, want empty (no candidate carried one)", got.FinishReason)
			}
			if got.UsageMetadata == nil {
				t.Error("UsageMetadata = nil, want it forwarded so spend is still accounted")
			}
			switch {
			case tt.wantNilContent && got.Content != nil:
				t.Errorf("Content = %v, want nil", got.Content)
			case tt.wantEmptyParts && got.Content == nil:
				t.Error("Content = nil, want an empty-parts model content")
			case tt.wantEmptyParts && len(got.Content.Parts) != 0:
				t.Errorf("Content.Parts = %v, want empty", got.Content.Parts)
			}
		})
	}
}
