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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestClientContentInput(t *testing.T) {
	userTurn := genai.NewContentFromText("hello", genai.RoleUser)
	modelTurn := genai.NewContentFromText("hi there", genai.RoleModel)
	falseVal := false
	trueVal := true

	tests := []struct {
		name         string
		req          *model.LiveRequest
		wantTurns    []*genai.Content
		wantComplete bool
	}{
		{
			name:         "single content defaults to turn complete",
			req:          &model.LiveRequest{Content: userTurn},
			wantTurns:    []*genai.Content{userTurn},
			wantComplete: true,
		},
		{
			name:         "single content explicit false",
			req:          &model.LiveRequest{Content: userTurn, TurnComplete: &falseVal},
			wantTurns:    []*genai.Content{userTurn},
			wantComplete: false,
		},
		{
			name: "batched contents preserve order and explicit false",
			req: &model.LiveRequest{
				Contents:     []*genai.Content{userTurn, modelTurn},
				TurnComplete: &falseVal,
			},
			wantTurns:    []*genai.Content{userTurn, modelTurn},
			wantComplete: false,
		},
		{
			name: "batched contents explicit true",
			req: &model.LiveRequest{
				Contents:     []*genai.Content{userTurn, modelTurn, userTurn},
				TurnComplete: &trueVal,
			},
			wantTurns:    []*genai.Content{userTurn, modelTurn, userTurn},
			wantComplete: true,
		},
		{
			name: "contents wins over content when both set",
			req: &model.LiveRequest{
				Content:      modelTurn,
				Contents:     []*genai.Content{userTurn},
				TurnComplete: &falseVal,
			},
			wantTurns:    []*genai.Content{userTurn},
			wantComplete: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clientContentInput(tt.req)
			if len(got.Turns) != len(tt.wantTurns) {
				t.Fatalf("Turns len = %d, want %d", len(got.Turns), len(tt.wantTurns))
			}
			for i := range got.Turns {
				if got.Turns[i] != tt.wantTurns[i] {
					t.Errorf("Turns[%d] = %p, want %p", i, got.Turns[i], tt.wantTurns[i])
				}
			}
			if got.TurnComplete == nil {
				t.Fatal("TurnComplete is nil, want explicit value")
			}
			if *got.TurnComplete != tt.wantComplete {
				t.Errorf("TurnComplete = %v, want %v", *got.TurnComplete, tt.wantComplete)
			}
		})
	}
}
