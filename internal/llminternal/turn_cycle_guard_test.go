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

import "testing"

func TestTurnCycleGuard(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T, g *turnCycleGuard)
	}{
		{
			name: "initial onModelContent returns false",
			run: func(t *testing.T, g *turnCycleGuard) {
				if g.onModelContent() {
					t.Error("expected onModelContent to return false initially")
				}
			},
		},
		{
			name: "after content and TurnComplete, onModelContent returns true",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				if !g.onModelContent() {
					t.Error("expected onModelContent to return true (suppress) after content+TurnComplete")
				}
			},
		},
		{
			name: "after reset, onModelContent returns false",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				g.reset()
				if g.onModelContent() {
					t.Error("expected onModelContent to return false after reset")
				}
			},
		},
		{
			name: "TurnComplete without prior content does not arm suppression",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onTurnComplete()
				if g.onModelContent() {
					t.Error("expected onModelContent to return false — TurnComplete without content should not arm")
				}
			},
		},
		{
			name: "multiple onModelContent before TurnComplete all return false",
			run: func(t *testing.T, g *turnCycleGuard) {
				for i := range 5 {
					if g.onModelContent() {
						t.Errorf("onModelContent[%d] should return false before TurnComplete", i)
					}
				}
				g.onTurnComplete()
				if !g.onModelContent() {
					t.Error("expected suppression after multiple content + TurnComplete")
				}
			},
		},
		{
			name: "suppression persists across multiple calls",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete()
				for i := range 3 {
					if !g.onModelContent() {
						t.Errorf("onModelContent[%d] should remain suppressed", i)
					}
				}
			},
		},
		{
			name: "reset is idempotent",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.reset()
				g.reset()
				if g.onModelContent() {
					t.Error("expected false after idempotent resets")
				}
			},
		},
		{
			// Once armed, redundant onTurnComplete calls must not flip back to
			// disarmed nor double-arm in a way that breaks subsequent
			// suppression. The model occasionally re-emits TurnComplete on
			// the same logical turn — those should be no-ops.
			name: "TurnComplete is idempotent when already armed",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent()
				g.onTurnComplete() // arms
				g.onTurnComplete() // redundant — must remain armed
				if !g.onModelContent() {
					t.Error("expected suppression to persist after redundant onTurnComplete")
				}
			},
		},
		{
			// reset() must clear BOTH contentDelivered and suppressActive.
			// Guards a regression where someone "optimizes" reset to only
			// touch suppressActive: a subsequent onTurnComplete would
			// incorrectly arm because the stale contentDelivered persists.
			name: "reset clears contentDelivered, not just suppressActive",
			run: func(t *testing.T, g *turnCycleGuard) {
				g.onModelContent() // contentDelivered = true
				g.reset()          // must clear both
				g.onTurnComplete() // post-reset: contentDelivered=false, must NOT arm
				if g.onModelContent() {
					t.Error("reset must clear contentDelivered; otherwise onTurnComplete after reset incorrectly arms")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &turnCycleGuard{}
			tt.run(t, g)
		})
	}
}
