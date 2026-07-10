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
	"errors"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// historyInvCtx builds an InvocationContext over an in-memory session
// seeded with the given event contents (nil entries append an event with
// nil Content, exercising the skip path).
func historyInvCtx(t *testing.T, contents []*genai.Content) agent.InvocationContext {
	t.Helper()
	parent := context.Background()
	svc := session.InMemoryService()
	resp, err := svc.Create(parent, &session.CreateRequest{
		AppName: "test-app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	for _, c := range contents {
		ev := session.NewEvent("inv-1")
		ev.Content = c
		if err := svc.AppendEvent(parent, resp.Session, ev); err != nil {
			t.Fatalf("append event: %v", err)
		}
	}
	got, err := svc.Get(parent, &session.GetRequest{
		AppName: "test-app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("session get: %v", err)
	}
	return icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Session: got.Session,
	})
}

// sendHistoryTo runs sendHistory against conn and returns its error.
func sendHistoryTo(t *testing.T, conn model.LiveConnection, contents []*genai.Content) error {
	t.Helper()
	lf := &LiveFlow{}
	invCtx := historyInvCtx(t, contents)
	return lf.sendHistory(context.Background(), invCtx, conn, &liveTimingState{})
}

func audioInlinePart() *genai.Part {
	return &genai.Part{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{1, 2, 3}}}
}

func audioFilePart() *genai.Part {
	return &genai.Part{FileData: &genai.FileData{MIMEType: "audio/wav", FileURI: "gs://b/a.wav"}}
}

func textsOf(t *testing.T, turns []*genai.Content) []string {
	t.Helper()
	var out []string
	for _, turn := range turns {
		if len(turn.Parts) == 0 {
			t.Fatalf("turn with no parts in send: %+v", turn)
		}
		out = append(out, turn.Parts[0].Text)
	}
	return out
}

// requireSingleBatch asserts conn saw exactly one send carrying a Contents
// batch with an explicit TurnComplete, and returns it.
func requireSingleBatch(t *testing.T, conn *mockLiveConnTel) *model.LiveRequest {
	t.Helper()
	conn.mu.Lock()
	defer conn.mu.Unlock()
	if len(conn.sendLog) != 1 {
		t.Fatalf("sends = %d, want exactly 1 batched history send", len(conn.sendLog))
	}
	req := conn.sendLog[0]
	if len(req.Contents) == 0 {
		t.Fatal("history send has empty Contents batch")
	}
	if req.Content != nil {
		t.Error("history send must use Contents, not Content")
	}
	if req.TurnComplete == nil {
		t.Fatal("history send must set TurnComplete explicitly")
	}
	return req
}

func TestSendHistory_ModelFinalAbsorbedSilently(t *testing.T) {
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
		genai.NewContentFromText("hi there", genai.RoleModel),
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if got, want := len(req.Contents), 2; got != want {
		t.Fatalf("batch size = %d, want %d", got, want)
	}
	if got := textsOf(t, req.Contents); got[0] != "hello" || got[1] != "hi there" {
		t.Errorf("turn order = %v, want [hello, hi there]", got)
	}
	if *req.TurnComplete {
		t.Error("model-final history must send TurnComplete=false (no unprompted response)")
	}
}

func TestSendHistory_UserFinalInvitesResponse(t *testing.T) {
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
		genai.NewContentFromText("hi there", genai.RoleModel),
		genai.NewContentFromText("what's the weather?", genai.RoleUser),
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if got, want := len(req.Contents), 3; got != want {
		t.Fatalf("batch size = %d, want %d", got, want)
	}
	if !*req.TurnComplete {
		t.Error("user-final history must send TurnComplete=true (answer the pending turn)")
	}
}

func TestSendHistory_StripsAudioPartsKeepsText(t *testing.T) {
	mixed := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		genai.NewPartFromText("spoken words"),
		audioInlinePart(),
	}}
	conn := newMockLiveConnTel()
	if err := sendHistoryTo(t, conn, []*genai.Content{mixed}); err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	turn := req.Contents[0]
	if turn.Role != genai.RoleUser {
		t.Errorf("role = %q, want user (must be preserved)", turn.Role)
	}
	if len(turn.Parts) != 1 || turn.Parts[0].Text != "spoken words" {
		t.Errorf("parts = %+v, want only the text part", turn.Parts)
	}
}

func TestSendHistory_TurnCompleteEvaluatedOnFilteredList(t *testing.T) {
	// adk-python parity: the trailing user turn is audio-only, so after
	// filtering the history ends on a model turn — no response invited.
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
		genai.NewContentFromText("hi there", genai.RoleModel),
		{Role: genai.RoleUser, Parts: []*genai.Part{audioInlinePart()}},
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if got, want := len(req.Contents), 2; got != want {
		t.Fatalf("batch size = %d, want %d (audio-only turn dropped)", got, want)
	}
	if *req.TurnComplete {
		t.Error("TurnComplete must be computed on the filtered list → false")
	}
}

func TestSendHistory_StripsFileDataAudio(t *testing.T) {
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		{Role: genai.RoleUser, Parts: []*genai.Part{
			genai.NewPartFromText("see attachment"),
			audioFilePart(),
		}},
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if len(req.Contents[0].Parts) != 1 || req.Contents[0].Parts[0].Text != "see attachment" {
		t.Errorf("parts = %+v, want file-data audio stripped", req.Contents[0].Parts)
	}
}

func TestSendHistory_NoSendWhenNothingSurvives(t *testing.T) {
	cases := map[string][]*genai.Content{
		"empty history": {},
		"all audio": {
			{Role: genai.RoleUser, Parts: []*genai.Part{audioInlinePart()}},
			{Role: genai.RoleModel, Parts: []*genai.Part{audioInlinePart()}},
		},
		"nil contents and empty parts": {nil, {Role: genai.RoleUser}},
	}
	for name, contents := range cases {
		t.Run(name, func(t *testing.T) {
			conn := newMockLiveConnTel()
			if err := sendHistoryTo(t, conn, contents); err != nil {
				t.Fatalf("sendHistory: %v", err)
			}
			conn.mu.Lock()
			defer conn.mu.Unlock()
			if len(conn.sendLog) != 0 {
				t.Errorf("sends = %d, want 0", len(conn.sendLog))
			}
		})
	}
}

func TestSendHistory_SkipsNilPartsWithoutPanic(t *testing.T) {
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		{Role: genai.RoleUser, Parts: []*genai.Part{nil, genai.NewPartFromText("still here")}},
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if len(req.Contents[0].Parts) != 1 || req.Contents[0].Parts[0].Text != "still here" {
		t.Errorf("parts = %+v, want nil part skipped", req.Contents[0].Parts)
	}
}

func TestSendHistory_ToolHistorySurvivesAndResumes(t *testing.T) {
	// Function-call/-response parts must pass the audio filter untouched,
	// and a history ending on an unanswered tool response (role user)
	// invites the model to resume — upstream semantics.
	fc := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "get_weather", Args: map[string]any{"city": "SJO"}}},
	}}
	fr := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "get_weather", Response: map[string]any{"temp": 24}}},
	}}
	conn := newMockLiveConnTel()
	err := sendHistoryTo(t, conn, []*genai.Content{
		genai.NewContentFromText("weather?", genai.RoleUser),
		fc,
		fr,
	})
	if err != nil {
		t.Fatalf("sendHistory: %v", err)
	}
	req := requireSingleBatch(t, conn)
	if got, want := len(req.Contents), 3; got != want {
		t.Fatalf("batch size = %d, want %d (tool turns must survive)", got, want)
	}
	if req.Contents[1].Parts[0].FunctionCall == nil {
		t.Error("function call part was dropped or altered by the filter")
	}
	if req.Contents[2].Parts[0].FunctionResponse == nil {
		t.Error("function response part was dropped or altered by the filter")
	}
	if !*req.TurnComplete {
		t.Error("tool-response-final history must send TurnComplete=true (model resumes)")
	}
}

func TestSendHistory_DoesNotMutatePersistedHistory(t *testing.T) {
	// Session events share Content pointers with persisted history (the
	// in-memory service appends shallow copies). Filtering must build fresh
	// slices — an in-place filter would permanently strip audio from the
	// stored session, not just the outbound payload.
	textPart := genai.NewPartFromText("spoken words")
	audioPart := audioInlinePart()
	original := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{textPart, audioPart}}

	conn := newMockLiveConnTel()
	if err := sendHistoryTo(t, conn, []*genai.Content{original}); err != nil {
		t.Fatalf("sendHistory: %v", err)
	}

	if len(original.Parts) != 2 {
		t.Fatalf("original parts len = %d, want 2 — filter mutated shared history", len(original.Parts))
	}
	if original.Parts[0] != textPart || original.Parts[1] != audioPart {
		t.Error("original part pointers changed — filter wrote through shared history")
	}
	if original.Role != genai.RoleUser {
		t.Errorf("original role = %q, want %q — filter mutated shared history", original.Role, genai.RoleUser)
	}
	req := requireSingleBatch(t, conn)
	if req.Contents[0] == original {
		t.Error("outbound content aliases the persisted event; filter must copy")
	}
}

func TestSendHistory_PropagatesSendError(t *testing.T) {
	wantErr := errors.New("synthetic send failure")
	conn := &sendErrConn{sendErr: wantErr}
	err := sendHistoryTo(t, conn, []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
}
