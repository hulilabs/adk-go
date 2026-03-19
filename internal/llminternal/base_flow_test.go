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
	"context"
	"errors"
	"io"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/runconfig"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type mockFunctionTool struct {
	name    string
	runFunc func(tool.Context, map[string]any) (map[string]any, error)
}

func (m *mockFunctionTool) Name() string {
	return m.name
}

func (m *mockFunctionTool) Description() string {
	return "mock tool"
}

func (m *mockFunctionTool) InputSchema() *genai.Schema {
	return nil
}

func (m *mockFunctionTool) OutputSchema() *genai.Schema {
	return nil
}

func (m *mockFunctionTool) IsLongRunning() bool {
	return false
}

func (m *mockFunctionTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return nil
}

func (m *mockFunctionTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	if m.runFunc != nil {
		return m.runFunc(ctx, args.(map[string]any))
	}
	return nil, nil
}

func (m *mockFunctionTool) Declaration() *genai.FunctionDeclaration {
	return nil
}

type testCase struct {
	name                 string
	tool                 toolinternal.FunctionTool
	args                 map[string]any
	beforeToolCallbacks  []BeforeToolCallback
	afterToolCallbacks   []AfterToolCallback
	onToolErrorCallbacks []OnToolErrorCallback
	want                 map[string]any
}

func TestCallTool(t *testing.T) {
	testCases := []testCase{
		{
			name: "tool runs successfully",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"result": "success"},
		},
		{
			name: "tool error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			args: map[string]any{"key": "value"},
			want: map[string]any{"error": "tool error"},
		},
		{
			name: "before callback returns result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "intercepted"}, nil
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "2nd callback should not be called"}, nil
				},
			},
			want: map[string]any{"result": "intercepted"},
		},
		{
			name: "before callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("before callback error")
				},
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "before callback error"},
		},
		{
			name: "after callback modifies result",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "original"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "modified"}, nil
				},
			},
			want: map[string]any{"result": "modified"},
		},
		{
			name: "after callback handles error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return nil, errors.New("tool error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return map[string]any{"result": "error handled"}, nil
					}
					return nil, nil
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return map[string]any{"result": "unexpected output"}, nil
				},
			},
			want: map[string]any{"result": "error handled"},
		},
		{
			name: "after callback returns error",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("after callback error")
				},
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, errors.New("unexpected error")
				},
			},
			want: map[string]any{"error": "after callback error"},
		},
		{
			name: "no-op callbacks return func results",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "success"}, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					return nil, nil
				},
			},
			want: map[string]any{"result": "success"},
		},
		{
			name: "before callback result passed to after callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "from_before"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if val, ok := result["result"]; !ok || val != "from_before" {
						return nil, errors.New("unexpected result in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			want: map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to after callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_after"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_after"},
		},
		{
			name: "before callback error passed to on tool error callback",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_on_tool_error_callback"},
		},
		{
			name: "before callback error passed to on tool error callback and after tool called",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err != nil {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "from_after"}, nil
				},
			},
			want: map[string]any{"result": "from_after"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after callback")
					}
					return nil, errors.New("error_from_after_tool")
				},
			},
			want: map[string]any{"error": "error_from_after_tool"},
		},
		{
			name: "before callback error passed to on tool error callback and passed to after tool called and handled",
			tool: &mockFunctionTool{
				name: "testTool",
				runFunc: func(ctx tool.Context, args map[string]any) (map[string]any, error) {
					t.Error("tool should not be called")
					return nil, nil
				},
			},
			beforeToolCallbacks: []BeforeToolCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
					return nil, errors.New("error_from_before")
				},
			},
			onToolErrorCallbacks: []OnToolErrorCallback{
				func(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_before" {
						t.Error("unexpected error in on tool error callback")
						return nil, errors.New("unexpected error in on tool error callback")
					}
					return nil, errors.New("error_from_on_tool_error")
				},
			},
			afterToolCallbacks: []AfterToolCallback{
				func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil || err.Error() != "error_from_on_tool_error" {
						return nil, errors.New("unexpected error in after callback")
					}
					return map[string]any{"result": "error_handled_in_on_tool_error_callback"}, nil
				},
			},
			want: map[string]any{"result": "error_handled_in_on_tool_error_callback"},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f := &Flow{
				BeforeToolCallbacks:  tc.beforeToolCallbacks,
				AfterToolCallbacks:   tc.afterToolCallbacks,
				OnToolErrorCallbacks: tc.onToolErrorCallbacks,
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			got := f.callTool(toolinternal.NewToolContext(ctx, "", nil, nil), tc.tool, tc.args)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("callTool() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMergeEventActions(t *testing.T) {
	tests := []struct {
		name  string
		base  *session.EventActions
		other *session.EventActions
		want  *session.EventActions
	}{
		{
			name:  "both nil",
			base:  nil,
			other: nil,
			want:  nil,
		},
		{
			name: "other nil returns base",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			other: nil,
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
		},
		{
			name: "base nil returns other",
			base: nil,
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
		},
		{
			name: "state delta merged with non-overlapping keys",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key2": "value2"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1", "key2": "value2"},
			},
		},
		{
			name: "state delta merged with overlapping keys - later wins",
			base: &session.EventActions{
				StateDelta: map[string]any{"key1": "original"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "overwritten"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{"key1": "overwritten"},
			},
		},
		{
			name: "state delta merged with nested map values",
			base: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key1": "value1", "key2": "value2"},
				},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key2": "updated", "key3": "value3"},
				},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{
					"outer": map[string]any{"key1": "value1", "key2": "updated", "key3": "value3"},
				},
			},
		},
		{
			name: "state delta merged with multiple keys from multiple tools",
			base: &session.EventActions{
				StateDelta: map[string]any{"tool1_key": "tool1_value"},
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"tool2_key": "tool2_value", "tool3_key": "tool3_value"},
			},
			want: &session.EventActions{
				StateDelta: map[string]any{
					"tool1_key": "tool1_value",
					"tool2_key": "tool2_value",
					"tool3_key": "tool3_value",
				},
			},
		},
		{
			name: "base has nil state delta, other has values",
			base: &session.EventActions{
				SkipSummarization: true,
			},
			other: &session.EventActions{
				StateDelta: map[string]any{"key1": "value1"},
			},
			want: &session.EventActions{
				SkipSummarization: true,
				StateDelta:        map[string]any{"key1": "value1"},
			},
		},
		{
			name: "skip summarization merging - any true wins",
			base: &session.EventActions{
				SkipSummarization: false,
			},
			other: &session.EventActions{
				SkipSummarization: true,
			},
			want: &session.EventActions{
				SkipSummarization: true,
			},
		},
		{
			name: "escalate merging - any true wins",
			base: &session.EventActions{
				Escalate: false,
			},
			other: &session.EventActions{
				Escalate: true,
			},
			want: &session.EventActions{
				Escalate: true,
			},
		},
		{
			name: "transfer to agent - last wins",
			base: &session.EventActions{
				TransferToAgent: "agent1",
			},
			other: &session.EventActions{
				TransferToAgent: "agent2",
			},
			want: &session.EventActions{
				TransferToAgent: "agent2",
			},
		},
		{
			name: "all fields merged correctly",
			base: &session.EventActions{
				StateDelta:        map[string]any{"key1": "value1"},
				SkipSummarization: false,
				TransferToAgent:   "agent1",
				Escalate:          false,
			},
			other: &session.EventActions{
				StateDelta:        map[string]any{"key2": "value2"},
				SkipSummarization: true,
				TransferToAgent:   "agent2",
				Escalate:          true,
			},
			want: &session.EventActions{
				StateDelta:        map[string]any{"key1": "value1", "key2": "value2"},
				SkipSummarization: true,
				TransferToAgent:   "agent2",
				Escalate:          true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mergeEventActions(tc.base, tc.other)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("mergeEventActions() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CFC (Compositional Function Calling) tests
// ---------------------------------------------------------------------------

// cfcMockLLM is a basic LLM that does NOT implement LiveCapableLLM.
type cfcMockLLM struct {
	name string
}

func (m *cfcMockLLM) Name() string { return m.name }

func (m *cfcMockLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

// cfcMockLiveLLM implements LiveCapableLLM.
type cfcMockLiveLLM struct {
	name    string
	conn    *cfcMockLiveConn
	mu      sync.Mutex
	lastReq *model.LLMRequest
}

func (m *cfcMockLiveLLM) Name() string { return m.name }

func (m *cfcMockLiveLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (m *cfcMockLiveLLM) ConnectLive(_ context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	m.mu.Lock()
	m.lastReq = req
	m.mu.Unlock()
	return m.conn, nil
}

var _ model.LiveCapableLLM = (*cfcMockLiveLLM)(nil)

// cfcMockLiveConn records sent messages and returns a turn-complete then EOF.
type cfcMockLiveConn struct {
	mu      sync.Mutex
	sent    []*model.LiveRequest
	recvIdx int
}

func (c *cfcMockLiveConn) Send(_ context.Context, req *model.LiveRequest) error {
	c.mu.Lock()
	c.sent = append(c.sent, req)
	c.mu.Unlock()
	return nil
}

func (c *cfcMockLiveConn) Receive(_ context.Context) (*model.LLMResponse, error) {
	c.mu.Lock()
	idx := c.recvIdx
	c.recvIdx++
	c.mu.Unlock()
	if idx == 0 {
		return &model.LLMResponse{TurnComplete: true}, nil
	}
	return nil, io.EOF
}

func (c *cfcMockLiveConn) Close() error { return nil }

func (c *cfcMockLiveConn) Sent() []*model.LiveRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	cp := make([]*model.LiveRequest, len(c.sent))
	copy(cp, c.sent)
	return cp
}

// cfcMockAgent satisfies agent.Agent for CFC tests by embedding the interface.
type cfcMockAgent struct {
	agent.Agent
	name string
}

func (a *cfcMockAgent) Name() string { return a.name }

func TestRunCFC_ErrorNonGemini2Model(t *testing.T) {
	f := &Flow{
		Model:             &cfcMockLLM{name: "gpt-4"},
		RequestProcessors: DefaultRequestProcessors,
	}
	ctx := runconfig.ToContext(context.Background(), &runconfig.RunConfig{})
	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:     &cfcMockAgent{name: "test"},
		RunConfig: &agent.RunConfig{SupportCFC: true},
	})

	var errs []error
	for _, err := range f.Run(invCtx) {
		if err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) == 0 {
		t.Fatal("expected error for non-gemini-2 model")
	}
	if errs[0].Error() != "compositional function calling (CFC) is only supported for gemini-2 models" {
		t.Errorf("unexpected error: %v", errs[0])
	}
}

func TestRunCFC_ErrorNotLiveCapable(t *testing.T) {
	f := &Flow{
		Model:             &cfcMockLLM{name: "gemini-2-flash"},
		RequestProcessors: DefaultRequestProcessors,
	}
	ctx := runconfig.ToContext(context.Background(), &runconfig.RunConfig{})
	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:     &cfcMockAgent{name: "test"},
		RunConfig: &agent.RunConfig{SupportCFC: true},
	})

	var errs []error
	for _, err := range f.Run(invCtx) {
		if err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) == 0 {
		t.Fatal("expected error for non-LiveCapable model")
	}
	want := `model "gemini-2-flash" does not support live connections required for CFC`
	if errs[0].Error() != want {
		t.Errorf("unexpected error: %v", errs[0])
	}
}

func TestRunCFC_SuccessfulRedirect(t *testing.T) {
	conn := &cfcMockLiveConn{}
	llm := &cfcMockLiveLLM{name: "gemini-2.0-flash", conn: conn}
	// Use a minimal ContentsRequestProcessor so req.Contents gets populated
	// from session events, bypassing processors that require a real LLMAgent.
	f := &Flow{
		Model:             llm,
		RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{ContentsRequestProcessor},
	}
	svc := session.InMemoryService()
	resp, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx := runconfig.ToContext(context.Background(), &runconfig.RunConfig{})
	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:       &cfcMockAgent{name: "test"},
		RunConfig:   &agent.RunConfig{SupportCFC: true},
		UserContent: genai.NewContentFromText("Hello", "user"),
		Session:     resp.Session,
	})

	var events []*session.Event
	var errs []error
	for ev, err := range f.Run(invCtx) {
		if err != nil {
			errs = append(errs, err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	// LiveFlow should have produced at least a turn-complete event.
	if len(events) == 0 {
		t.Fatal("expected at least one event from CFC live flow")
	}

	// Verify ConnectLive was called with a LiveConfig.
	llm.mu.Lock()
	req := llm.lastReq
	llm.mu.Unlock()
	if req == nil || req.LiveConfig == nil {
		t.Fatal("expected ConnectLive to be called with LiveConfig")
	}
	if len(req.LiveConfig.ResponseModalities) == 0 || req.LiveConfig.ResponseModalities[0] != genai.ModalityText {
		t.Error("expected text-only response modalities in CFC mode")
	}

	// Verify preprocessed contents were sent to the live connection
	// (not raw UserContent).
	sent := conn.Sent()
	foundContent := false
	for _, s := range sent {
		if s.Content != nil {
			foundContent = true
		}
	}
	if !foundContent {
		t.Error("expected preprocessed contents to be sent to the live connection")
	}
}
