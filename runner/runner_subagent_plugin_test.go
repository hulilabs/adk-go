// Copyright 2026 Google LLC
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

// This test lives in the external package runner_test on purpose: tool/agenttool
// imports runner, so an in-package (package runner) test importing agenttool
// would create an import cycle. Because external tests cannot reach the
// unexported live-mock helpers in runner_live_test.go, this file defines its own
// minimal scripted model/connection mocks.
package runner_test

import (
	"context"
	"io"
	"iter"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
)

// ---------------------------------------------------------------------------
// Minimal local mocks (external package can't see runner_live_test.go helpers)
// ---------------------------------------------------------------------------

// scriptedLLM is a model.LLM that replays a pre-queued response per turn.
// Each GenerateContent call consumes the next turn's slice of responses.
type scriptedLLM struct {
	name  string
	mu    sync.Mutex
	turns [][]*model.LLMResponse
	idx   int
}

func (s *scriptedLLM) Name() string { return s.name }

func (s *scriptedLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	s.mu.Lock()
	var turn []*model.LLMResponse
	if s.idx < len(s.turns) {
		turn = s.turns[s.idx]
		s.idx++
	}
	s.mu.Unlock()

	return func(yield func(*model.LLMResponse, error) bool) {
		for _, resp := range turn {
			if !yield(resp, nil) {
				return
			}
		}
	}
}

// liveConn is a model.LiveConnection that replays a fixed list of responses,
// then returns io.EOF to signal the end of the stream.
type liveConn struct {
	mu        sync.Mutex
	responses []*model.LLMResponse
	idx       int
}

func (c *liveConn) Send(_ context.Context, _ *model.LiveRequest) error { return nil }

func (c *liveConn) Receive(_ context.Context) (*model.LLMResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.idx >= len(c.responses) {
		return nil, io.EOF
	}
	resp := c.responses[c.idx]
	c.idx++
	return resp, nil
}

func (c *liveConn) Close() error { return nil }

// liveLLM is a model.LiveCapableLLM whose ConnectLive returns a scripted liveConn.
type liveLLM struct {
	name string
	conn *liveConn
}

func (m *liveLLM) Name() string { return m.name }

func (m *liveLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (m *liveLLM) ConnectLive(_ context.Context, _ *model.LLMRequest) (model.LiveConnection, error) {
	return m.conn, nil
}

var _ model.LiveCapableLLM = (*liveLLM)(nil)

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

func textLLMResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, "model")}
}

func functionCallLLMResponse(name string, args map[string]any) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: name, Args: args}}},
		},
	}
}

// ---------------------------------------------------------------------------
// Counting plugin (registered ONLY on the parent runner)
// ---------------------------------------------------------------------------

// callCounter records plugin callback invocations. beforeAgent/afterModel are
// keyed by agent name so the sub-agent's callbacks are isolated from the
// parent's. All counters are mutex-guarded because live mode runs concurrently.
type callCounter struct {
	mu          sync.Mutex
	beforeRun   int
	afterRun    int
	beforeAgent map[string]int
	afterModel  map[string]int
}

func newCallCounter() *callCounter {
	return &callCounter{
		beforeAgent: make(map[string]int),
		afterModel:  make(map[string]int),
	}
}

func (c *callCounter) plugin(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "counter",
		BeforeRunCallback: func(_ agent.InvocationContext) (*genai.Content, error) {
			c.mu.Lock()
			c.beforeRun++
			c.mu.Unlock()
			return nil, nil
		},
		AfterRunCallback: func(_ agent.InvocationContext) {
			c.mu.Lock()
			c.afterRun++
			c.mu.Unlock()
		},
		BeforeAgentCallback: func(ctx agent.CallbackContext) (*genai.Content, error) {
			c.mu.Lock()
			c.beforeAgent[ctx.AgentName()]++
			c.mu.Unlock()
			return nil, nil
		},
		AfterModelCallback: func(ctx agent.CallbackContext, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			c.mu.Lock()
			c.afterModel[ctx.AgentName()]++
			c.mu.Unlock()
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New: %v", err)
	}
	return p
}

func (c *callCounter) snapshot() (beforeRun, afterRun, subBeforeAgent, subAfterModel int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.beforeRun, c.afterRun, c.beforeAgent["sub_agent"], c.afterModel["sub_agent"]
}

// newSubAgent builds a plugin-less sub-agent (wrapped by agenttool with an empty
// PluginConfig sub-runner) whose model emits a single text turn.
func newSubAgent(t *testing.T) agent.Agent {
	t.Helper()
	subModel := &scriptedLLM{
		name:  "sub-model",
		turns: [][]*model.LLMResponse{{textLLMResponse("sub done")}},
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "a sub agent invoked as a tool",
		Model:       subModel,
	})
	if err != nil {
		t.Fatalf("llmagent.New(sub): %v", err)
	}
	return a
}

// newParentRunner wires a parent runner that owns the counting plugin and a
// sub-agent attached as a tool. The session is pre-created so the runner can run.
func newParentRunner(t *testing.T, parentModel model.LLM, counter *callCounter) *runner.Runner {
	t.Helper()
	subAgent := newSubAgent(t)
	parentAgent, err := llmagent.New(llmagent.Config{
		Name:        "parent_agent",
		Description: "the parent agent",
		Model:       parentModel,
		Tools:       []tool.Tool{agenttool.New(subAgent, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New(parent): %v", err)
	}

	svc := session.InMemoryService()
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName:   "test",
		UserID:    "user1",
		SessionID: "sess1",
	}); err != nil {
		t.Fatalf("session create: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        "test",
		Agent:          parentAgent,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{counter.plugin(t)}},
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r
}

// ---------------------------------------------------------------------------
// Regression test: sub-runner must inherit the parent's plugin manager.
// ---------------------------------------------------------------------------

func TestSubRunnerInheritsParentPlugins(t *testing.T) {
	t.Run("Run", func(t *testing.T) {
		counter := newCallCounter()
		parentModel := &scriptedLLM{
			name: "parent-model",
			turns: [][]*model.LLMResponse{
				{functionCallLLMResponse("sub_agent", map[string]any{"request": "go"})},
				{textLLMResponse("all done")},
			},
		}
		r := newParentRunner(t, parentModel, counter)

		for _, err := range r.Run(
			context.Background(), "user1", "sess1",
			genai.NewContentFromText("start", "user"), agent.RunConfig{},
		) {
			if err != nil {
				t.Fatalf("Run yielded error: %v", err)
			}
		}

		assertInheritance(t, counter)
	})

	t.Run("RunLive", func(t *testing.T) {
		counter := newCallCounter()
		conn := &liveConn{responses: []*model.LLMResponse{
			functionCallLLMResponse("sub_agent", map[string]any{"request": "go"}),
			textLLMResponse("all done"),
			{TurnComplete: true},
		}}
		parentModel := &liveLLM{name: "parent-live", conn: conn}
		r := newParentRunner(t, parentModel, counter)

		queue := agent.NewLiveRequestQueue(100)
		queue.Close()

		for _, err := range r.RunLive(
			context.Background(), "user1", "sess1", queue, agent.RunConfig{},
		) {
			if err != nil && err != io.EOF {
				t.Fatalf("RunLive yielded error: %v", err)
			}
		}

		assertInheritance(t, counter)
	})
}

// assertInheritance verifies both halves of the fix:
//
//	(a) the sub-agent's model/agent callbacks fired (afterModel/beforeAgent >= 1)
//	    — proves the plugin-less sub-runner inherited the parent's manager;
//	(b) run-scoped callbacks fired EXACTLY once (BeforeRun/AfterRun == 1)
//	    — proves the sub-runner did not double-fire them with its own manager.
func assertInheritance(t *testing.T, counter *callCounter) {
	t.Helper()
	beforeRun, afterRun, subBeforeAgent, subAfterModel := counter.snapshot()

	if subAfterModel < 1 {
		t.Errorf("sub-agent afterModel count = %d, want >= 1 (inherited plugin manager)", subAfterModel)
	}
	if subBeforeAgent < 1 {
		t.Errorf("sub-agent beforeAgent count = %d, want >= 1 (inherited plugin manager)", subBeforeAgent)
	}
	if beforeRun != 1 {
		t.Errorf("BeforeRun count = %d, want exactly 1 (no double-fire)", beforeRun)
	}
	if afterRun != 1 {
		t.Errorf("AfterRun count = %d, want exactly 1 (no double-fire)", afterRun)
	}
}
