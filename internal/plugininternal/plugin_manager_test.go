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

package plugininternal

import (
	"context"
	"testing"

	"google.golang.org/adk/plugin"
)

func TestPluginManagerHasPlugins(t *testing.T) {
	p, err := plugin.New(plugin.Config{Name: "test_plugin"})
	if err != nil {
		t.Fatalf("plugin.New: %v", err)
	}

	tests := []struct {
		name string
		pm   func(t *testing.T) *PluginManager
		want bool
	}{
		{
			name: "nil receiver",
			pm:   func(t *testing.T) *PluginManager { return nil },
			want: false,
		},
		{
			name: "empty config",
			pm: func(t *testing.T) *PluginManager {
				pm, err := NewPluginManager(PluginConfig{})
				if err != nil {
					t.Fatalf("NewPluginManager: %v", err)
				}
				return pm
			},
			want: false,
		},
		{
			name: "nil plugins slice",
			pm: func(t *testing.T) *PluginManager {
				pm, err := NewPluginManager(PluginConfig{Plugins: nil})
				if err != nil {
					t.Fatalf("NewPluginManager: %v", err)
				}
				return pm
			},
			want: false,
		},
		{
			name: "with plugin",
			pm: func(t *testing.T) *PluginManager {
				pm, err := NewPluginManager(PluginConfig{Plugins: []*plugin.Plugin{p}})
				if err != nil {
					t.Fatalf("NewPluginManager: %v", err)
				}
				return pm
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.pm(t).HasPlugins(); got != tt.want {
				t.Errorf("HasPlugins() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPluginManagerContextRoundTrip(t *testing.T) {
	p, err := plugin.New(plugin.Config{Name: "test_plugin"})
	if err != nil {
		t.Fatalf("plugin.New: %v", err)
	}
	pm, err := NewPluginManager(PluginConfig{Plugins: []*plugin.Plugin{p}})
	if err != nil {
		t.Fatalf("NewPluginManager: %v", err)
	}

	t.Run("round trip", func(t *testing.T) {
		ctx := ToContext(context.Background(), pm)
		if got := FromContext(ctx); got != pm {
			t.Errorf("FromContext() = %p, want %p", got, pm)
		}
	})

	t.Run("nil when absent", func(t *testing.T) {
		if got := FromContext(context.Background()); got != nil {
			t.Errorf("FromContext() = %p, want nil", got)
		}
	})
}
