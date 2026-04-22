package multiagent

import (
	"strings"
	"testing"

	"cyberstrike-ai/internal/agent"
)

func TestBuildUserContextForSubAgent_SingleMessage(t *testing.T) {
	result := buildUserContextForSubAgent("http://8.163.32.73:8081 测试命令执行", nil, 0)
	if result == "" {
		t.Fatal("expected non-empty context")
	}
	if !strings.Contains(result, "http://8.163.32.73:8081") {
		t.Error("expected URL in context")
	}
}

func TestBuildUserContextForSubAgent_MultiTurn(t *testing.T) {
	history := []agent.ChatMessage{
		{Role: "user", Content: "http://8.163.32.73:8081 这是一个pikachu靶场，尝试测试命令执行"},
		{Role: "assistant", Content: "好的，我来测试..."},
		{Role: "user", Content: "继续，并持久化webshell"},
		{Role: "assistant", Content: "正在处理..."},
	}
	result := buildUserContextForSubAgent("你好", history, 0)
	if !strings.Contains(result, "http://8.163.32.73:8081") {
		t.Error("expected first turn URL to be preserved")
	}
	if !strings.Contains(result, "你好") {
		t.Error("expected current message in context")
	}
}

func TestBuildUserContextForSubAgent_EmptyMessages(t *testing.T) {
	result := buildUserContextForSubAgent("", nil, 0)
	if result != "" {
		t.Errorf("expected empty context, got %q", result)
	}
}

func TestBuildUserContextForSubAgent_DeduplicateCurrentMessage(t *testing.T) {
	history := []agent.ChatMessage{
		{Role: "user", Content: "你好"},
	}
	result := buildUserContextForSubAgent("你好", history, 0)
	if strings.Count(result, "你好") != 1 {
		t.Errorf("expected '你好' exactly once, got: %s", result)
	}
}

func TestBuildUserContextForSubAgent_SkipsNonUserMessages(t *testing.T) {
	history := []agent.ChatMessage{
		{Role: "user", Content: "目标是 10.0.0.1"},
		{Role: "assistant", Content: "这个不应该出现"},
		{Role: "user", Content: "开始扫描"},
	}
	result := buildUserContextForSubAgent("确认", history, 0)
	if strings.Contains(result, "这个不应该出现") {
		t.Error("assistant message should not be included")
	}
	if !strings.Contains(result, "10.0.0.1") {
		t.Error("expected IP from first user message")
	}
}

func TestBuildUserContextForSubAgent_TruncatesLongConversation(t *testing.T) {
	first := "http://target.com " + strings.Repeat("A", 500)
	var history []agent.ChatMessage
	history = append(history, agent.ChatMessage{Role: "user", Content: first})
	for i := 0; i < 10; i++ {
		history = append(history, agent.ChatMessage{Role: "user", Content: strings.Repeat("B", 500)})
	}
	last := "最后一条指令"
	result := buildUserContextForSubAgent(last, history, 0)

	if !strings.Contains(result, "http://target.com") {
		t.Error("first message (target URL) should be preserved after truncation")
	}
	if !strings.Contains(result, last) {
		t.Error("last message should be preserved after truncation")
	}
}

func TestBuildUserContextForSubAgent_DisabledByNegativeMax(t *testing.T) {
	result := buildUserContextForSubAgent("http://example.com test", nil, -1)
	if result != "" {
		t.Errorf("expected empty when disabled, got %q", result)
	}
}

func TestBuildUserContextForSubAgent_CustomMaxRunes(t *testing.T) {
	msg := strings.Repeat("A", 200)
	result := buildUserContextForSubAgent(msg, nil, 50)
	body := strings.TrimPrefix(result, "\n\n## 本次会话用户原始请求（自动注入，确保你了解完整上下文）\n")
	if len([]rune(body)) > 50 {
		t.Errorf("body should be capped at 50 runes, got %d", len([]rune(body)))
	}
}

func TestTruncateKeepFirstLast_BothPreserved(t *testing.T) {
	first := strings.Repeat("F", 100)
	last := strings.Repeat("L", 100)
	msgs := []string{first, "middle1", "middle2", last}
	result := truncateKeepFirstLast(msgs, 250)

	if !strings.HasPrefix(result, "FFFF") {
		t.Error("first message should be at the start")
	}
	if !strings.HasSuffix(result, "LLLL") {
		t.Error("last message should be at the end")
	}
	if !strings.Contains(result, "中间对话省略") {
		t.Error("should contain truncation marker")
	}
}
