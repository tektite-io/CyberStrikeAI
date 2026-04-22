package multiagent

import (
	"strings"

	"cyberstrike-ai/internal/agent"
)

const defaultSubAgentUserContextMaxRunes = 2000

// buildUserContextForSubAgent collects all user messages from conversation
// history plus the current user message, and returns a formatted string to
// append to sub-agent instructions. This ensures sub-agents always have
// access to the full user intent (target URLs, scope, etc.) even when the
// orchestrator forgets to include them in the task description.
//
// maxRunes controls the character budget for the user-context body:
//   - 0 uses defaultSubAgentUserContextMaxRunes
//   - negative disables injection (returns "")
//
// When truncation is needed, the first and last user messages are each
// allocated half the budget so neither is lost entirely.
func buildUserContextForSubAgent(userMessage string, history []agent.ChatMessage, maxRunes int) string {
	if maxRunes < 0 {
		return ""
	}
	if maxRunes == 0 {
		maxRunes = defaultSubAgentUserContextMaxRunes
	}

	var userMsgs []string
	for _, h := range history {
		if h.Role == "user" {
			if m := strings.TrimSpace(h.Content); m != "" {
				userMsgs = append(userMsgs, m)
			}
		}
	}
	if um := strings.TrimSpace(userMessage); um != "" {
		if len(userMsgs) == 0 || userMsgs[len(userMsgs)-1] != um {
			userMsgs = append(userMsgs, um)
		}
	}
	if len(userMsgs) == 0 {
		return ""
	}

	joined := strings.Join(userMsgs, "\n---\n")

	if len([]rune(joined)) > maxRunes {
		joined = truncateKeepFirstLast(userMsgs, maxRunes)
	}

	return "\n\n## 本次会话用户原始请求（自动注入，确保你了解完整上下文）\n" + joined
}

// truncateKeepFirstLast keeps the first and last user messages, giving each
// half the rune budget. The first message typically contains target info;
// the last is the current instruction.
func truncateKeepFirstLast(msgs []string, maxRunes int) string {
	if len(msgs) == 1 {
		return truncateRunes(msgs[0], maxRunes)
	}

	first := msgs[0]
	last := msgs[len(msgs)-1]
	sep := "\n---\n...(中间对话省略)...\n---\n"
	sepLen := len([]rune(sep))

	budget := maxRunes - sepLen
	if budget <= 0 {
		return truncateRunes(first+"\n---\n"+last, maxRunes)
	}

	halfBudget := budget / 2
	firstTrunc := truncateRunes(first, halfBudget)
	lastTrunc := truncateRunes(last, budget-len([]rune(firstTrunc)))

	return firstTrunc + sep + lastTrunc
}

func truncateRunes(s string, max int) string {
	rs := []rune(s)
	if len(rs) <= max {
		return s
	}
	if max <= 0 {
		return ""
	}
	return string(rs[:max])
}
