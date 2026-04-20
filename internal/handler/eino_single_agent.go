package handler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"cyberstrike-ai/internal/multiagent"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// EinoSingleAgentLoopStream Eino ADK 单代理（ChatModelAgent + Runner）流式对话；不依赖 multi_agent.enabled。
func (h *AgentHandler) EinoSingleAgentLoopStream(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		ev := StreamEvent{Type: "error", Message: "请求参数错误: " + err.Error()}
		b, _ := json.Marshal(ev)
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		done := StreamEvent{Type: "done", Message: ""}
		db, _ := json.Marshal(done)
		fmt.Fprintf(c.Writer, "data: %s\n\n", db)
		if flusher, ok := c.Writer.(http.Flusher); ok {
			flusher.Flush()
		}
		return
	}

	c.Header("X-Accel-Buffering", "no")

	var baseCtx context.Context
	clientDisconnected := false
	var sseWriteMu sync.Mutex
	sendEvent := func(eventType, message string, data interface{}) {
		if clientDisconnected {
			return
		}
		if eventType == "error" && baseCtx != nil && errors.Is(context.Cause(baseCtx), ErrTaskCancelled) {
			return
		}
		select {
		case <-c.Request.Context().Done():
			clientDisconnected = true
			return
		default:
		}
		ev := StreamEvent{Type: eventType, Message: message, Data: data}
		b, _ := json.Marshal(ev)
		sseWriteMu.Lock()
		_, err := fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		if err != nil {
			sseWriteMu.Unlock()
			clientDisconnected = true
			return
		}
		if flusher, ok := c.Writer.(http.Flusher); ok {
			flusher.Flush()
		} else {
			c.Writer.Flush()
		}
		sseWriteMu.Unlock()
	}

	h.logger.Info("收到 Eino ADK 单代理流式请求",
		zap.String("conversationId", req.ConversationID),
	)

	prep, err := h.prepareMultiAgentSession(&req)
	if err != nil {
		sendEvent("error", err.Error(), nil)
		sendEvent("done", "", nil)
		return
	}
	if prep.CreatedNew {
		sendEvent("conversation", "会话已创建", map[string]interface{}{
			"conversationId": prep.ConversationID,
		})
	}

	conversationID := prep.ConversationID
	assistantMessageID := prep.AssistantMessageID

	if prep.UserMessageID != "" {
		sendEvent("message_saved", "", map[string]interface{}{
			"conversationId": conversationID,
			"userMessageId":  prep.UserMessageID,
		})
	}

	progressCallback := h.createProgressCallback(conversationID, assistantMessageID, sendEvent)

	var cancelWithCause context.CancelCauseFunc
	baseCtx, cancelWithCause = context.WithCancelCause(context.Background())
	taskCtx, timeoutCancel := context.WithTimeout(baseCtx, 600*time.Minute)
	defer timeoutCancel()
	defer cancelWithCause(nil)

	if _, err := h.tasks.StartTask(conversationID, req.Message, cancelWithCause); err != nil {
		var errorMsg string
		if errors.Is(err, ErrTaskAlreadyRunning) {
			errorMsg = "⚠️ 当前会话已有任务正在执行中，请等待当前任务完成或点击「停止任务」后再尝试。"
			sendEvent("error", errorMsg, map[string]interface{}{
				"conversationId": conversationID,
				"errorType":      "task_already_running",
			})
		} else {
			errorMsg = "❌ 无法启动任务: " + err.Error()
			sendEvent("error", errorMsg, nil)
		}
		if assistantMessageID != "" {
			_, _ = h.db.Exec("UPDATE messages SET content = ? WHERE id = ?", errorMsg, assistantMessageID)
		}
		sendEvent("done", "", map[string]interface{}{"conversationId": conversationID})
		return
	}

	taskStatus := "completed"
	defer h.tasks.FinishTask(conversationID, taskStatus)

	sendEvent("progress", "正在启动 Eino ADK 单代理（ChatModelAgent）...", map[string]interface{}{
		"conversationId": conversationID,
	})

	stopKeepalive := make(chan struct{})
	go sseKeepalive(c, stopKeepalive, &sseWriteMu)
	defer close(stopKeepalive)

	if h.config == nil {
		sendEvent("error", "服务器配置未加载", nil)
		sendEvent("done", "", map[string]interface{}{"conversationId": conversationID})
		return
	}

	result, runErr := multiagent.RunEinoSingleChatModelAgent(
		taskCtx,
		h.config,
		&h.config.MultiAgent,
		h.agent,
		h.logger,
		conversationID,
		prep.FinalMessage,
		prep.History,
		prep.RoleTools,
		progressCallback,
	)

	if runErr != nil {
		cause := context.Cause(baseCtx)
		if errors.Is(cause, ErrTaskCancelled) {
			taskStatus = "cancelled"
			h.tasks.UpdateTaskStatus(conversationID, taskStatus)
			cancelMsg := "任务已被用户取消，后续操作已停止。"
			if assistantMessageID != "" {
				_, _ = h.db.Exec("UPDATE messages SET content = ? WHERE id = ?", cancelMsg, assistantMessageID)
				_ = h.db.AddProcessDetail(assistantMessageID, conversationID, "cancelled", cancelMsg, nil)
			}
			sendEvent("cancelled", cancelMsg, map[string]interface{}{
				"conversationId": conversationID,
				"messageId":        assistantMessageID,
			})
			sendEvent("done", "", map[string]interface{}{"conversationId": conversationID})
			return
		}

		h.logger.Error("Eino ADK 单代理执行失败", zap.Error(runErr))
		taskStatus = "failed"
		h.tasks.UpdateTaskStatus(conversationID, taskStatus)
		errMsg := "执行失败: " + runErr.Error()
		if assistantMessageID != "" {
			_, _ = h.db.Exec("UPDATE messages SET content = ? WHERE id = ?", errMsg, assistantMessageID)
			_ = h.db.AddProcessDetail(assistantMessageID, conversationID, "error", errMsg, nil)
		}
		sendEvent("error", errMsg, map[string]interface{}{
			"conversationId": conversationID,
			"messageId":      assistantMessageID,
		})
		sendEvent("done", "", map[string]interface{}{"conversationId": conversationID})
		return
	}

	if assistantMessageID != "" {
		mcpIDsJSON := ""
		if len(result.MCPExecutionIDs) > 0 {
			jsonData, _ := json.Marshal(result.MCPExecutionIDs)
			mcpIDsJSON = string(jsonData)
		}
		_, _ = h.db.Exec(
			"UPDATE messages SET content = ?, mcp_execution_ids = ? WHERE id = ?",
			result.Response,
			mcpIDsJSON,
			assistantMessageID,
		)
	}

	if result.LastReActInput != "" || result.LastReActOutput != "" {
		if err := h.db.SaveReActData(conversationID, result.LastReActInput, result.LastReActOutput); err != nil {
			h.logger.Warn("保存 ReAct 数据失败", zap.Error(err))
		}
	}

	sendEvent("response", result.Response, map[string]interface{}{
		"mcpExecutionIds": result.MCPExecutionIDs,
		"conversationId":  conversationID,
		"messageId":       assistantMessageID,
		"agentMode":       "eino_single",
	})
	sendEvent("done", "", map[string]interface{}{"conversationId": conversationID})
}

// EinoSingleAgentLoop Eino ADK 单代理非流式对话。
func (h *AgentHandler) EinoSingleAgentLoop(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.logger.Info("收到 Eino ADK 单代理非流式请求", zap.String("conversationId", req.ConversationID))

	prep, err := h.prepareMultiAgentSession(&req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var progressBuf strings.Builder
	progressCallback := func(eventType, message string, data interface{}) {
		progressBuf.WriteString(eventType)
		progressBuf.WriteByte('\n')
	}

	if h.config == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "服务器配置未加载"})
		return
	}

	result, runErr := multiagent.RunEinoSingleChatModelAgent(
		c.Request.Context(),
		h.config,
		&h.config.MultiAgent,
		h.agent,
		h.logger,
		prep.ConversationID,
		prep.FinalMessage,
		prep.History,
		prep.RoleTools,
		progressCallback,
	)
	if runErr != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": runErr.Error()})
		return
	}

	if prep.AssistantMessageID != "" {
		mcpIDsJSON := ""
		if len(result.MCPExecutionIDs) > 0 {
			jsonData, _ := json.Marshal(result.MCPExecutionIDs)
			mcpIDsJSON = string(jsonData)
		}
		_, _ = h.db.Exec(
			"UPDATE messages SET content = ?, mcp_execution_ids = ? WHERE id = ?",
			result.Response,
			mcpIDsJSON,
			prep.AssistantMessageID,
		)
	}
	if result.LastReActInput != "" || result.LastReActOutput != "" {
		_ = h.db.SaveReActData(prep.ConversationID, result.LastReActInput, result.LastReActOutput)
	}

	c.JSON(http.StatusOK, gin.H{
		"response":         result.Response,
		"conversationId":   prep.ConversationID,
		"mcpExecutionIds":  result.MCPExecutionIDs,
		"assistantMessageId": prep.AssistantMessageID,
		"agentMode":        "eino_single",
	})
}
