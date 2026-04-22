package main

import (
	"context"
	"cyberstrike-ai/internal/app"
	"cyberstrike-ai/internal/config"
	"cyberstrike-ai/internal/logger"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	var configPath = flag.String("config", "config.yaml", "配置文件路径")
	flag.Parse()

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Printf("加载配置失败: %v\n", err)
		return
	}

	// MCP 启用且 auth_header_value 为空时，自动生成随机密钥并写回配置
	if err := config.EnsureMCPAuth(*configPath, cfg); err != nil {
		fmt.Printf("MCP 鉴权配置失败: %v\n", err)
		return
	}
	if cfg.MCP.Enabled {
		config.PrintMCPConfigJSON(cfg.MCP)
	}

	// 初始化日志
	log := logger.New(cfg.Log.Level, cfg.Log.Output)

	// 创建可取消的根 context，用于优雅关闭
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 监听系统信号
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// 创建应用
	application, err := app.New(cfg, log)
	if err != nil {
		log.Fatal("应用初始化失败", "error", err)
	}

	// 在后台监听信号
	go func() {
		sig := <-sigCh
		log.Info("收到系统信号，开始优雅关闭: " + sig.String())
		application.Shutdown()
		cancel()
	}()

	// 启动服务器（传入 context 以支持优雅关闭）
	if err := application.RunWithContext(ctx); err != nil {
		// context 取消导致的关闭不视为错误
		if ctx.Err() != nil {
			log.Info("服务器已优雅关闭")
		} else {
			log.Fatal("服务器启动失败", "error", err)
		}
	}
}
