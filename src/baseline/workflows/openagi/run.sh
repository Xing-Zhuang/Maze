#!/bin/bash

echo "🧹 清理现有服务..."

# 停止所有相关服务（忽略权限错误）
pkill -f redis-server 2>/dev/null || echo "⚠️  无法杀死某些Redis进程（权限不足或进程不存在）"
pkill -f ray 2>/dev/null || echo "⚠️  无法杀死某些Ray进程（权限不足或进程不存在）"
pkill -f api_server.py 2>/dev/null || echo "⚠️  无法杀死某些API服务进程（权限不足或进程不存在）"
pkill -f scheduler.py 2>/dev/null || echo "⚠️  无法杀死某些调度器进程（权限不足或进程不存在）"
sleep 3

echo "🚀 启动AgentOS服务..."

# 1. 启动Redis
echo "📊 启动Redis服务器..."

# 检查Redis可执行文件
if [ -f "$CONDA_PREFIX/bin/redis-server" ]; then
    REDIS_SERVER="$CONDA_PREFIX/bin/redis-server"
    REDIS_CLI="$CONDA_PREFIX/bin/redis-cli"
    echo "  使用conda环境Redis: $REDIS_SERVER"
elif command -v redis-server &> /dev/null; then
    REDIS_SERVER="redis-server"
    REDIS_CLI="redis-cli"
    echo "  使用系统Redis: $(which redis-server)"
else
    echo "❌ Redis未找到"
    exit 1
fi

# 创建Redis配置文件
REDIS_CONFIG="/tmp/redis_6380.conf"
cat > $REDIS_CONFIG << EOF
port 6380
bind 0.0.0.0
protected-mode no
daemonize yes
pidfile /tmp/redis_6380.pid
logfile /tmp/redis_6380.log
dir /tmp
EOF

echo "  Redis配置文件: $REDIS_CONFIG"

# 启动Redis
echo "  启动Redis..."
$REDIS_SERVER $REDIS_CONFIG

sleep 3

# 检查Redis进程
if pgrep -f "redis-server.*6380" > /dev/null; then
    echo "  ✅ Redis进程已启动"
else
    echo "  ❌ Redis进程未找到"
    echo "  查看Redis日志:"
    cat /tmp/redis_6380.log 2>/dev/null || echo "    无日志文件"
    exit 1
fi

# 检查Redis是否响应
echo "  测试Redis连接..."
if $REDIS_CLI -p 6380 ping 2>/dev/null | grep -q PONG; then
    echo "✅ Redis启动成功"
else
    echo "❌ Redis连接失败"
    echo "  Redis进程状态:"
    ps aux | grep redis-server | grep 6380
    echo "  端口状态:"
    netstat -tulpn 2>/dev/null | grep 6380 || ss -tulpn | grep 6380
    echo "  Redis日志:"
    tail -20 /tmp/redis_6380.log 2>/dev/null || echo "    无日志文件"
    exit 1
fi

# 2. 启动Ray
echo "⚡ 启动Ray..."
ray stop > /dev/null 2>&1  # 确保之前的Ray已停止
ray start --head --port=6379 --object-manager-port=8076 --node-manager-port=8077 > /dev/null 2>&1
sleep 3

# 检查Ray是否启动成功
if ray status > /dev/null 2>&1; then
    echo "✅ Ray启动成功"
else
    echo "❌ Ray启动失败"
    echo "  Ray状态:"
    ray status
    exit 1
fi

echo "🎉 基础服务启动完成！"
echo ""
echo "📝 服务状态检查:"
echo "  Redis端口: $(netstat -tulpn 2>/dev/null | grep 6380 || ss -tulpn | grep 6380)"
echo "  Ray状态: $(ray status | head -1)"
echo ""
echo "📝 接下来请在其他终端中启动："
echo "终端1 - 资源层："
echo "  cd /home/hustlbw/AgentOS/src/agentos/resource"
echo "  python api_server.py --redis_ip 127.0.0.1 --redis_port 6380 --flask_port 5000"
echo ""
echo "终端2 - 调度层："
echo "  cd /home/hustlbw/AgentOS/src/agentos/scheduler"
echo "  python scheduler.py --master_addr 127.0.0.1:5000 --redis_ip 127.0.0.1 --redis_port 6380 --strategy mlq --flask_port 5001"
echo ""
echo "终端3 - 运行测试："
echo "  cd /home/hustlbw/AgentOS/src/agentos/scheduler"
echo "  python dispatch_task.py"
echo ""
echo "🔧 停止服务命令:"
echo "  pkill -f redis-server"
echo "  ray stop"