import subprocess
import os
import tomli
from pathlib import Path
import click
import sys

# 确定项目根目录 (假设 cli.py 在 maze/ 目录下)
# 这使得脚本可以从任何地方正确找到配置文件
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # 在某些交互式环境（如REPL）中 __file__ 未定义
    PROJECT_ROOT = Path(".").resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "maze" / "config" / "config.toml"

def run_command(command, detached=False):
    """一个辅助函数，用于执行并打印系统命令"""
    click.echo(click.style(f"执行命令: {' '.join(command)}", fg="yellow"))
    try:
        if detached:
            # 在后台以分离模式运行（例如 Ray head/worker）
            # 注意：这在Windows上的行为可能不同
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            # 在前台运行并等待完成（例如 maze_main 进程）
            subprocess.run(command, check=True)
        click.echo(click.style("命令执行成功。", fg="green"))
    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"错误: 命令执行失败，返回码 {e.returncode}", fg="red"))
        click.echo(e.output)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(click.style(f"错误: 命令 '{command[0]}' 未找到。请确保 Ray 已正确安装并且在您的 PATH 中。", fg="red"))
        sys.exit(1)

@click.group()
def main():
    """Maze 分布式框架的命令行接口。"""
    pass

@main.command()
@click.option('--head', is_flag=True, help="启动中心节点 (Master Node)。")
@click.option('--worker', is_flag=True, help="启动从节点 (Worker Node)。")
def start(head, worker):
    """启动 Maze 集群节点。"""
    if not head and not worker:
        click.echo(click.style("错误: 请指定启动模式，--head 或 --worker", fg="red"))
        return

    if head:
        click.echo(click.style("🚀 正在启动 Maze 中心节点 (Master Node)...", fg="cyan"))
        
        # 1. 启动 Ray head
        ray_head_cmd = ["ray", "start", "--head", "--port=6379"]
        run_command(ray_head_cmd, detached=True)
        click.echo(click.style("✅ Ray Head 进程已在后台启动。", fg="green"))

        # 2. 自动运行 main.py
        click.echo(click.style("🚀 正在启动 Maze 中央调度服务 (main.py)...", fg="cyan"))
        main_py_path = str(PROJECT_ROOT / "maze" / "main.py")
        run_command([sys.executable, main_py_path])

    if worker:
        click.echo(click.style("🚀 正在启动 Maze 从节点 (Worker Node)...", fg="cyan"))

        # 1. 从 config.toml 读取中心节点 IP 和端口
        try:
            with open(CONFIG_PATH, "rb") as f:
                config = tomli.load(f)
            server_config = config.get("server", {})
            head_ip = server_config.get("host")
            # Ray 的端口是 Ray Head 的端口，而不是 Flask 服务的端口
            ray_port = "6379" 
            
            if not head_ip:
                click.echo(click.style(f"错误: 未在 {CONFIG_PATH} 的 [server] 部分找到 'host' 配置。", fg="red"))
                return
            
            head_address = f"{head_ip}:{ray_port}"

        except FileNotFoundError:
            click.echo(click.style(f"错误: 配置文件 {CONFIG_PATH} 未找到。", fg="red"))
            return
        except Exception as e:
            click.echo(click.style(f"读取配置文件时出错: {e}", fg="red"))
            return

        # 2. 启动 Ray worker
        ray_worker_cmd = ["ray", "start", f"--address={head_address}"]
        run_command(ray_worker_cmd, detached=True)
        click.echo(click.style(f"✅ Ray Worker 进程已在后台启动，并尝试连接到 {head_address}。", fg="green"))
        click.echo("请在新终端中使用 'ray status' 命令检查集群状态。")

if __name__ == '__main__':
    main()
