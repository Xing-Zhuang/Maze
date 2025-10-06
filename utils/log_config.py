import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(mode: str = 'server', log_dir_parent: str = None):
    """
    配置全局日志系统，兼容服务器和客户端模式，并屏蔽第三方库噪音。

    :param mode: 'server' 或 'client'。
                 'server'模式下日志写入项目统一的artifacts目录。
                 'client'模式下日志写入当前工作目录。
    :param log_dir_parent: 在 'server' 模式下，是 runtime_artifacts 目录的路径。
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        return

    logger.setLevel(logging.DEBUG)

    # --- 1. 根据模式确定日志文件路径 ---
    if mode == 'client':
        log_file_path = Path.cwd() / 'maze_client.log'
    else: # server mode
        if log_dir_parent:
            log_dir = Path(log_dir_parent) / 'logs'
        else:
            project_root = Path(__file__).resolve().parents[1]
            log_dir = project_root / 'runtime_artifacts' / 'logs'
        
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / 'maze.log'

    # --- 2. 配置控制台处理器 (StreamHandler) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- 3. 配置文件处理器 (RotatingFileHandler) ---
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- 4. (核心修改) 扩展对第三方库的日志噪音压制 ---
    # 将常用且日志输出非常频繁的库的级别调高，只看重要问题
    noisy_loggers = [
        "ray", "urllib3", "filelock", 
        "matplotlib", "h5py", "PIL", "asyncio"
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info(f"Logging system configured in '{mode}' mode. Log file at: {log_file_path}")