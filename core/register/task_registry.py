# maze/core/register/task_registry.py

import os
import importlib
import inspect
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TaskRegistry:
    def __init__(self):
        self._tasks = {} # Renamed from _tools

    def register_task(self, func: callable): # Renamed from register_tool
        """
        注册一个被 @tool 装饰器包装的函数作为任务。
        """
        if not hasattr(func, '_tool_meta'):
            raise TypeError(f"Function '{func.__name__}' is not a valid Maze Task. Please decorate it with @tool.")
        
        task_name = func._tool_meta.get('name')
        if not task_name:
            raise ValueError(f"The @tool decorator for function '{func.__name__}' must include a 'name' attribute.")

        if task_name in self._tasks:
            logger.warning(f"Task name '{task_name}' is already registered. The existing task will be overwritten.")

        self._tasks[task_name] = func
        logger.info(f"  -> Successfully registered task: {task_name}")

    def discover_tasks(self, tasks_dir="maze/library/tasks"): # Renamed from discover_tools
        """
        自动扫描并注册指定目录下的所有任务函数。
        """
        logger.info(f"Starting auto-discovery of tasks in '{tasks_dir}'...")
        tasks_path = Path(tasks_dir)
        # 修正路径拼接逻辑
        package_prefix = tasks_dir.replace('/', '.')

        for file_path in tasks_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            module_name = f"{package_prefix}.{file_path.stem}"
            try:
                module = importlib.import_module(module_name)
                for name, member in inspect.getmembers(module):
                    if inspect.isfunction(member) and hasattr(member, '_tool_meta'):
                        self.register_task(member)
            except Exception as e:
                logger.error(f"Failed to discover tasks from module '{module_name}': {e}", exc_info=True)

    def get_task(self, name: str): # Renamed from get_tool
        """通过名称获取任务函数。"""
        task = self._tasks.get(name)
        if not task:
            raise AttributeError(f"Task '{name}' not found in the registry.")
        return task

    def __getattr__(self, name: str):
        """允许通过属性访问方式获取任务，例如 registry.add"""
        try:
            return self.get_task(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# 创建一个全局唯一的注册表实例
task_registry = TaskRegistry() # Renamed from tool_registry