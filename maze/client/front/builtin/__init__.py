"""
内置任务库

这个包包含了预定义的任务函数，使用@task装饰器标记
"""

from maze.client.front.builtin import simpleTask
from maze.client.front.builtin import fileTask
from maze.client.front.builtin import healthTask

__all__ = ['simpleTask', 'fileTask', 'healthTask']

