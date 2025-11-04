"""
任务装饰器，用于定义任务的元数据和配置
"""

import inspect
import textwrap
import cloudpickle
import base64
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class TaskMetadata:
    """任务元数据"""
    func: Callable
    func_name: str
    code_str: str
    code_ser: str  # 序列化的函数（使用 cloudpickle）
    inputs: List[str]
    outputs: List[str]
    resources: Dict[str, Any]
    data_types: Dict[str, str]  # 参数的数据类型
    node_type: str  # 节点类型: 'task' 或 'tool'


def task(inputs: List[str], 
         outputs: List[str],
         resources: Dict[str, Any],
         data_types: Dict[str, str] = None):
    """
    任务装饰器 - 用于资源密集型任务（LLM调用、图像处理、模型推理等）
    
    Args:
        inputs: 输入参数名列表
        outputs: 输出参数名列表
        resources: 资源需求配置（必需），例如 {"cpu": 2, "cpu_mem": 2048, "gpu": 1, "gpu_mem": 4096}
        data_types: 参数数据类型映射，默认全部为 "str"
        
    示例:
        @task(
            inputs=["text"],
            outputs=["result"],
            resources={"cpu": 2, "cpu_mem": 2048, "gpu": 1, "gpu_mem": 4096}
        )
        def call_llm(params):
            text = params.get("text")
            # 调用LLM处理
            return {"result": processed_result}
    """
    def decorator(func: Callable) -> Callable:
        # 获取函数源代码（不包含装饰器）
        source_lines = inspect.getsourcelines(func)[0]
        
        # 找到函数定义的开始（跳过装饰器行）
        func_start_idx = 0
        for idx, line in enumerate(source_lines):
            if line.strip().startswith('def '):
                func_start_idx = idx
                break
        
        # 提取从函数定义开始的代码
        func_lines = source_lines[func_start_idx:]
        code_str = ''.join(func_lines)
        
        # 去除多余的缩进（处理嵌套函数的情况）
        code_str = textwrap.dedent(code_str)
        
        # 使用 cloudpickle 序列化整个函数（包括外部 import 和依赖）
        code_ser = base64.b64encode(cloudpickle.dumps(func)).decode('utf-8')
        
        # 默认数据类型都是str
        if data_types is None:
            types_config = {param: "str" for param in inputs + outputs}
        else:
            types_config = {param: "str" for param in inputs + outputs}
            types_config.update(data_types)
        
        # 创建元数据
        metadata = TaskMetadata(
            func=func,
            func_name=func.__name__,
            code_str=code_str,
            code_ser=code_ser,
            inputs=inputs,
            outputs=outputs,
            resources=resources,
            data_types=types_config,
            node_type="task"  # 标记为 task 类型
        )
        
        # 将元数据附加到函数上
        func._maze_task_metadata = metadata
        
        return func
    
    return decorator


def tool(inputs: List[str], 
         outputs: List[str],
         data_types: Dict[str, str] = None):
    """
    工具装饰器 - 用于轻量级工具任务（数据转换、格式化、简单计算等）
    
    Tool任务不需要指定resources，会使用最小默认资源配置
    
    Args:
        inputs: 输入参数名列表
        outputs: 输出参数名列表
        data_types: 参数数据类型映射，默认全部为 "str"
        
    示例:
        @tool(
            inputs=["data"],
            outputs=["formatted_data"]
        )
        def format_json(params):
            data = params.get("data")
            # 简单的数据处理
            return {"formatted_data": processed_data}
    """
    def decorator(func: Callable) -> Callable:
        # 获取函数源代码（不包含装饰器）
        source_lines = inspect.getsourcelines(func)[0]
        
        # 找到函数定义的开始（跳过装饰器行）
        func_start_idx = 0
        for idx, line in enumerate(source_lines):
            if line.strip().startswith('def '):
                func_start_idx = idx
                break
        
        # 提取从函数定义开始的代码
        func_lines = source_lines[func_start_idx:]
        code_str = ''.join(func_lines)
        
        # 去除多余的缩进（处理嵌套函数的情况）
        code_str = textwrap.dedent(code_str)
        
        # 使用 cloudpickle 序列化整个函数（包括外部 import 和依赖）
        code_ser = base64.b64encode(cloudpickle.dumps(func)).decode('utf-8')
        
        # Tool任务使用最小资源配置
        resources_config = {"cpu": 1, "cpu_mem": 128, "gpu": 0, "gpu_mem": 0}
        
        # 默认数据类型都是str
        if data_types is None:
            types_config = {param: "str" for param in inputs + outputs}
        else:
            types_config = {param: "str" for param in inputs + outputs}
            types_config.update(data_types)
        
        # 创建元数据
        metadata = TaskMetadata(
            func=func,
            func_name=func.__name__,
            code_str=code_str,
            code_ser=code_ser,
            inputs=inputs,
            outputs=outputs,
            resources=resources_config,
            data_types=types_config,
            node_type="tool"  # 标记为 tool 类型
        )
        
        # 将元数据附加到函数上
        func._maze_task_metadata = metadata
        
        return func
    
    return decorator


def get_task_metadata(func: Callable) -> TaskMetadata:
    """
    获取函数的任务元数据
    
    Args:
        func: 被@task或@tool装饰的函数
        
    Returns:
        TaskMetadata: 任务元数据
        
    Raises:
        ValueError: 如果函数没有被@task或@tool装饰
    """
    if not hasattr(func, '_maze_task_metadata'):
        raise ValueError(f"函数 {func.__name__} 没有使用 @task 或 @tool 装饰器")
    
    return func._maze_task_metadata

