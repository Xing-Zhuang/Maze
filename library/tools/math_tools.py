from typing import Tuple, List, Dict
# 导入新的统一 tool 装饰器
from maze.library.tools.definitions import tool

@tool(
    name="add",
    description="计算两个整数的和。",
    task_type='cpu',
    mem=256, # 基础内存需求
    input_parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "第一个加数。"
            },
            "b": {
                "type": "integer",
                "description": "第二个加数。"
            }
        },
        "required": ["a", "b"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "sum": {
                "type": "integer",
                "description": "两个整数的和。"
            }
        }
    }
)
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    print(f"Executing add(a={a}, b={b})")
    return a + b

@tool(
    name="calculate_stats",
    description="计算一个数字列表的平均值和总和。",
    task_type='cpu',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "description": "一个包含浮点数的列表。",
                "items": {"type": "number"}
            }
        },
        "required": ["numbers"]
    },
    output_parameters={
        "type": "array",
        "description": "一个包含两个元素的元组：(平均值, 总和)。",
        "items": [
            {"type": "number", "description": "列表中所有数字的平均值。"},
            {"type": "number", "description": "列表中所有数字的总和。"}
        ]
    }
)
def calculate_stats(numbers: List[float]) -> Tuple[float, float]:
    """计算列表的平均值和总和。返回 (平均值, 总和)。"""
    print(f"Executing calculate_stats(numbers={numbers})")
    total = sum(numbers)
    mean = total / len(numbers) if numbers else 0
    return mean, total