from typing import Optional

def tool(
    name: str,
    description: str,
    input_parameters: dict,
    output_parameters: dict,
    task_type: str = 'cpu',
    cpu_num: int = 1,
    mem: int = 1024,
    gpu_mem: int = 0,
    model_name: Optional[str] = None,
    backend: Optional[str] = 'huggingface'
):
    """
    一个统一的、功能强大的装饰器，用于为函数式工具附加所有元数据。
    它将所有元数据打包存放到函数的 '_tool_meta' 属性中。

    :param name: 工具的唯一名称。
    :param description: 工具功能的详细描述。
    :param input_parameters: 描述输入参数的字典，遵循JSON Schema格式。
    :param output_parameters: 描述输出参数的字典。
    :param task_type: 任务类型 ('cpu', 'gpu', 'io')。
    :param cpu_num: 所需CPU核心数。
    :param mem: 所需内存（单位MB）。
    :param gpu_mem: 所需GPU显存（单位MB）。
    :param model_name: 任务所需的模型名称。
    :param backend: 执行后端 ('huggingface', 'vllm')。
    """
    def decorator(func):
        # 将所有逻辑元数据和物理资源元数据打包到一个字典中
        func._tool_meta = {
            'name': name,
            'description': description,
            'input_parameters': input_parameters,
            'output_parameters': output_parameters,
            'resources': {
                'type': task_type,
                'cpu_num': cpu_num,
                'mem': mem,
                'gpu_mem': gpu_mem,
                'model_name': model_name,
                'backend': backend
            }
        }
        return func
    return decorator

# --- 定义的特殊类型字符串，供用户在定义参数时使用 ---
TYPE_FILEPATH = "filepath"
TYPE_FOLDERPATH = "folderpath"