# maze/decorators.py

def cpu(cpu_num=1, mem=1024):
    """
    CPU任务装饰器。
    :param cpu_num: 所需CPU核心数。
    :param mem: 所需内存（单位MB）。
    """
    def decorator(func):
        # 将资源需求直接附加到函数对象的属性上
        func._task_decorator = {
            'type': 'cpu',
            'cpu_num': cpu_num,
            'mem': mem
        }
        return func
    return decorator

def io(mem=1024):
    """
    IO任务装饰器。
    :param mem: 所需内存（单位MB）。
    """
    def decorator(func):
        func._task_decorator = {
            'type': 'io',
            'cpu_num': 1, # IO任务通常也需要少量CPU
            'mem': mem
        }
        return func
    return decorator

def gpu(mem=1024, gpu_mem=2048, model_name="", backend="huggingface"):
    """
    GPU任务装饰器。
    :param mem: 所需内存（单位MB）。
    :param gpu_mem: 所需GPU显存（单位MB）。
    :param model_name: 任务所需的模型名称。
    :param backend: 执行后端 (e.g., 'huggingface', 'vllm')。
    """
    def decorator(func):
        func._task_decorator = {
            'type': 'gpu',
            'cpu_num': 1, # GPU任务也需要CPU进行数据预处理等
            'mem': mem,
            'gpu_mem': gpu_mem,
            'model_name': model_name,
            'backend': backend
        }
        return func
    return decorator