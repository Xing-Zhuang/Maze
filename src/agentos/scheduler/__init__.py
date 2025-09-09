def cpu(cpu_num=1, mem=1):
    """CPU任务装饰器"""
    def decorator(func):
        # 将装饰器参数存储为函数属性
        func._task_decorator = {
            'type': 'cpu',
            'cpu_num': cpu_num,
            'mem': mem
        }
        
        def wrapper(*args, **kwargs):
            # 在wrapper中可以访问这些属性
            task_info = func._task_decorator
            print(f"执行CPU任务 - 核心数: {task_info['cpu_num']}, 内存: {task_info['mem']}MB")
            return func(*args, **kwargs)
        
        # 同时保留wrapper上的属性以便外部访问
        wrapper._task_decorator = func._task_decorator
        return wrapper
    return decorator

def io(mem=1):
    """IO任务装饰器"""
    def decorator(func):
        func._task_decorator = {
            'type': 'io',
            'cpu_num': 0,
            'mem': mem
        }
        
        def wrapper(*args, **kwargs):
            task_info = func._task_decorator
            print(f"执行IO任务 - 内存: {task_info['mem']}MB")
            return func(*args, **kwargs)
        
        wrapper._task_decorator = func._task_decorator
        return wrapper
    return decorator

def gpu(mem=0, gpu_mem=1, model_name="", backend="huggingface"):
    """GPU任务装饰器"""
    def decorator(func):
        func._task_decorator = {
            'type': 'gpu',
            'cpu_num': 0,
            'mem': mem,
            'gpu_mem': gpu_mem,
            'model_name': model_name,
            'backend': backend
        }
        
        def wrapper(*args, **kwargs):
            task_info = func._task_decorator
            print(f"执行GPU任务 - 显存: {task_info['gpu_mem']}MB, 模型: {task_info['model_name']}, 后端: {task_info['backend']}")
            return func(*args, **kwargs)
        
        wrapper._task_decorator = func._task_decorator
        return wrapper
    return decorator