# Maze Client SDK 使用文档

Maze Client SDK 是一个简洁易用的 Python 客户端库，用于创建和管理分布式工作流。

## 目录

- [核心概念](#核心概念)
- [安装与配置](#安装与配置)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [装饰器使用](#装饰器使用)
- [任务配置详解](#任务配置详解)
- [完整示例](#完整示例)

---

## 核心概念

### Workflow（工作流）

工作流是一个有向无环图（DAG），由多个任务和它们之间的依赖关系组成。工作流定义了任务的执行顺序和数据流向。

**特点：**
- 支持并行执行：没有依赖关系的任务可以同时运行
- 自动调度：根据任务依赖关系自动确定执行顺序
- 资源管理：根据任务资源需求智能分配计算资源

### Task（任务）

任务是工作流中的基本执行单元，每个任务包含：
- **输入参数**：任务执行所需的数据
- **执行代码**：任务的具体处理逻辑
- **输出参数**：任务执行后产生的结果
- **资源需求**：任务运行所需的 CPU、内存、GPU 等资源

**任务类型：**
- `code`：Python 代码任务（目前主要支持）
- 其他类型可扩展

### Edge（边）

边表示任务之间的依赖关系和数据流向：
- **依赖关系**：`A → B` 表示任务 B 必须在任务 A 完成后才能执行
- **数据传递**：任务 A 的输出可以作为任务 B 的输入

**自动边创建：**
当使用 `task2.outputs["key"]` 引用其他任务的输出时，系统会自动创建依赖边，无需手动调用 `add_edge()`。

---

## 安装与配置

### 前提条件

- Python 3.7+
- Maze 服务器运行中（默认 `http://localhost:8000`）

### 安装

```bash
pip install maze
```

### 导入模块

```python
from maze.client.maze.client import MaClient
from maze.client.maze.decorator import task
```

---

## 快速开始

### 基本流程

```python
from maze.client.maze.client import MaClient
from maze.client.maze.decorator import task

# 1. 定义任务函数
@task(
    inputs=["input_text"],
    outputs=["output_text"],
    resources={"cpu": 1, "cpu_mem": 128, "gpu": 0, "gpu_mem": 0}
)
def process_task(params):
    text = params.get("input_text")
    result = text.upper()  # 简单的处理逻辑
    return {"output_text": result}

# 2. 创建客户端并连接服务器
client = MaClient("http://localhost:8000")

# 3. 创建工作流
workflow = client.create_workflow()

# 4. 添加任务
task1 = workflow.add_task(
    process_task,
    inputs={"input_text": "hello world"}
)

# 5. 运行工作流
workflow.run()

# 6. 获取执行结果
for message in workflow.get_results():
    if message.get("type") == "finish_task":
        print(f"任务完成: {message.get('data')}")
```

### 多任务依赖示例

```python
@task(inputs=["input"], outputs=["output"])
def task_a(params):
    value = params.get("input")
    return {"output": value + " -> A"}

@task(inputs=["input"], outputs=["output"])
def task_b(params):
    value = params.get("input")
    return {"output": value + " -> B"}

# 创建工作流
client = MaClient()
workflow = client.create_workflow()

# 添加任务
t1 = workflow.add_task(task_a, inputs={"input": "start"})
t2 = workflow.add_task(task_b, inputs={"input": t1.outputs["output"]})

# t2 引用了 t1 的输出，自动建立依赖关系：t1 → t2

workflow.run()
for msg in workflow.get_results():
    print(msg)
```

---

## API 参考

### MaClient

客户端类，用于连接 Maze 服务器并创建工作流。

#### 构造函数

```python
MaClient(server_url: str = "http://localhost:8000")
```

**参数：**
- `server_url` (str): Maze 服务器地址，默认 `http://localhost:8000`

**示例：**
```python
# 连接本地服务器
client = MaClient()

# 连接远程服务器
client = MaClient("http://192.168.1.100:8000")
```

#### 方法

##### `create_workflow()`

创建一个新的工作流。

**返回值：**
- `MaWorkflow`: 工作流对象

**异常：**
- `Exception`: 创建失败时抛出

**示例：**
```python
workflow = client.create_workflow()
```

##### `get_workflow(workflow_id: str)`

获取已存在的工作流对象。

**参数：**
- `workflow_id` (str): 工作流 ID

**返回值：**
- `MaWorkflow`: 工作流对象

**示例：**
```python
workflow = client.get_workflow("workflow_12345")
```

##### `get_ray_head_port()`

获取 Ray 集群头节点端口信息（用于 Worker 连接）。

**返回值：**
- `dict`: 包含端口信息的字典

**示例：**
```python
port_info = client.get_ray_head_port()
print(port_info)
```

---

### MaWorkflow

工作流类，用于管理任务和执行流程。

#### 构造函数

通常不直接构造，而是通过 `MaClient.create_workflow()` 或 `MaClient.get_workflow()` 获取。

#### 属性

- `workflow_id` (str): 工作流唯一标识符
- `server_url` (str): 服务器地址

#### 方法

##### `add_task(task_func, inputs=None, task_name=None)`

添加任务到工作流（推荐使用装饰器方式）。

**参数：**
- `task_func` (Callable): 使用 `@task` 装饰的函数
- `inputs` (Dict[str, Any]): 输入参数字典，值可以是：
  - 直接值：`{"key": "value"}`
  - 其他任务的输出引用：`{"key": other_task.outputs["out_key"]}`
- `task_name` (str, optional): 任务名称，默认使用函数名

**返回值：**
- `MaTask`: 创建的任务对象

**示例：**
```python
# 简单任务
task1 = workflow.add_task(my_func, inputs={"input": "value"})

# 引用其他任务的输出
task2 = workflow.add_task(
    another_func,
    inputs={"input": task1.outputs["output"]}
)

# 自定义任务名
task3 = workflow.add_task(
    my_func,
    inputs={"input": "value"},
    task_name="自定义任务名称"
)
```

**注意：**
- 当 `inputs` 中引用了其他任务的输出（`TaskOutput`），系统会自动创建依赖边
- 旧的手动配置方式仍然支持，但不推荐使用

##### `get_tasks()`

获取工作流中的所有任务列表。

**返回值：**
- `List[Dict[str, str]]`: 任务列表，每个任务包含 `id` 和 `name`

**示例：**
```python
tasks = workflow.get_tasks()
for task in tasks:
    print(f"任务ID: {task['id']}, 名称: {task['name']}")
```

##### `add_edge(source_task, target_task)`

手动添加任务间的依赖边。

**参数：**
- `source_task` (MaTask): 源任务（前置任务）
- `target_task` (MaTask): 目标任务（后置任务）

**异常：**
- `Exception`: 添加失败时抛出

**示例：**
```python
workflow.add_edge(task1, task2)  # task1 → task2
```

**注意：**
- 使用任务输出引用时会自动创建边，通常不需要手动调用此方法

##### `del_edge(source_task, target_task)`

删除任务间的依赖边。

**参数：**
- `source_task` (MaTask): 源任务
- `target_task` (MaTask): 目标任务

**异常：**
- `Exception`: 删除失败时抛出

**示例：**
```python
workflow.del_edge(task1, task2)
```

##### `run()`

提交工作流执行请求。

**异常：**
- `Exception`: 提交失败时抛出

**示例：**
```python
workflow.run()
```

**注意：**
- 此方法只是提交执行请求，需要调用 `get_results()` 获取执行结果

##### `get_results(verbose=True)`

通过 WebSocket 实时获取工作流执行结果（生成器）。

**参数：**
- `verbose` (bool): 是否在控制台打印消息，默认 `True`

**返回值：**
- `Iterator[Dict[str, Any]]`: 消息生成器，每个消息包含：
  - `type` (str): 消息类型
    - `"start_task"`: 任务开始
    - `"finish_task"`: 任务完成
    - `"finish_workflow"`: 工作流完成
  - `data` (dict): 消息数据

**示例：**
```python
workflow.run()
for message in workflow.get_results():
    msg_type = message.get("type")
    msg_data = message.get("data", {})
    
    if msg_type == "start_task":
        print(f"▶ 任务开始: {msg_data.get('task_id')}")
        
    elif msg_type == "finish_task":
        print(f"✓ 任务完成: {msg_data.get('task_id')}")
        print(f"  结果: {msg_data.get('result')}")
        
    elif msg_type == "finish_workflow":
        print("🎉 工作流执行完成!")
        break
```

---

### MaTask

任务类，用于管理单个任务。

#### 属性

- `task_id` (str): 任务唯一标识符
- `workflow_id` (str): 所属工作流 ID
- `task_name` (str): 任务名称
- `outputs` (TaskOutputs): 输出参数集合，用于在任务间传递数据

#### 输出引用

通过 `task.outputs["key"]` 可以引用任务的输出参数：

```python
task1 = workflow.add_task(func1, inputs={"in": "value"})
task2 = workflow.add_task(
    func2,
    inputs={"in": task1.outputs["out"]}  # 引用 task1 的输出
)
```

#### 方法

##### `save(code_str, task_input, task_output, resources)`

手动保存任务配置（旧 API，不推荐使用）。

**参数：**
- `code_str` (str): 任务代码字符串
- `task_input` (dict): 输入参数配置
- `task_output` (dict): 输出参数配置
- `resources` (dict): 资源需求配置

**注意：**
- 使用 `@task` 装饰器和 `add_task()` 时会自动保存，无需手动调用

##### `delete()`

删除任务。

**异常：**
- `Exception`: 删除失败时抛出

**示例：**
```python
task.delete()
```

---

## 装饰器使用

### @task 装饰器

使用 `@task` 装饰器定义任务函数，自动提取函数代码、管理输入输出参数。

#### 语法

```python
@task(
    inputs: List[str],
    outputs: List[str],
    resources: Dict[str, Any] = None,
    data_types: Dict[str, str] = None
)
def task_function(params):
    # 任务逻辑
    return {"output_key": output_value}
```

#### 参数说明

- **`inputs`** (List[str]): 输入参数名列表
  - 定义任务需要的输入参数
  - 与 `params.get()` 中的键对应

- **`outputs`** (List[str]): 输出参数名列表
  - 定义任务返回字典中的键
  - 用于其他任务引用此任务的输出

- **`resources`** (Dict[str, Any], optional): 资源需求配置
  - `cpu` (int/float): CPU 核心数，默认 1
  - `cpu_mem` (int): CPU 内存（MB），默认 128
  - `gpu` (int/float): GPU 数量，默认 0
  - `gpu_mem` (int): GPU 内存（MB），默认 0

- **`data_types`** (Dict[str, str], optional): 参数数据类型
  - 指定输入/输出参数的数据类型
  - 默认全部为 `"str"`
  - 支持的类型：`"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"dict"` 等

#### 函数要求

被装饰的函数必须满足：
1. 接受一个 `params` 参数（字典）
2. 返回一个字典，包含所有声明的输出参数

#### 示例

```python
@task(
    inputs=["text", "count"],
    outputs=["result"],
    resources={"cpu": 2, "cpu_mem": 256, "gpu": 0, "gpu_mem": 0},
    data_types={"text": "str", "count": "int", "result": "str"}
)
def repeat_text(params):
    """将文本重复指定次数"""
    text = params.get("text")
    count = params.get("count", 1)
    
    result = text * count
    
    return {"result": result}
```

### 函数内导入

任务函数内可以导入第三方库：

```python
@task(inputs=["data"], outputs=["processed"])
def process_with_pandas(params):
    import pandas as pd
    import numpy as np
    
    data = params.get("data")
    df = pd.DataFrame(data)
    processed = df.describe().to_dict()
    
    return {"processed": processed}
```

**注意：**
- 使用 `cloudpickle` 序列化函数，支持导入和闭包
- Worker 环境需要安装相应的依赖库

---

## 任务配置详解

### 输入参数（inputs）

任务输入参数有两种来源：

#### 1. 用户直接输入（from_user）

```python
task = workflow.add_task(
    my_func,
    inputs={"input_key": "直接输入的值"}
)
```

内部转换为：
```python
{
    "input_schema": "from_user",
    "value": "直接输入的值"
}
```

#### 2. 引用其他任务输出（from_task）

```python
task1 = workflow.add_task(func1, inputs={"in": "value"})
task2 = workflow.add_task(
    func2,
    inputs={"in": task1.outputs["out"]}
)
```

内部转换为：
```python
{
    "input_schema": "from_task",
    "value": "{task_id}.output.{output_key}"
}
```

**自动边创建：**
- 当使用 `task.outputs["key"]` 引用时，系统自动创建依赖边
- 无需手动调用 `workflow.add_edge()`

### 输出参数（outputs）

任务函数必须返回包含所有声明输出参数的字典：

```python
@task(inputs=["a", "b"], outputs=["sum", "product"])
def calculate(params):
    a = params.get("a")
    b = params.get("b")
    
    return {
        "sum": a + b,
        "product": a * b
    }
```

其他任务可以引用：
```python
next_task = workflow.add_task(
    another_func,
    inputs={
        "total": task.outputs["sum"],
        "multiply": task.outputs["product"]
    }
)
```

### 资源配置（resources）

合理配置资源以优化任务执行：

```python
@task(
    inputs=["data"],
    outputs=["result"],
    resources={
        "cpu": 4,          # 4 个 CPU 核心
        "cpu_mem": 2048,   # 2GB CPU 内存
        "gpu": 1,          # 1 个 GPU
        "gpu_mem": 4096    # 4GB GPU 内存
    }
)
def heavy_computation(params):
    # 高性能计算任务
    pass
```

**建议：**
- 根据实际需求配置资源，避免浪费
- CPU 密集型任务：增加 `cpu` 核心数
- 内存密集型任务：增加 `cpu_mem`
- 深度学习任务：配置 `gpu` 和 `gpu_mem`

---

## 完整示例

### 示例 1：简单数据处理流水线

```python
from maze.client.maze.client import MaClient
from maze.client.maze.decorator import task

# 任务1：读取数据
@task(
    inputs=["file_path"],
    outputs=["raw_data"],
    resources={"cpu": 1, "cpu_mem": 128, "gpu": 0, "gpu_mem": 0}
)
def read_data(params):
    file_path = params.get("file_path")
    # 模拟读取数据
    data = f"Data from {file_path}"
    return {"raw_data": data}

# 任务2：数据清洗
@task(
    inputs=["data"],
    outputs=["cleaned_data"],
    resources={"cpu": 2, "cpu_mem": 256, "gpu": 0, "gpu_mem": 0}
)
def clean_data(params):
    data = params.get("data")
    cleaned = data.strip().upper()
    return {"cleaned_data": cleaned}

# 任务3：数据分析
@task(
    inputs=["data"],
    outputs=["analysis_result"],
    resources={"cpu": 4, "cpu_mem": 512, "gpu": 0, "gpu_mem": 0}
)
def analyze_data(params):
    data = params.get("data")
    result = f"Analysis: {len(data)} characters"
    return {"analysis_result": result}

# 创建并运行工作流
client = MaClient()
workflow = client.create_workflow()

# 添加任务并建立依赖
t1 = workflow.add_task(read_data, inputs={"file_path": "/data/input.txt"})
t2 = workflow.add_task(clean_data, inputs={"data": t1.outputs["raw_data"]})
t3 = workflow.add_task(analyze_data, inputs={"data": t2.outputs["cleaned_data"]})

# 执行流程：t1 → t2 → t3

workflow.run()

# 获取结果
for message in workflow.get_results():
    msg_type = message.get("type")
    if msg_type == "finish_task":
        data = message.get("data", {})
        print(f"✓ 任务完成: {data.get('task_id')}")
        print(f"  结果: {data.get('result')}\n")
    elif msg_type == "finish_workflow":
        print("🎉 工作流完成!")
        break
```

### 示例 2：并行任务处理

```python
from maze.client.maze.client import MaClient
from maze.client.maze.decorator import task
from datetime import datetime

# 数据源任务
@task(inputs=["source"], outputs=["data"])
def fetch_data(params):
    source = params.get("source")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"data": f"Data from {source} at {now}"}

# 处理任务A
@task(inputs=["input"], outputs=["output"])
def process_a(params):
    data = params.get("input")
    return {"output": f"{data} -> Processed by A"}

# 处理任务B
@task(inputs=["input"], outputs=["output"])
def process_b(params):
    data = params.get("input")
    return {"output": f"{data} -> Processed by B"}

# 合并任务
@task(inputs=["data_a", "data_b"], outputs=["merged"])
def merge_results(params):
    a = params.get("data_a")
    b = params.get("data_b")
    return {"merged": f"Merged: [{a}] + [{b}]"}

# 创建工作流
client = MaClient()
workflow = client.create_workflow()

# 构建并行处理流程
#        ┌─→ process_a ─┐
# fetch ─┤              ├─→ merge
#        └─→ process_b ─┘

t0 = workflow.add_task(fetch_data, inputs={"source": "database"})
t1 = workflow.add_task(process_a, inputs={"input": t0.outputs["data"]})
t2 = workflow.add_task(process_b, inputs={"input": t0.outputs["data"]})
t3 = workflow.add_task(
    merge_results,
    inputs={
        "data_a": t1.outputs["output"],
        "data_b": t2.outputs["output"]
    }
)

# t1 和 t2 可以并行执行（都依赖 t0）
# t3 等待 t1 和 t2 都完成后执行

workflow.run()

for message in workflow.get_results(verbose=False):
    if message.get("type") == "finish_workflow":
        print("并行处理完成!")
        break
```

### 示例 3：使用外部库

```python
from maze.client.maze.client import MaClient
from maze.client.maze.decorator import task

@task(
    inputs=["numbers"],
    outputs=["stats"],
    resources={"cpu": 2, "cpu_mem": 512, "gpu": 0, "gpu_mem": 0}
)
def calculate_statistics(params):
    """使用 NumPy 计算统计信息"""
    import numpy as np
    
    numbers = params.get("numbers")
    arr = np.array(numbers)
    
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }
    
    return {"stats": str(stats)}

# 使用
client = MaClient()
workflow = client.create_workflow()

task = workflow.add_task(
    calculate_statistics,
    inputs={"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
)

workflow.run()

for msg in workflow.get_results():
    if msg.get("type") == "finish_task":
        print("统计结果:", msg.get("data", {}).get("result"))
```

---

## 常见问题

### 1. 如何处理任务执行失败？

监听消息流中的错误信息：

```python
for message in workflow.get_results():
    if message.get("type") == "error":
        print(f"错误: {message.get('data')}")
```

### 2. 任务间如何传递复杂数据？

建议序列化为字符串或使用支持的数据类型：

```python
@task(inputs=["data"], outputs=["result"])
def process_json(params):
    import json
    
    data_str = params.get("data")
    data = json.loads(data_str)  # 反序列化
    
    # 处理...
    result = {"processed": True}
    
    return {"result": json.dumps(result)}  # 序列化
```

### 3. 如何调试任务代码？

在任务函数中添加日志：

```python
@task(inputs=["value"], outputs=["result"])
def debug_task(params):
    value = params.get("value")
    print(f"DEBUG: 输入值 = {value}")  # 会在 worker 日志中显示
    
    result = value * 2
    print(f"DEBUG: 输出值 = {result}")
    
    return {"result": result}
```

### 4. 可以在任务中访问文件系统吗？

可以，但需要注意：
- Worker 环境的文件路径
- 使用绝对路径或约定的相对路径
- 确保 Worker 有相应的读写权限

```python
@task(inputs=["filename"], outputs=["content"])
def read_file(params):
    filename = params.get("filename")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    return {"content": content}
```

---

## 最佳实践

1. **合理拆分任务**：每个任务完成单一职责，便于调试和复用

2. **明确输入输出**：使用清晰的参数命名，添加函数文档

3. **资源配置优化**：根据实际需求配置资源，避免资源浪费

4. **错误处理**：在任务函数中添加异常处理

5. **使用装饰器**：优先使用 `@task` 装饰器定义任务，代码更简洁

6. **输出引用**：使用 `task.outputs["key"]` 引用任务输出，自动管理依赖

7. **测试任务**：先测试单个任务函数，确保逻辑正确后再集成到工作流

---

## 更多资源

- **测试示例**：`test/` 目录下的测试文件
- **GitHub 仓库**：[Maze 项目地址]
- **技术支持**：[联系方式]

---

**版本信息**
- 文档版本：1.0
- SDK 版本：>=0.1.0
- 最后更新：2025


