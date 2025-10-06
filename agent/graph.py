import os
import re
import time
import uuid
import json
import redis
import shutil
import base64
import inspect
import tempfile
import threading
import requests
import cloudpickle
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict
from maze.agent.config import config
from dataclasses import dataclass, field, asdict

@dataclass
class TaskPayload:
    """单个任务的数据传输对象"""
    name: str
    serialized_func: bytes  # Cloudpickled 函数对象
    inputs: Dict[str, Any]
    resources: Dict[str, Any]
    meta: Dict[str, Any]

@dataclass
class WorkflowPayload:
    """整个工作流的数据传输对象"""
    dag_id: str
    name: str
    root_path: str
    graph_definition: Dict[str, Any]  # NetworkX 图的字典表示
    tasks: Dict[str, TaskPayload]     # 任务ID到任务Payload的映射

class DAG:
    """
    负责管理工作流图（DAG）的结构和可视化。
    """
    def __init__(self):
        self.id = f"dag_{uuid.uuid4().hex[:8]}"
        self._graph = nx.DiGraph()
        # _tasks 字典现在以 task_id 为键
        self._tasks = {}
        self.name = ""

    def validate(self):
        """
        对工作流图进行合规性检查，确保其结构正确。
        如果检查失败，则抛出 ValueError。
        """
        if not self._graph:
            # 如果图为空，则无需检查
            return

        # 检查 1: 是否为有向无环图 (DAG)
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Workflow validation failed: A cycle was detected in the graph. Workflows cannot have circular dependencies.")

        # 检查 2: 是否有唯一的“起始点”（入度为0的节点）
        source_nodes = [node for node, degree in self._graph.in_degree() if degree == 0]
        if len(source_nodes) != 1:
            raise ValueError(f"Workflow validation failed: Must have exactly one starting node (with no inputs), but found {len(source_nodes)}.")

        # 检查 3: 是否有唯一的“终点”（出度为0的节点）
        sink_nodes = [node for node, degree in self._graph.out_degree() if degree == 0]
        if len(sink_nodes) != 1:
            raise ValueError(f"Workflow validation failed: Must have exactly one final output node (with no outputs), but found {len(sink_nodes)}.")
        
        # 如果所有检查都通过，可以记录一条日志或直接返回
        print("Workflow validation successful.")

    def update_task(self, task_id: str, new_name: str = None, new_inputs: dict = None):
        """
        安全地更新一个已存在任务的属性，特别是其输入依赖。

        如果修改 inputs 导致工作流产生循环，操作将被回滚并抛出异常。
        """
        # 0. 检查任务是否存在
        if not self._graph.has_node(task_id):
            raise ValueError(f"Task with id '{task_id}' not found in the workflow.")
        # 1. 更新简单的元数据 (如 task_name)
        if new_name:
            self._tasks[task_id]['name'] = new_name
            print(f"Task '{task_id}' name updated to '{new_name}'.")

        # 2. 处理复杂的输入依赖更新
        if new_inputs is not None:
            # --- 开始“事务” ---
            # a. 备份旧状态，用于可能的回滚
            old_inputs = self._tasks[task_id].get('inputs', {})
            old_in_edges = list(self._graph.in_edges(task_id, data=True))
            try:
                # b. 应用新状态：先删除旧的边，再根据 new_inputs 添加新边
                self._graph.remove_edges_from(old_in_edges)
                self._tasks[task_id]['inputs'] = new_inputs
                for param, source in new_inputs.items():
                    source_list = source if isinstance(source, list) else [source]
                    for item_source in source_list:
                        if isinstance(item_source, str) and ".output" in item_source:
                            upstream_task_id = item_source.split('.')[0]
                            mapping = {param: item_source}
                            # 复用内部的 _add_edge_internal 方法
                            self._add_edge_internal(upstream_task_id, task_id, mapping)

                # c. 验证新图的合规性
                if not nx.is_directed_acyclic_graph(self._graph):
                    # 如果验证失败，触发回滚
                    raise ValueError("This update would create a cycle in the workflow.")
                # d. 如果验证成功，“事务”提交，操作完成
                print(f"Task '{task_id}' inputs updated successfully.")

            except Exception as e:
                # --- 4. 如果验证失败或发生任何异常，执行回滚 ---
                print(f"Failed to update task '{task_id}', rolling back changes. Reason: {e}")
                # 清理掉刚刚添加的（错误的）新边
                current_in_edges = list(self._graph.in_edges(task_id))
                self._graph.remove_edges_from(current_in_edges)
                # 恢复旧的边和旧的inputs定义
                self._graph.add_edges_from(old_in_edges)
                self._tasks[task_id]['inputs'] = old_inputs
                # 将原始异常重新抛出给上层
                raise ValueError(f"Failed to update task '{task_id}': {e}") from e

    def _add_node_internal(self, task_id, task_def):
        """内部方法：将任务数据存入字典，并将 task_id 添加到图中。"""
        if task_id in self._tasks:
            raise ValueError(f"任务ID '{task_id}' 已存在 (这通常不应发生)。")
        self._tasks[task_id] = task_def
        self._graph.add_node(task_id)

    def _add_edge_internal(self, source_task_id, target_task_id, mapping):
        """内部方法：根据 task_id 连接边。"""
        if not self._graph.has_node(source_task_id):
            raise ValueError(f"上游任务ID '{source_task_id}' 不存在。")
        if not self._graph.has_node(target_task_id):
            raise ValueError(f"下游任务ID '{target_task_id}' 不存在。")
        
        if self._graph.has_edge(source_task_id, target_task_id):
            self._graph[source_task_id][target_task_id]['mapping'].update(mapping)
        else:
            self._graph.add_edge(source_task_id, target_task_id, mapping=mapping)

    def remove_task(self, task_id: str):
        """
        安全地从工作流图中移除一个任务节点。

        在删除前会检查该节点是否有下游依赖（后继节点）。
        如果存在依赖，则会抛出 ValueError 异常，防止破坏图的结构。

        :param task_id: 由 add_task 方法返回的唯一任务ID。
        """
        # 1. 检查任务是否存在
        if not self._graph.has_node(task_id):
            raise ValueError(f"Task with id '{task_id}' not found in the workflow.")

        # 2. (核心) 检查是否存在下游依赖
        successors = list(self._graph.successors(task_id))
        if successors:
            # 如果存在后继节点，则拒绝删除，并给出清晰提示
            raise ValueError(
                f"Cannot remove task '{self._tasks[task_id]['name']}' ({task_id}), "
                f"because other tasks depend on it: {successors}. "
                "Please remove the dependent tasks first."
            )

        # 3. 如果检查通过，则执行删除
        self._graph.remove_node(task_id)
        self._tasks.pop(task_id, None)

    def add_task(self, func, task_name: str = None, inputs: dict = None, file_paths: list = None, resources: dict = None) -> str:
        """向工作流中添加一个任务节点，并自动读取其元数据。"""
        if not hasattr(func, '_tool_meta'): # 假设我们统一使用 _tool_meta
            raise TypeError(f"函数 '{func.__name__}' 不是一个有效的工具，请使用 @tool 装饰器进行定义。")

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        meta = func._tool_meta
        name = task_name or meta.get('name', func.__name__)

        # 合并元数据中的资源和用户指定的资源
        final_resources = {**meta.get('resources', {}), **(resources or {})}

        task_def = {
            "id": task_id, "name": name, "func": func,
            "inputs": inputs or {}, "file_paths": file_paths or [],
            "resources": final_resources, "meta": meta
        }
        self._add_node_internal(task_id, task_def)
        
        # 自动根据输入创建边
        if inputs:
            for param, source in inputs.items():
                source_list = source if isinstance(source, list) else [source]
                for item_source in source_list:
                    if isinstance(item_source, str) and ".output" in item_source:
                        upstream_task_id = item_source.split('.')[0]
                        mapping = {param: item_source}
                        self._add_edge_internal(upstream_task_id, task_id, mapping)
        return task_id

    def visualize(self, style_options: dict = None):
        """
        (最终修正版 v2) 可视化工作流图，并正确显示参数到参数的映射。
        """
        if not self._graph:
            print("工作流为空。")
            return
            
        opts = { 'figsize': (12, 8), 'title_fontsize': 16, 'node_size': 4000, 'node_fontsize': 8, 'edge_fontsize': 7, 'arrow_size': 15, 'line_width': 1.5, }
        if style_options:
            opts.update(style_options)

        pos = None
        try:
            layout_graph = nx.DiGraph(self._graph.edges())
            pos = nx.drawing.nx_pydot.graphviz_layout(layout_graph, prog="dot")
        except Exception as e:
            print(f"Graphviz 'dot' 布局失败({e})，将使用默认的弹簧布局。")
            pos = nx.spring_layout(self._graph, seed=42)

        labels = {task_id: task_def['meta'].get('name', task_id) for task_id, task_def in self._tasks.items()}

        plt.figure(figsize=opts['figsize'])
        
        node_opts = { "node_size": opts['node_size'], "node_color": "#a0c4ff", "edgecolors": "#333333", "linewidths": 1.5 }
        nx.draw_networkx(self._graph, pos, labels=labels, with_labels=True, **node_opts)
        
        edge_opts = { "edge_color": "#6c757d", "width": opts['line_width'], "arrowstyle": "-|>", "arrowsize": opts['arrow_size'], "arrows": True, }
        nx.draw_networkx_edges(self._graph, pos, **edge_opts)
        
        edge_label_opts = { "font_size": opts['edge_fontsize'], "font_color": "#c9184a", "bbox": dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2) }
        edge_labels = {}
        
        for u_id, v_id, data in self._graph.edges(data=True):
            mappings = []
            for target_param, source_ref in data.get("mapping", {}).items():
                source_output = source_ref.split('.', 1)[1] # "output" 或 "output[1]"
                
                try:
                    source_output_name = "unknown"
                    output_meta = self._tasks[u_id]['meta']['output_parameters']

                    if '[' in source_output and ']' in source_output:
                        # 情况1: 对于带索引的输出，直接使用 'output[index]' 作为名称
                        # 这是最准确的表示方式
                        index_str = source_output[source_output.find('[') + 1:source_output.find(']')]
                        source_output_name = f"output[{index_str}]"
                    else:
                        # 情况2: 对于单个输出，从 'properties' 中获取其键名
                        if 'properties' in output_meta:
                            source_output_name = list(output_meta['properties'].keys())[0]
                    
                    mappings.append(f"{source_output_name}-> {target_param}")

                except (ValueError, KeyError, IndexError):
                    mappings.append(f"{source_output}-> {target_param}")
            
            edge_labels[(u_id, v_id)] = "\n".join(mappings)
            
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels, **edge_label_opts)
        
        plt.title(f"Workflow: {self.name}", size=opts['title_fontsize'], weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_structure(self):
        """在终端中以文本形式打印工作流的结构，包含新元数据。"""
        if not self._graph:
            print("工作流为空。")
            return
            
        print(f"--- Workflow Structure: {self.name} (ID: {self.id}) ---")
        
        for task_id in nx.topological_sort(self._graph):
            task_def = self._tasks[task_id]
            # --- 核心修改：从 _tool_meta 中读取信息 ---
            meta = task_def.get('meta', {})
            task_name = meta.get('name', task_id)
            resources = meta.get('resources', {})
            
            print(f"\n[Task: {task_name}] (ID: {task_id})")
            print(f"  - Description: {meta.get('description', 'N/A')}")
            
            res_str = ", ".join([f"{k}: {v}" for k, v in resources.items()])
            print(f"  - Resources: {{ {res_str} }}")
            
            inputs = task_def.get('inputs', {})
            if not inputs:
                print("  - Inputs: None")
            else:
                print("  - Inputs:")
                for param, source in inputs.items():
                    print(f"    - {param}: <-- {source}")
            
            successors = list(self._graph.successors(task_id))
            if not successors:
                print("  - Outputs to: End of workflow")
            else:
                # 将后继ID转换为可读的Name
                succ_names = [self._tasks[succ_id]['meta'].get('name', succ_id) for succ_id in successors]
                print(f"  - Outputs to: {', '.join(succ_names)}")

@dataclass
class TaskResult:
    """用于存储单个任务执行结果和元数据的结构化对象。"""
    task_id: str
    task_name: str
    status: str = "pending"
    task_type: Optional[str] = None
    submit_time: Optional[float] = None
    finish_time: Optional[float] = None
    resources: dict = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.submit_time and self.finish_time:
            return self.finish_time - self.submit_time
        return None

class ExecuteDAG(DAG):
    """
    一个可执行的DAG类，封装了所有执行逻辑和结果管理。
    """
    def __init__(self, scheduler_addr: str = "127.0.0.1:6382", root_path: str = None):
        super().__init__()
        self.root_path= root_path
        self.scheduler_addr = f"http://{scheduler_addr}"
        self._local_run_results = {}
        self._local_run_threads = {}

    def submit(self, mode: str = 'local'):
        """
        提交工作流执行。
        - 'local' 模式: 在后台线程中启动本地执行，并立即返回run_id。
        - 'server' 模式: 将工作流提交到远程服务器。
        """
        self.validate()
        self.results = {
            task_id: TaskResult(task_id=task_id, task_name=t_def['name'], task_type=t_def['resources'].get('type'), resources=t_def['resources'])
            for task_id, t_def in self._tasks.items()
        }
        if mode == 'local':
            # 1. 在主线程中生成 run_id 以便立即返回
            run_id = f"local_{uuid.uuid4().hex}"
            # 2. 创建并启动一个后台线程来执行 _run_local
            local_run_thread = threading.Thread(
                target= self._run_local, 
                args= (run_id,) # 将 run_id 作为参数传递给线程
            )
            local_run_thread.daemon = True  # 设置为守护线程，主程序退出时线程也会退出
            self._local_run_threads[run_id] = local_run_thread
            local_run_thread.start()
            # 3. 立即返回，不等待执行完成
            return {"status": "success", "msg": "Local execution started in background.", "run_id": run_id}
        elif mode == 'server':
            return self._submit_to_server()
        else:
            raise ValueError(f"未知的执行模式: '{mode}'。")

    def _to_payload(self) -> WorkflowPayload:
        """
        将 ExecuteDAG 实例的核心数据提取到一个 WorkflowPayload DTO 中。
        """
        # 1. 提取任务定义
        tasks_payload: Dict[str, TaskPayload] = {}
        for task_id, definition in self._tasks.items():
            tasks_payload[task_id] = TaskPayload(
                name=definition['name'],
                # 直接序列化函数对象
                serialized_func=cloudpickle.dumps(definition['func']),
                inputs=definition['inputs'],
                resources=definition['resources'],
                meta=definition['meta']
            )

        # 2. 提取图结构
        graph_def = nx.node_link_data(self._graph)

        # 3. 构建并返回顶层 Payload 对象
        return WorkflowPayload(
            dag_id= self.id,
            name=self.name,
            root_path=self.root_path,
            graph_definition=graph_def,
            tasks=tasks_payload
        )

    def _resolve_single_source(self, source: str, context_data: dict) -> Any:
        """解析单个数据源引用字符串，例如 'task_id.output.key[index]'。"""
        # 1. 分离索引和基础部分
        index = None
        base_source = source
        index_match = re.search(r'\[(\d+)\]$', source)
        if index_match:
            index = int(index_match.group(1))
            base_source = source[:index_match.start()]
        
        # 2. 解析基础部分 (必须是 task_id.output.key 格式)
        parts = base_source.split('.')
        if len(parts) < 3:
            raise ValueError(f"无效的输入源格式: '{source}'。必须使用 'task_id.output.key' 的格式。")
            
        upstream_task_id, _, output_key = parts[0], parts[1], ".".join(parts[2:])
        base_output_key = f"{upstream_task_id}.output"

        if base_output_key not in context_data:
            raise ValueError(f"在上下文中找不到上游任务 '{upstream_task_id}' 的输出。")
            
        upstream_output_dict = context_data[base_output_key]
        
        if not isinstance(upstream_output_dict, dict):
            raise TypeError(f"任务 '{upstream_task_id}' 的输出不是字典，无法访问。")
            
        if output_key not in upstream_output_dict:
            raise KeyError(f"在任务 '{upstream_task_id}' 的输出中找不到键: '{output_key}'。")
            
        resolved_object = upstream_output_dict[output_key]

        # 3. 如果有索引，应用索引
        if index is not None:
            if isinstance(resolved_object, (list, tuple)):
                if index >= len(resolved_object):
                    raise IndexError(f"索引 {index} 超出范围。源: '{source}' 的列表长度为 {len(resolved_object)}。")
                return resolved_object[index]
            else:
                raise TypeError(f"尝试对非序列类型进行索引访问。源: '{source}'")
        
        return resolved_object

    def _resolve_inputs(self, task_id: str, context_data: dict) -> dict:
        """
        按优先级为函数参数寻找值：1. 用户指定输入 2. 全局配置 3. 函数默认值
        """
        task_def = self._tasks[task_id]
        func = task_def['func']
        inputs_def = task_def.get('inputs', {})
        resolved_kwargs = {}

        for param in inspect.signature(func).parameters.values():
            param_name = param.name
            # 优先级 1: 检查用户在 add_task 时提供的 'inputs' 字典
            if param_name in inputs_def:
                source = inputs_def[param_name]
                if isinstance(source, list):
                    resolved_list = []
                    for item_source in source:
                        if isinstance(item_source, str) and ".output" in item_source:
                            resolved_list.append(self._resolve_single_source(item_source, context_data))
                        else:
                            resolved_list.append(item_source)
                    resolved_kwargs[param_name] = resolved_list
                elif isinstance(source, str) and ".output" in source:
                    resolved_kwargs[param_name] = self._resolve_single_source(source, context_data)
                else:
                    resolved_kwargs[param_name] = source            
            # 如果在 inputs 中没找到，则进入 优先级2 和 3
            else:
                found_in_config = False
                # --- 核心修改：智能注入逻辑 ---
                # 优先级 2.1: 在 [paths] 配置节中按键名匹配
                paths_config = config.get('paths', {})
                if param_name in paths_config:
                    project_root = paths_config.get('project_root')
                    # 对路径进行拼接，得到绝对路径
                    full_path = os.path.join(project_root, paths_config[param_name])
                    resolved_kwargs[param_name] = full_path
                    found_in_config = True

                # 优先级 2.2: 如果在 paths 中没找到，则在 [online_apis] 的所有子节中匹配
                if not found_in_config and hasattr(config, 'online_apis'):
                    for api_profile in config.online_apis.values():
                        if param_name in api_profile:
                            resolved_kwargs[param_name] = api_profile[param_name]
                            found_in_config = True
                            break # 找到第一个匹配就使用
                # 优先级 3: 如果配置中也没找到，则使用函数定义的默认值
                if not found_in_config and param.default is not inspect.Parameter.empty:
                    resolved_kwargs[param_name] = param.default
        return resolved_kwargs

    def _run_local(self, run_id: str):
        """
        本地执行引擎。此方法在一个独立的线程中运行。
        """
        print(f"Starting local run with run_id: {run_id}")
        execution_order = list(nx.topological_sort(self._graph))
        context_data = {}
        
        for task_id in execution_order:
            task_result = self.results[task_id]
            task_def = self._tasks[task_id]
            
            task_result.status = "running"
            task_result.submit_time = time.time()
            
            try:
                print(f"▷ Preparing to run local task: '{task_def['name']}'")
                kwargs = self._resolve_inputs(task_id, context_data)
                result = task_def['func'](**kwargs)
                
                task_result.result = result
                task_result.status = "finished"
                if result is not None:
                    output_dict = {}
                    output_params = task_def.get('meta', {}).get('output_parameters', {})
                    keys = list(output_params.get('properties', {}).keys())
                    if len(keys) != 1:
                        raise ValueError(
                            f"任务 '{task_def['name']}' 的 @tool 装饰器中的 output_parameters "
                            f"必须且只能定义一个输出属性, 但现在定义了 {len(keys)} 个: {keys}"
                        )
                    output_key = keys[0]
                    output_dict[output_key] = result
                    context_data[f"{task_id}.output"] = output_dict
                print(f"✔ Local task '{task_def['name']}' finished successfully.")

            except Exception as e:
                task_result.status = "failed"
                task_result.error = str(e)
                print(f"❌ ERROR: Local task '{task_def['name']}' failed!", exc_info=True)
                break 
            finally:
                task_result.finish_time = time.time()
        
        # 将最终的上下文数据存入实例变量中，供 agent.get() 查询
        self._local_run_results[run_id] = context_data
        print(f"Local run {run_id} finished.")

    def _submit_to_server(self):
        """
        提取工作流数据到DTO，并与项目目录一起打包提交到服务器。
        """
        print(f"--- Submitting workflow to SERVER using DTO: {self.name} ---")

        if not self.root_path or not os.path.isdir(self.root_path):
            raise ValueError(f"Invalid 'root_path': {self.root_path}. It must be a valid directory.")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # 1. 提取工作流数据为 DTO
                workflow_dto = self._to_payload()
                print("  - Workflow data extracted into DTO.")

                # 2. 将 DTO 转换为字典，并对其中的二进制数据进行 Base64 编码
                payload_dict = asdict(workflow_dto)
                for task_id, task_data in payload_dict['tasks'].items():
                    task_data['serialized_func'] = base64.b64encode(task_data['serialized_func']).decode('utf-8')
                
                # 3. 将最终的字典转换为 JSON 字符串
                payload_json = json.dumps(payload_dict, indent=2)
                print("  - DTO serialized to JSON with Base64 encoded functions.")

                # 4. 打包项目目录
                archive_base_path = os.path.join(tmpdir, "project_archive")
                archive_path = shutil.make_archive(base_name=archive_base_path, format='zip', root_dir=self.root_path)
                print(f"  - Project directory '{self.root_path}' packed into '{archive_path}'")

                # 5. 构造并发送 multipart/form-data 请求
                files_payload = {
                    'workflow_payload': ('workflow.json', payload_json, 'application/json'),
                    'project_archive': ('project.zip', open(archive_path, 'rb'), 'application/zip')
                }
                
                submit_url = f"{self.scheduler_addr}/submit_agent/"
                
                print(f"  - Sending workflow payload and project archive to {submit_url}...")
                response = requests.post(submit_url, files=files_payload, timeout=600)
                response.raise_for_status()
                
                print("✅ Workflow submitted successfully to server!")
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"❌ Error submitting workflow to server: {e}")
                return None
            except Exception as e:
                print(f"❌ An unexpected error occurred during submission: {e}")
                raise

    def get_results_as_json(self, indent=4) -> str:
        """将所有任务的执行结果序列化为JSON字符串。"""
        # dataclasses可以很方便地转为字典
        from dataclasses import asdict
        results_dict = {tid: asdict(res) for tid, res in self.results.items()}
        return json.dumps(results_dict, indent=indent, default=str)