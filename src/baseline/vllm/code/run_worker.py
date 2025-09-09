import os
import json
import time
import base64  # 确保导入 base64
import asyncio
import inspect
import networkx as nx
import importlib.util
from itertools import cycle
from typing import Dict, List, Any, Optional, Callable

# ==============================================================================
# 1. VLLM 服务管理器 (与之前一致)
# ==============================================================================

class VLLMServiceManager:
    """管理VLLM服务节点信息，并提供负载均衡的端点访问。"""
    def __init__(self, endpoints_data: dict):
        self._endpoints: Dict[str, List[str]] = endpoints_data
        self._next_endpoint_iterators: Dict[str, cycle] = {}
        for model_name, base_urls in self._endpoints.items():
            if not base_urls: continue
            self._next_endpoint_iterators[model_name] = cycle(base_urls)
        print(f"✅ [Worker] VLLM服务管理器已初始化，共管理 {len(self._endpoints)} 个模型。")

    def get_next_endpoint(self, model_name: str) -> Optional[str]:
        """通过轮询方式获取指定模型的下一个可用API基础地址。"""
        iterator = self._next_endpoint_iterators.get(model_name)
        return next(iterator) if iterator else None

# ==============================================================================
# 2. DAG Worker (核心执行引擎 - 全新重写)
# ==============================================================================

class DAGWorker:
    """
    负责从队列消费任务、动态加载工作流并异步执行DAG的核心工作类。
    （此版本参考AutoGen Actor模式重写，以确保与task.py的兼容性）
    """
    def __init__(self, queue: asyncio.Queue, statuses: dict, results: dict, csv_logger: Callable, jsonl_logger: Callable, worker_args: 'argparse.Namespace', endpoints_data: dict):
        self.dag_queue = queue
        self.dag_statuses = statuses
        self.dag_results = results
        self.csv_logger = csv_logger
        self.jsonl_logger = jsonl_logger
        self.args = worker_args
        self.vllm_manager = VLLMServiceManager(endpoints_data)

    async def consume_tasks(self):
        """作为后台任务，持续从队列中获取并处理DAG。"""
        while True:
            dag_package = await self.dag_queue.get()
            uuid = dag_package["uuid"]
            
            try:
                self.dag_statuses[uuid].update({"status": "running", "start_time": time.time()})
                # 调用重写后的核心执行逻辑
                await self.execute_dag_pipeline(dag_package)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"❌ DAG {uuid} 执行失败: {e}\n{error_details}")
                finish_time = time.time()
                self.dag_statuses[uuid].update({"status": "error", "finish_time": finish_time, "error_message": str(e)})
                self.dag_results[uuid] = {"error": error_details}
                
                # 记录失败日志
                self._log_task_completion(dag_package, self.dag_statuses[uuid].get('start_time', 0), finish_time, status="error")
            finally:
                self.dag_queue.task_done()

    async def _load_dag_resources(self, task_body: dict) -> (nx.DiGraph, dict):
        """动态加载特定工作流的资源（dag.json 和 task.py）。(与之前版本逻辑一致)"""
        dag_source = task_body.get("dag_source")
        dag_type = task_body.get("dag_type")
        
        workflow_dir = os.path.join(self.args.dag_path, dag_source, dag_type)
        dag_json_path = os.path.join(workflow_dir, "dag.json")
        task_py_path = os.path.join(workflow_dir, "task.py")
        
        if not os.path.exists(dag_json_path):
            raise FileNotFoundError(f"DAG定义文件未找到: {dag_json_path}")
        if not os.path.exists(task_py_path):
            raise FileNotFoundError(f"任务实现文件未找到: {task_py_path}")

        with open(dag_json_path, 'r') as f:
            dag_data = json.load(f)
        dag_graph = nx.DiGraph()
        dag_graph.add_nodes_from(node["task"] for node in dag_data["nodes"])
        dag_graph.add_edges_from((edge["source"], edge["target"]) for edge in dag_data["edges"])
        
        module_name = f"baseline.workflows.{dag_source}.{dag_type}.task"
        spec = importlib.util.spec_from_file_location(module_name, task_py_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        
        tools = {node["task"]: getattr(task_module, node["task"]) for node in dag_data["nodes"]}
        
        print(f"  -> [Worker] 已为 {dag_source}/{dag_type} 动态加载DAG定义和工具集。")
        return dag_graph, tools

    def _log_task_completion(self, dag_package: dict, start_time: float, finish_time: float, status: str = "finished"):
        """统一的日志记录函数。"""
        uuid = dag_package["uuid"]
        task_body = dag_package.get("task_body", {})
        leave_time = time.time()

        time_log = {
            'dag_id': task_body.get('dag_id'), 'uuid': uuid,
            'sub_time': task_body.get('sub_time'), 'arrival_time': dag_package.get('arrival_time'),
            'start_exec_time': start_time, 'finish_exec_time': finish_time,
            'exec_time': finish_time - start_time, 'leave_time': leave_time,
            'completion_time': finish_time - dag_package.get('arrival_time', 0) if dag_package.get('arrival_time') else 0,
            'response_time': leave_time - task_body.get('sub_time', 0) if task_body.get('sub_time') else 0,
        }
        self.csv_logger(time_log)
        
        json_log = {**self.dag_results[uuid], "status": status}
        self.jsonl_logger(json_log)
        print(f"  -> [Worker] 已为任务 {uuid} 调用日志记录函数。")


    async def execute_dag_pipeline(self, dag_package: dict):
        """
        通用的DAG执行引擎 (参考AutoGen Actor模式重写)。
        """
        uuid = dag_package["uuid"]
        task_body = dag_package.get("task_body", {})
        start_exec_time = time.time()
        
        dag_graph, tools = await self._load_dag_resources(task_body)
        sink_nodes = [node for node, degree in dag_graph.out_degree() if degree == 0]

        # 1. 准备工作流上下文，并一次性解码所有文件
        workflow_context = {
            "dag_id": task_body.get('dag_id'),
            "question": task_body.get("question"),
            "args": self.args,
            "vllm_manager": self.vllm_manager
        }
        # 将Base64编码的字符串文件解码成bytes
        supplementary_files_str = task_body.get("supplementary_files", {})
        workflow_context["supplementary_files"] = {
            k: base64.b64decode(v) for k, v in supplementary_files_str.items()
        }
        print(f"  -> [Worker] 已为任务 {uuid} 解码所有文件内容。")

        print(f"  -> [Worker] 开始执行DAG: {workflow_context['dag_id']} (UUID: {uuid})")
        # 2. 循环执行，直到图为空
        while dag_graph.number_of_nodes() > 0:
            ready_tasks = [node for node, degree in dag_graph.in_degree() if degree == 0]
            if not ready_tasks:
                raise RuntimeError("DAG中存在环或无法执行的节点！")

            # 串行执行（一次一个），与参考代码逻辑对齐
            for task_name in ready_tasks:
                print(f"  -> [Worker] 执行任务: {task_name}")
                task_func = tools[task_name]
                sig = inspect.signature(task_func)
                
                # 准备参数
                kwargs = {param: workflow_context[param] for param in sig.parameters if param in workflow_context}

                # 异步化执行同步函数
                result = await asyncio.to_thread(task_func, **kwargs)
                
                if isinstance(result, dict):
                    workflow_context.update(result)
                
                # 为最终结果检索做准备，将每个任务的直接输出也存入上下文
                workflow_context[task_name] = result
                
                dag_graph.remove_node(task_name)

        finish_exec_time = time.time()
        print(f"  -> [Worker] DAG {uuid} 执行完毕。")
        
        # 3. 获取最终结果
        if not sink_nodes:
            raise Exception("工作流中未找到终止节点 (sink node)。")
        final_node_name = sink_nodes[0]
        final_result = workflow_context.get(final_node_name, {"error": f"Result for final node '{final_node_name}' not found."})

        # 4. 更新全局状态和结果
        self.dag_statuses[uuid].update({"status": "finished", "finish_time": finish_exec_time})
        self.dag_results[uuid] = {
            "dag_id": task_body.get('dag_id'), 
            "uuid": uuid, 
            "final_answer": final_result
        }
        
        # 5. 记录日志
        self._log_task_completion(dag_package, start_exec_time, finish_exec_time, status="finished")