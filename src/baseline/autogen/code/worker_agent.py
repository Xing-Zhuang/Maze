import os
import sys
import csv
import time
import json
import fcntl # 用于文件锁
import socket
import base64
import inspect
import asyncio
import argparse
import traceback
import pytz
from datetime import datetime
import importlib.util
import networkx as nx
from pydantic import BaseModel
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
from autogen_core import MessageContext, RoutedAgent, message_handler, default_subscription, AgentId

TASK_LOG_FILE = "task_granularity_exec_time.csv"
TASK_LOG_HEADER = ["dag_id", "type", "dag_uuid", "task_name", "start_time_unix", "end_time_unix", "duration_s"]
def initialize_task_log_if_needed():
    """如果日志文件不存在，则创建并写入表头。使用文件锁保证线程/进程安全。"""
    if not os.path.exists(TASK_LOG_FILE):
        with FileLock(): # 使用您已有的文件锁，防止并发创建
            if not os.path.exists(TASK_LOG_FILE): # 双重检查，更安全
                with open(TASK_LOG_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(TASK_LOG_HEADER)
                    print(f"INFO: Created task granularity log file: {TASK_LOG_FILE}")

# --- 消息定义 ---
class DAGMessage(BaseModel):
    dag_id: str
    uuid: str
    type: str
    question: str
    sub_time: Optional[float] = 0
    arrival_time: Optional[float] = 0
    start_time: Optional[float] = 0
    end_time: Optional[float] = 0
    arg_src: Dict
    result: Optional[str] = None

@dataclass
class AckMessage():
    status: str = "ok"
    comment: str = ""

class FileLock:
    """一个使用 fcntl 实现的跨进程文件锁上下文管理器。"""
    def __init__(self, lock_dir="./lock"): # 传入目录而非完整路径
        # 在构造时只确定路径，不打开文件
        hostname = socket.gethostname()
        self._lock_file_path = os.path.join(lock_dir, f"{hostname}.lock")
        self._lock_file = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self._lock_file_path), exist_ok=True)

    def __enter__(self):
        # 在进入 with 代码块时才打开文件和加锁
        self._lock_file = open(self._lock_file_path, 'w')
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 在退出 with 代码块时解锁和关闭
        if self._lock_file:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

# --- 1. 基类定义 (通用DAG执行引擎) ---
class BaseActorAgent(RoutedAgent):
    """
    一个实现了 Actor 模型的、通用的 DAG 执行引擎基类。
    子类只需通过定义 self.tools 和 self.workflow 来配置具体的工作流。
    """
    def __init__(self, id: str):
        super().__init__(id)
        self._task_queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._work_loop())
        # 这两个属性需要由子类来定义
        self.tools: Dict[str, callable] = {}
        self.workflow: Dict[str, Any] = {}
        initialize_task_log_if_needed()
        print(f"✅ Actor基类 '{id}' 已初始化，内部队列和工作循环已启动。")

    async def _process_workflow(self, message: DAGMessage, ctx: MessageContext) -> DAGMessage:
        """
        通用的、使用 networkx 实现的 DAG 执行逻辑。
        它会使用子类定义的 self.tools 和 self.workflow。
        """
        cst_tz = pytz.timezone('Asia/Shanghai')
        start_time= time.time()
        execution_result = ""
        try:
            if not self.tools or not self.workflow:
                raise ValueError("子类必须定义 self.tools 和 self.workflow 属性。")

            # 1. 构建用于执行的图 (会被消耗)
            exec_graph = nx.DiGraph()
            exec_graph.add_nodes_from(node["task"] for node in self.workflow["nodes"])
            exec_graph.add_edges_from((edge["source"], edge["target"]) for edge in self.workflow["edges"])
            sink_nodes = [node for node, degree in exec_graph.out_degree() if degree == 0]

            # 2. 初始化工作流上下文（全局内存）
            workflow_context = {
                **message.arg_src, 
                "dag_id": message.dag_id, 
                "question": message.question,
                "args": argparse.Namespace(**json.loads(message.arg_src["args"])) if "args" in message.arg_src and message.arg_src.get("args") else None
            }
            if "supplementary_files" in workflow_context and workflow_context.get("supplementary_files"):
                 workflow_context["supplementary_files"] = {k: base64.b64decode(v) for k, v in workflow_context["supplementary_files"].items()}
            print(f"💢 supplementary_files已经加载完成")
            # 3. 循环执行，直到所有任务完成
            with FileLock():
                while exec_graph.number_of_nodes() > 0:
                    ready_tasks = [node for node, degree in exec_graph.in_degree() if degree == 0]
                    
                    if not ready_tasks:
                        raise Exception("DAG 中存在环或无法执行的节点！")
                    task_name_to_run = ready_tasks[0]
                    print(f"  -> [串行执行] 下一个任务: '{task_name_to_run}'")
                    task_func = self.tools[task_name_to_run]
                    task_start_time_obj = datetime.now(cst_tz)
                    task_start_time_str = task_start_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                    print("\n--- DEBUGGING INFO ---")
                    import inspect
                    print(f"Function to run: {task_func.__name__}")
                    try:
                        print(f"Source file Python is using: {inspect.getsourcefile(task_func)}")
                        print(f"Signature Python is seeing: {inspect.signature(task_func)}")
                    except TypeError:
                        print("Could not inspect the source file (might be a built-in or C module).")
                    print("--- END DEBUGGING INFO ---\n")
                    sig = inspect.signature(task_func)
                    kwargs = {key: workflow_context[key] for key in sig.parameters if key in workflow_context}
                    result = await asyncio.to_thread(task_func, **kwargs)

                    task_end_time_obj = datetime.now(cst_tz)
                    task_end_time_str = task_end_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                    duration = (task_end_time_obj - task_start_time_obj).total_seconds()
                    log_row = [message.dag_id, message.type, message.uuid, task_name_to_run, task_start_time_str, task_end_time_str, duration]
                    with open(TASK_LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(log_row)

                    if isinstance(result, dict):
                        workflow_context.update(result)
                    print(f"  -> [串行执行] 任务 '{task_name_to_run}' 完成")
                    workflow_context[task_name_to_run] = result
                    exec_graph.remove_node(task_name_to_run) # 从图中移除这一个已完成的任务

            # 4. MODIFIED: 改进的最终结果获取逻辑
            if not sink_nodes:
                raise Exception("工作流中未找到终止节点。")
            
            final_node_name = sink_nodes[0]
            # 直接获取该终止节点的执行结果
            final_result = workflow_context.get(final_node_name, f"Result not found for final node({final_node_name}).")
            
            # 将结果序列化为字符串
            try:
                execution_result = json.dumps(final_result, indent=2, ensure_ascii=False)
            except TypeError:
                # 如果 final_result 包含无法序列化的对象，则回退到字符串表示
                execution_result = json.dumps(str(final_result))

        except Exception:
            error_details = traceback.format_exc()
            print(f"❌ [{self.id}] 工作流 {message.uuid} 执行失败:\n{error_details}")
            execution_result = f"EXECUTION_ERROR: {error_details}"
        finally:
            end_time= time.time()
            return DAGMessage(
                dag_id= message.dag_id, type= message.type, question= message.question, uuid= message.uuid, sub_time= message.sub_time, arrival_time= message.arrival_time,
                start_time= start_time, end_time= end_time, arg_src= {}, result= execution_result
            )

    @message_handler
    async def handle_message(self, message: DAGMessage, ctx: MessageContext) -> AckMessage:
        # MODIFIED: 接收指令，返回一个有效的、可序列化的 AckMessage 作为“收条”
        if message.type == self.workflow_type:
            asyncio.create_task(self._task_queue.put((message, ctx)))
            return AckMessage(comment=f"Task {message.uuid} enqueued by {self.id}.")
        return AckMessage(status="ignored", comment="Workflow type mismatch.")

    async def _work_loop(self):
        print(f"[{self.id}] 后台工作循环已启动...")
        while True:
            try:
                message, ctx = await self._task_queue.get()
                response_message = await self._process_workflow(message, ctx)
                print(f"💚 获取response_message: {response_message}")
                # MODIFIED: 使用正确的 self.send_message 方法
                if response_message:
                    final_ack= await self.send_message(response_message, recipient= AgentId("master_dispatcher", "default"))
                    print(f"   -> [{self.id}] 收到 Master 的最终签收回执: {final_ack.comment if isinstance(final_ack, AckMessage) else 'N/A'}")
            except asyncio.CancelledError: break
            except Exception as e: 
                print(f"💥 [{self.id}] 工作循环出现严重错误: {e}\n{traceback.format_exc()}")

# --- 2. 具体实现类 (无需改动) ---
@default_subscription
class GAIA_File_Process_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.gaia.file.task import task1_file_process, \
            task2_llm_process_qwen, task3_llm_process_deepseek, task4_fuse_llm_answer
        self.workflow_type= workflow_type
        self.tools = {
            "task1_file_process": task1_file_process,
            "task2_llm_process_qwen": task2_llm_process_qwen,
            "task3_llm_process_deepseek": task3_llm_process_deepseek,
            "task4_fuse_llm_answer": task4_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_file_process", "target": "task2_llm_process_qwen"},
                {"source": "task1_file_process", "target": "task3_llm_process_deepseek"},
                {"source": "task2_llm_process_qwen", "target": "task4_fuse_llm_answer"},
                {"source": "task3_llm_process_deepseek", "target": "task4_fuse_llm_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


@default_subscription
class GAIA_Reason_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.gaia.reason.task import task1_obtain_content, \
            task2_llm_process_qwen, task3_llm_process_deepseek, task4_fuse_llm_answer              
        self.workflow_type= workflow_type
        self.tools = {
            "task1_obtain_content": task1_obtain_content,
            "task2_llm_process_qwen": task2_llm_process_qwen,
            "task3_llm_process_deepseek": task3_llm_process_deepseek,
            "task4_fuse_llm_answer": task4_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_obtain_content", "target": "task2_llm_process_qwen"},
                {"source": "task1_obtain_content", "target": "task3_llm_process_deepseek"},
                {"source": "task2_llm_process_qwen", "target": "task4_fuse_llm_answer"},
                {"source": "task3_llm_process_deepseek", "target": "task4_fuse_llm_answer"}    
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


@default_subscription
class GAIA_Speech_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.gaia.speech.task import task1_speech_process, task2_speech_recognition, \
            task3_llm_process_qwen, task4_llm_process_deepseek, task5_fuse_llm_answer              
        self.workflow_type= workflow_type
        self.tools = {
            "task1_speech_process": task1_speech_process,
            "task2_speech_recognition": task2_speech_recognition,
            "task3_llm_process_qwen": task3_llm_process_qwen,
            "task4_llm_process_deepseek": task4_llm_process_deepseek,
            "task5_fuse_llm_answer": task5_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_speech_process", "target": "task2_speech_recognition"},
                {"source": "task2_speech_recognition", "target": "task3_llm_process_qwen"},
                {"source": "task2_speech_recognition", "target": "task4_llm_process_deepseek"},        
                {"source": "task3_llm_process_qwen", "target": "task5_fuse_llm_answer"},
                {"source": "task4_llm_process_deepseek", "target": "task5_fuse_llm_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


@default_subscription
class GAIA_Vision_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.gaia.vision.task import task1_obtain_content, task2_vlm_process, task3_output_final_answer           
        self.workflow_type= workflow_type
        self.tools = {
            "task1_obtain_content": task1_obtain_content,
            "task2_vlm_process": task2_vlm_process,
            "task3_output_final_answer": task3_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_obtain_content", "target": "task2_vlm_process"},
                {"source": "task2_vlm_process", "target": "task3_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class tbench_airline_book_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.airline_book.task import task0_init, task1_llm_process, task2a_search_direct_flight, task2b_search_onestop_flight, task2c_get_user_details, task3_llm_process_filter_and_decide, task4_book_reservation
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_search_direct_flight": task2a_search_direct_flight,
            "task2b_search_onestop_flight": task2b_search_onestop_flight,
            "task2c_get_user_details": task2c_get_user_details,
            "task3_llm_process_filter_and_decide": task3_llm_process_filter_and_decide,
            "task4_book_reservation": task4_book_reservation
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task0_init", "target": "task1_llm_process"},
                {"source": "task1_llm_process", "target": "task2a_search_direct_flight"},
                {"source": "task1_llm_process","target": "task2b_search_onestop_flight"},
                {"source": "task1_llm_process","target": "task2c_get_user_details"},
                {"source": "task2a_search_direct_flight","target": "task3_llm_process_filter_and_decide"},
                {"source": "task2b_search_onestop_flight","target": "task3_llm_process_filter_and_decide"},
                {"source": "task2c_get_user_details","target": "task3_llm_process_filter_and_decide"},
                {"source": "task3_llm_process_filter_and_decide","target": "task4_book_reservation"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class tbench_airline_cancel_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.airline_cancel.task import task0_init, task1_llm_process1, task2_get_user_and_reservation_details, task3_cancel_reservation, task4_search_new_flights, task5_llm_process2, task6_book_new_reservation
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process1": task1_llm_process1,
            "task2_get_user_and_reservation_details": task2_get_user_and_reservation_details,
            "task3_cancel_reservation": task3_cancel_reservation,
            "task4_search_new_flights": task4_search_new_flights,
            "task5_llm_process2": task5_llm_process2,
            "task6_book_new_reservation": task6_book_new_reservation
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task0_init", "target": "task1_llm_process1"},
                {"source": "task1_llm_process1", "target": "task2_get_user_and_reservation_details"},
                {"source": "task2_get_user_and_reservation_details", "target": "task3_cancel_reservation"},
                {"source": "task3_cancel_reservation", "target": "task4_search_new_flights"},
                {"source": "task4_search_new_flights", "target": "task5_llm_process2"},
                {"source": "task5_llm_process2", "target": "task6_book_new_reservation"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class tbench_retail_cancel_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.retail_cancel.task import task0_init, task1_llm_process, task2_execute_cancel, task3_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2_execute_cancel": task2_execute_cancel,
            "task3_output_result": task3_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2_execute_cancel"},
                {"source": "task2_execute_cancel","target": "task3_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


@default_subscription
class tbench_retail_return_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.retail_return.task import task0_init, task1_llm_process, task2_find_user, task3_get_order_details, task4_execute_return, task5_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2_find_user": task2_find_user,
            "task3_get_order_details": task3_get_order_details,
            "task4_execute_return": task4_execute_return,
            "task5_output_result": task5_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                 {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2_find_user"},
                {"source": "task2_find_user","target": "task3_get_order_details"},
                {"source": "task3_get_order_details","target": "task4_execute_return"},
                {"source": "task4_execute_return","target": "task5_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class tbench_retail_modify_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.retail_modify.task import task0_init, task1_llm_process, task2a_find_user, task2b_get_order_details, task3_execute_modifications, task4_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_find_user": task2a_find_user,
            "task2b_get_order_details": task2b_get_order_details,
            "task3_execute_modifications": task3_execute_modifications,
            "task4_output_result": task4_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                 {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2a_find_user"},
                {"source": "task1_llm_process","target": "task2b_get_order_details"},
                {"source": "task2a_find_user","target": "task3_execute_modifications"},
                {"source": "task2b_get_order_details","target": "task3_execute_modifications"},
                {"source": "task3_execute_modifications","target": "task4_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class tbench_retail_cancel_modify_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.tbench.retail_cancel_modify.task import task0_init, task1_llm_process, task2a_find_user, task2b_get_order_details, task3_execute_operations,task4_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_find_user": task2a_find_user,
            "task2b_get_order_details": task2b_get_order_details,
            "task3_execute_operations": task3_execute_operations,
            "task4_output_result": task4_output_result

        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {
                    "source": "task0_init",
                    "target": "task1_llm_process"
                },
                {
                    "source": "task1_llm_process",
                    "target": "task2a_find_user"
                },
                {
                    "source": "task1_llm_process",
                    "target": "task2b_get_order_details"
                },
                {
                    "source": "task1_llm_process",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task2a_find_user",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task2b_get_order_details",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task3_execute_operations",
                    "target": "task4_output_result"
                }
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")
@default_subscription
class openagi_document_qa_Agent(BaseActorAgent):
    """
    OpenAGI 文档问答工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.openagi.document_qa.task import (
            task1_start_receive_task,
            task2_read_file,
            task3a_extract_text_content,
            task3b_llm_process_extract_structure_info,
            task3c_load_questions_batch,
            task4a_merge_document_analysis,
            task4b_prepare_qa_context,
            task5a_llm_process_batch_1,
            task5b_llm_process_batch_2,
            task5c_llm_process_batch_3,
            task7_merge_all_answers,
            task8_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file": task2_read_file,
            "task3a_extract_text_content": task3a_extract_text_content,
            "task3b_llm_process_extract_structure_info": task3b_llm_process_extract_structure_info,
            "task3c_load_questions_batch": task3c_load_questions_batch,
            "task4a_merge_document_analysis": task4a_merge_document_analysis,
            "task4b_prepare_qa_context": task4b_prepare_qa_context,
            "task5a_llm_process_batch_1": task5a_llm_process_batch_1,
            "task5b_llm_process_batch_2": task5b_llm_process_batch_2,
            "task5c_llm_process_batch_3": task5c_llm_process_batch_3,
            "task7_merge_all_answers": task7_merge_all_answers,
            "task8_output_final_answer": task8_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file"},
                {"source": "task2_read_file", "target": "task3a_extract_text_content"},
                {"source": "task2_read_file", "target": "task3b_llm_process_extract_structure_info"},
                {"source": "task2_read_file", "target": "task3c_load_questions_batch"},
                {"source": "task3a_extract_text_content", "target": "task4a_merge_document_analysis"},
                {"source": "task3b_llm_process_extract_structure_info", "target": "task4a_merge_document_analysis"},
                {"source": "task4a_merge_document_analysis", "target": "task4b_prepare_qa_context"},
                {"source": "task3c_load_questions_batch", "target": "task4b_prepare_qa_context"},
                {"source": "task4b_prepare_qa_context", "target": "task5a_llm_process_batch_1"},
                {"source": "task4b_prepare_qa_context", "target": "task5b_llm_process_batch_2"},
                {"source": "task4b_prepare_qa_context", "target": "task5c_llm_process_batch_3"},
                {"source": "task5a_llm_process_batch_1", "target": "task7_merge_all_answers"},
                {"source": "task5b_llm_process_batch_2", "target": "task7_merge_all_answers"},
                {"source": "task5c_llm_process_batch_3", "target": "task7_merge_all_answers"},
                {"source": "task7_merge_all_answers", "target": "task8_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class openagi_image_captioning_complex_Agent(BaseActorAgent):
    """
    OpenAGI 图像复杂描述工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.openagi.image_captioning_complex.task import (
            task1_start_receive_task,
            task2_read_and_enhance_images,
            task3a_extract_blip_captions,
            task3b_extract_ocr_text,
            task4_merge_image_features,
            task5a_vlm_process,
            task5b_vlm_process,
            task5c_vlm_process,
            task5d_vlm_process,
            task5_merge_results,
            task6_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_and_enhance_images": task2_read_and_enhance_images,
            "task3a_extract_blip_captions": task3a_extract_blip_captions,
            "task3b_extract_ocr_text": task3b_extract_ocr_text,
            "task4_merge_image_features": task4_merge_image_features,
            "task5a_vlm_process": task5a_vlm_process,
            "task5b_vlm_process": task5b_vlm_process,
            "task5c_vlm_process": task5c_vlm_process,
            "task5d_vlm_process": task5d_vlm_process,
            "task5_merge_results": task5_merge_results,
            "task6_output_final_answer": task6_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_and_enhance_images"},
                {"source": "task2_read_and_enhance_images", "target": "task3a_extract_blip_captions"},
                {"source": "task2_read_and_enhance_images", "target": "task3b_extract_ocr_text"},
                {"source": "task3a_extract_blip_captions", "target": "task4_merge_image_features"},
                {"source": "task3b_extract_ocr_text", "target": "task4_merge_image_features"},
                {"source": "task4_merge_image_features", "target": "task5a_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5b_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5c_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5d_vlm_process"},
                {"source": "task5a_vlm_process", "target": "task5_merge_results"},
                {"source": "task5b_vlm_process", "target": "task5_merge_results"},
                {"source": "task5c_vlm_process", "target": "task5_merge_results"},
                {"source": "task5d_vlm_process", "target": "task5_merge_results"},
                {"source": "task5_merge_results", "target": "task6_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription
class openagi_multimodal_vqa_complex_Agent(BaseActorAgent):
    """
    OpenAGI 多模态VQA复杂工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.openagi.multimodal_vqa_complex.task import (
            task1_start_receive_task,
            task2_read_file,
            task3_file_process,
            task4a_vision_llm_process,
            task4b_vision_llm_process,
            task4c_vision_llm_process,
            task4d_vision_llm_process,
            task4_merge_results,
            task5_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file": task2_read_file,
            "task3_file_process": task3_file_process,
            "task4a_vision_llm_process": task4a_vision_llm_process,
            "task4b_vision_llm_process": task4b_vision_llm_process,
            "task4c_vision_llm_process": task4c_vision_llm_process,
            "task4d_vision_llm_process": task4d_vision_llm_process,
            "task4_merge_results": task4_merge_results,
            "task5_output_final_answer": task5_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file"},
                {"source": "task2_read_file", "target": "task3_file_process"},
                {"source": "task3_file_process", "target": "task4a_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4b_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4c_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4d_vision_llm_process"},
                {"source": "task4a_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4b_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4c_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4d_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4_merge_results", "target": "task5_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

@default_subscription        
class openagi_text_processing_multilingual_Agent(BaseActorAgent):
    """
    OpenAGI 多语言文本处理工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        from baseline.workflows.openagi.text_processing_multilingual.task import (
            task1_start_receive_task,
            task2_read_file_and_split_questions,
            task3_language_detect,
            task4_translate_text,
            task5a_text_analysis_summarize,
            task5b_text_analysis_sentiment,
            task6_prepare_llm_batches,
            task7a_llm_process_batch_1,
            task7b_llm_process_batch_2,
            task7c_llm_process_batch_3,
            task8_merge_answers,
            task9_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file_and_split_questions": task2_read_file_and_split_questions,
            "task3_language_detect": task3_language_detect,
            "task4_translate_text": task4_translate_text,
            "task5a_text_analysis_summarize": task5a_text_analysis_summarize,
            "task5b_text_analysis_sentiment": task5b_text_analysis_sentiment,
            "task6_prepare_llm_batches": task6_prepare_llm_batches,
            "task7a_llm_process_batch_1": task7a_llm_process_batch_1,
            "task7b_llm_process_batch_2": task7b_llm_process_batch_2,
            "task7c_llm_process_batch_3": task7c_llm_process_batch_3,
            "task8_merge_answers": task8_merge_answers,
            "task9_output_final_answer": task9_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file_and_split_questions"},
                {"source": "task2_read_file_and_split_questions", "target": "task3_language_detect"},
                {"source": "task3_language_detect", "target": "task4_translate_text"},
                {"source": "task4_translate_text", "target": "task5a_text_analysis_summarize"},
                {"source": "task4_translate_text", "target": "task5b_text_analysis_sentiment"},
                {"source": "task2_read_file_and_split_questions", "target": "task6_prepare_llm_batches"},
                {"source": "task5a_text_analysis_summarize", "target": "task6_prepare_llm_batches"},
                {"source": "task5b_text_analysis_sentiment", "target": "task6_prepare_llm_batches"},
                {"source": "task6_prepare_llm_batches", "target": "task7a_llm_process_batch_1"},
                {"source": "task6_prepare_llm_batches", "target": "task7b_llm_process_batch_2"},
                {"source": "task6_prepare_llm_batches", "target": "task7c_llm_process_batch_3"},
                {"source": "task7a_llm_process_batch_1", "target": "task8_merge_answers"},
                {"source": "task7b_llm_process_batch_2", "target": "task8_merge_answers"},
                {"source": "task7c_llm_process_batch_3", "target": "task8_merge_answers"},
                {"source": "task8_merge_answers", "target": "task9_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")