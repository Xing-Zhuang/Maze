import base64
import os
import time
import json
import uuid
import threading
import asyncio
import argparse
import ast
from itertools import cycle
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from baseline.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader
import traceback
import csv
from worker_agent import *
from agentscope.message import Msg
from agentscope.rpc.retry_strategy import RetryFixedTimes
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# 将任务队列定义在全局，以便Flask和后台循环都能访问
dag_queue = asyncio.Queue()

# 将此映射同样置于全局，以便子进程可以访问
WORKFLOW_TYPE_TO_CLASS_MAP = {
    "gaia_file": GAIA_File_Process_Agent,
    "gaia_speech": GAIA_Speech_Agent,
    "gaia_reason": GAIA_Reason_Agent,
    "gaia_vision": GAIA_Vision_Agent,
    "tbench_airline_book": tbench_airline_book_Agent,
    "tbench_airline_cancel": tbench_airline_cancel_Agent,
    "tbench_retail_cancel": tbench_retail_cancel_Agent,
    "tbench_retail_return": tbench_retail_return_Agent,
    "tbench_retail_modify": tbench_retail_modify_Agent,
    "tbench_retail_cancel_modify": tbench_retail_cancel_modify_Agent,
    "openagi_document_qa": openagi_document_qa_Agent,
    "openagi_image_captioning_complex": openagi_image_captioning_complex_Agent,
    "openagi_multimodal_vqa_complex": openagi_multimodal_vqa_complex_Agent,
    "openagi_text_processing_multilingual": openagi_text_processing_multilingual_Agent
}

# ==================== 顶层函数，用于进程池执行 ====================
def run_reply_in_process(workflow_type: str, agent_name: str, host: str, port: int, msg_content: str) -> str:
    """
    这个函数将在一个完全独立的子进程中被执行，从而隔离所有资源。
    注意：为了跨进程传递，我们将完整的Msg对象简化为只传递其content字符串。
    """
    try:
        # 在子进程中重新构建Msg对象
        msg = Msg(name="", role="assistant", content=msg_content)

        AgentClass = WORKFLOW_TYPE_TO_CLASS_MAP.get(workflow_type)
        if not AgentClass:
            raise Exception(f"子进程中未找到类型 {workflow_type} 对应的 Agent 类")

        # 在子进程中重新创建RPC代理
        agent_proxy = AgentClass(id=agent_name, workflow_type=workflow_type).to_dist(
            host=host,
            port=port,
            retry_strategy=RetryFixedTimes(max_retries=600, delay=10)
        )
        # 执行阻塞的RPC调用
        response_msg = agent_proxy.reply(msg)
        # 只返回可序列化的结果内容
        return response_msg.content
    except Exception as e:
        # 在子进程中捕获异常，并将其作为字符串返回，以便主进程可以处理
        error_info = {
            "error": f"子进程执行失败: {e}",
            "traceback": traceback.format_exc()
        }
        return json.dumps(error_info)
# =================================================================================

def str_to_bool(val):
    if isinstance(val, bool): return val
    if val.lower() in ('true', 't', 'yes', '1'): return True
    elif val.lower() in ('false', 'f', 'no', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentOS Host: 启动主服务。")
    parser.add_argument("--agent_pools", type=str, required=True, help='定义工作流类型与Worker池的JSON字符串。')
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask API 监听的主机地址。")
    parser.add_argument("--port", type=int, default=5002, help="Flask API 的监听端口。")
    parser.add_argument("--proj_path", type=str, default="/root/workspace/d23oa7cp420c73acue30/AgentOS", help="项目根目录的绝对路径。")
    parser.add_argument("--model_folder", default="model_cache", help="模型缓存目录")
    parser.add_argument("--data_path", type=str, default="data/", help="数据目录相对路径。")
    parser.add_argument("--dag_path", type=str, default="src/benchmarks/workflows", help="DAG 定义目录相对路径。")
    parser.add_argument("--api_url", default="https://api.siliconflow.cn/v1/chat/completions", help="在线模型API URL")
    parser.add_argument("--api_key", default="Bearer sk-jbkxfkvrtluiezhqcvflmvenetulbluzpshppqqqtgxzswce", help="在线模型API Key")
    parser.add_argument("--temperature", type=float, default=0.6, help="模型采样温度")
    parser.add_argument("--max_token", type=int, default=1024, help="模型最大token")
    parser.add_argument("--top_p", type=float, default=0.9, help="模型top_p")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="模型重复惩罚")
    parser.add_argument("--use_online_model", type=str_to_bool, default=False, help="是否使用在线模型")
    parser.add_argument("--vlm_batch_size", type=int, default=8, help="VLM批处理大小")
    parser.add_argument("--text_batch_size", type=int, default=8, help="文本批处理大小")
    parser.add_argument("--task_exec_time_csv_path", default="src/baseline/agentscope/results/task_exec_time.csv")
    parser.add_argument("--task_exec_result_jsonl_path", default="src/baseline/agentscope/results/task_exec_result.jsonl")
    args = parser.parse_args()
    args.agent_pools = json.loads(args.agent_pools)
    args.model_folder = os.path.join(args.proj_path, args.model_folder)
    args.task_exec_time_csv_path = os.path.join(args.proj_path, args.task_exec_time_csv_path)
    args.task_exec_result_jsonl_path = os.path.join(args.proj_path, args.task_exec_result_jsonl_path)
    return args

args = parse_arguments()

class RoundRobinRouter:
    def __init__(self, agent_pools: Dict[str, List[str]]):
        self._next_agent_iterators = {
            workflow_type: cycle(agents)
            for workflow_type, agents in agent_pools.items()
        }
        print("✅ 轮询路由器已初始化，Worker池配置如下:")
        for w_type, agents in agent_pools.items():
            print(f"   - 类型 '{w_type}': {agents}")

    def get_next_worker_id(self, workflow_type: str) -> Optional[str]:
        iterator = self._next_agent_iterators.get(workflow_type)
        return next(iterator) if iterator else None

class TaskLoader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.query_loader_factory = {"gaia": GaiaLoader, "tbench": TBenchLoader, "openagi": OpenAGILoader}

    def load_workflow_message(self, dag_id: str, dag_type: str, dag_source: str, supplementary_files: List[str], sub_time: float) -> Msg:
        loader_class = self.query_loader_factory.get(dag_source)
        loader = loader_class(args=self.args, dag_id=dag_id, dag_type=dag_type, dag_source=dag_source, supplementary_files=supplementary_files, sub_time=sub_time)
        question, answer, arrival_time = loader.question, loader.answer, loader.arrival_time
        file_contents_base64 = {}
        file_paths_map = loader.get_supplementary_files()
        if file_paths_map:
            for filename, file_path in file_paths_map.items():
                with open(file_path, 'rb') as f:
                    content_bytes = f.read()
                file_contents_base64[filename] = base64.b64encode(content_bytes).decode('utf-8')
        arg_src = {"dag_id": dag_id, "question": question, "args": json.dumps(vars(self.args)), "supplementary_files": file_contents_base64}
        task_info = {"dag_id": dag_id, "question": question, "uuid": str(uuid.uuid4()), "arrival_time": arrival_time, "sub_time": sub_time, "arg_src": arg_src, "answer": answer, "type": f"{dag_source}_{dag_type}"}
        return Msg(name="", role="assistant", content=str(task_info))

class Dispatcher:
    """最终版调度器，实现了机器级（基于Host IP）的并发控制"""
    def __init__(self, router: "RoundRobinRouter", active_tasks: Dict, final_results: Dict, executor):
        self.router = router
        self.active_tasks = active_tasks
        self.final_results = final_results
        self.executor = executor
        self.machine_locks: Dict[str, asyncio.Lock] = {}
        self.lock_management_lock = asyncio.Lock()

    async def _execute_and_process_result(
        self, msg_content: str, uid: str, workflow_type: str, worker_id: str,
        agent_name: str, host: str, port: int, machine_lock: asyncio.Lock
    ):
        """这个方法负责所有耗时操作，并在执行前获取机器锁"""
        print(f"⏳ 任务 {uid} 正在排队，等待机器 [{host}] 变为空闲...")
        async with machine_lock:
            print(f"🟢 机器 [{host}] 已锁定，任务 {uid} 开始在子进程中执行。")
            self.active_tasks[uid].update({"status": "Running", "dispatched_to": worker_id})
            try:
                loop = asyncio.get_running_loop()
                response_content_str = await loop.run_in_executor(
                    self.executor, run_reply_in_process,
                    workflow_type, agent_name, host, port, msg_content
                )
                
                try:
                    potential_error = json.loads(response_content_str)
                    if isinstance(potential_error, dict) and "error" in potential_error:
                        raise Exception(potential_error.get('traceback', potential_error['error']))
                except (json.JSONDecodeError, TypeError):
                    pass
                
                response_content = ast.literal_eval(response_content_str)
                res_uuid = response_content.get("uuid")
                
                print(f"✅ [Dispatcher] 任务 {res_uuid} 执行成功。")
                self.active_tasks[res_uuid].update({"status": "Finished"})
                self.final_results[res_uuid] = response_content

                # ==================== 填充您缺失的文件写入逻辑 ====================
                try:
                    response_content["leave_time"] = time.time()
                    with open(args.task_exec_time_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [
                            response_content.get("dag_id"), response_content.get("uuid"), response_content.get("sub_time"),
                            response_content.get("arrival_time"), response_content.get("start_time"), response_content.get("end_time"),
                            response_content.get("end_time", 0) - response_content.get("start_time", 0),
                            response_content.get("leave_time"),
                            response_content.get("end_time", 0) - response_content.get("arrival_time", 0),
                            response_content.get("leave_time", 0) - response_content.get("sub_time", 0)
                        ]
                        writer.writerow(row)
                    print(f"DAG {response_content.get('dag_id')} execution times have been successfully logged.")
                except Exception as e:
                    print(f"Error logging DAG {response_content.get('dag_id')} to CSV: {e}")

                try:
                    with open(args.task_exec_result_jsonl_path, 'a', encoding='utf-8') as f:
                        json_str = json.dumps(response_content, ensure_ascii=False)
                        f.write(json_str + '\n')
                    print(f"💠 任务结果数据已成功追加到: {args.task_exec_result_jsonl_path}")
                except Exception as e:
                    print(f"写入文件失败: {e}")
                # =================================================================

            except Exception as e:
                print(f"❌ [Dispatcher] 任务 {uid} 的执行或结果处理协程失败: {e}\n{traceback.format_exc()}")
                if uid in self.active_tasks:
                    self.active_tasks[uid].update({"status": "Failed", "error": str(e)})
        
        print(f"🔵 机器 [{host}] 已释放，任务 {uid} 处理完毕。")

    async def dispatch_and_wait(self, msg: Msg):
        """这个函数现在只负责派发，并为后台任务准备好机器锁"""
        uid = "unknown"
        try:
            task_info = ast.literal_eval(msg.content)
            print("💨 Now we deal with task_info")
            uid = task_info.get("uuid")
            workflow_type = task_info.get("type")

            worker_id = self.router.get_next_worker_id(workflow_type)
            if not worker_id:
                raise ValueError(f"未找到处理 '{workflow_type}' 类型的Worker池。")

            agent_name, address = worker_id.split('@')
            host, port_str = address.split(':')
            port = int(port_str)
            
            async with self.lock_management_lock:
                if host not in self.machine_locks:
                    self.machine_locks[host] = asyncio.Lock()
            machine_lock_for_task = self.machine_locks[host]
            
            print(f"🚀 [Dispatcher] 任务 {uid} 已提交至后台执行队列 -> {worker_id}")

            asyncio.create_task(self._execute_and_process_result(
                msg.content, uid, workflow_type, worker_id, agent_name, host, port,
                machine_lock=machine_lock_for_task
            ))
        except Exception as e:
            print(f"❌ [Dispatcher] 任务 {uid} 在派发阶段失败: {e}\n{traceback.format_exc()}")
            if uid != 'unknown' and uid in self.active_tasks:
                 self.active_tasks[uid].update({"status": "Failed", "error": f"Dispatch error: {e}"})

def create_flask_app(loop: asyncio.AbstractEventLoop, task_loader: TaskLoader, active_tasks: Dict, final_results: Dict) -> Flask:
    app = Flask(__name__)
    @app.route("/submit_dag", methods=["POST"])
    def submit_dag() -> Any:
        payload = request.get_json()
        if not payload: return jsonify({"error": "Request body must be JSON."}), 400
        print(f"👹 接收到任务提交请求...")
        results = []
        sub_time = payload["sub_time"]
        for dag_id, dag_type, dag_source, supplementary_files in zip(payload["dag_ids"], payload["dag_types"], payload["dag_sources"], payload["dag_supplementary_files"]):
            try:
                message = task_loader.load_workflow_message(dag_id, dag_type, dag_source, supplementary_files, sub_time)
                uid = ast.literal_eval(message.content).get("uuid")
                active_tasks[uid] = {"status": "Queued", "dag_id": dag_id}
                asyncio.run_coroutine_threadsafe(dag_queue.put(message), loop)
                results.append({"dag_id": dag_id, "uuid": uid, "status": "scheduled"})
            except Exception as e:
                results.append({"dag_id": dag_id, "error": str(e)})
        print(f"📦 [Flask] 收到并处理了 {len(payload['dag_ids'])} 个DAG的提交请求。")
        return jsonify({"submitted": results}), 200

    @app.route("/dag_status/<dag_uuid>", methods=["GET"])
    def dag_status(dag_uuid: str) -> Any:
        return jsonify(active_tasks.get(dag_uuid, {"error": "DAG not found"}))

    @app.route("/get_final_result/<dag_uuid>", methods=["GET"])
    def get_final_result(dag_uuid: str) -> Any:
        return jsonify(final_results.get(dag_uuid, {"error": "Result not found"}))
    return app

async def background_dispatcher_loop(dispatcher: Dispatcher):
    """后台调度循环，从队列中取任务并并发地处理。"""
    print("🚀 后台调度循环已启动...")
    while True:
        msg = await dag_queue.get()
        await dispatcher.dispatch_and_wait(msg)
        await asyncio.sleep(0.05)  # 控制调度频率，避免过于频繁的调度

def main():
    """主函数，负责初始化并启动所有服务。"""
    csv_head = ['dag_id', 'uuid', 'sub_time', 'arrival_time', 'start_exec_time', 'finish_exec_time', 'exec_time', 'leave_time', 'completion_time', 'response_time']
    if not os.path.exists(args.task_exec_time_csv_path):
        with open(args.task_exec_time_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head)
        print(f"'{args.task_exec_time_csv_path}' created with headers.")
    if not os.path.exists(args.task_exec_result_jsonl_path):
        with open(args.task_exec_result_jsonl_path, 'w') as f:
            pass
        print(f"'{args.task_exec_result_jsonl_path}' created.")

    executor = ThreadPoolExecutor(max_workers= os.cpu_count()* 4)  # 使用线程池来处理子进程的调用
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    active_tasks: Dict[str, Any] = {}
    final_results: Dict[str, Any] = {}
    router = RoundRobinRouter(args.agent_pools)
    dispatcher = Dispatcher(router, active_tasks, final_results, executor=executor)
    task_loader = TaskLoader(args)

    app = create_flask_app(loop, task_loader, active_tasks, final_results)
    flask_thread = threading.Thread(target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    print(f"✅ Flask API 服务已在 http://{args.host}:{args.port} 上启动")

    try:
        print("主线程已启动，正在运行 asyncio 事件循环来调度任务...")
        loop.run_until_complete(
            background_dispatcher_loop(dispatcher)
        )
    except KeyboardInterrupt:
        print("\n收到退出信号...")
    finally:
        print("正在关闭进程池...")
        executor.shutdown(wait=True)
        print("正在关闭事件循环...")
        loop.close()
        print("服务已关闭。")

if __name__ == "__main__":
    main()
