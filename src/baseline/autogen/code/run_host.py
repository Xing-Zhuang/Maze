import os
import csv
import time
import json
import uuid
import base64
import asyncio
import argparse
import threading
from worker_agent import *
from itertools import cycle
from flask import Flask, request, jsonify
from autogen_core import try_get_known_serializers_for_type
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost, GrpcWorkerAgentRuntime
from autogen_core import AgentId, RoutedAgent, message_handler, default_subscription, MessageContext
# 导入我们之前定义好的 Agent 类和消息体
from typing import Dict, List, Any, Optional
from baseline.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader

def str_to_bool(val):
    """将字符串 'true' 或 'false' (不区分大小写) 转为布尔值"""
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', 't', 'yes', '1'):
        return True
    elif val.lower() in ('false', 'f', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# parse_arguments, RoundRobinRouter, TaskLoader 类保持不变，在此省略以保持版面整洁
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentOS Host: 启动主服务及MasterDispatcherAgent。")
    parser.add_argument("--agent_pools", type=str, required=True, help='定义工作流类型与Agent池的JSON字符串。例如: \'{"gaia_file": ["agent1"], "gaia_reason": ["agent2"], "gaia_speech": ["agent3"], "gaia_vision": ["agent4"]}\'')
    parser.add_argument("--host_addr", type=str, default="0.0.0.0:5003", help="AutoGen Host 监听的地址和端口。")
    parser.add_argument("--flask_port", type=int, default=5002, help="Flask API 的监听端口。")
    parser.add_argument("--proj_path", type=str, default= "/root/workspace/d23oa7cp420c73acue30/AgentOS", help="项目根目录的绝对路径。")
    parser.add_argument("--model_folder",  default="model_cache", 
                        help="Directory for caching downloaded models and intermediate results")
    parser.add_argument("--data_path", type=str, default="data/", help="数据目录相对路径。")
    parser.add_argument("--dag_path", type=str, default="src/benchmarks/workflows", help="DAG 定义目录相对路径。")
    parser.add_argument("--api_url", default="https://api.siliconflow.cn/v1/chat/completions",
                        help="API endpoint URL for online model inference requests")
    parser.add_argument("--api_key", default="Bearer sk-jbkxfkvrtluiezhqcvflmvenetulbluzpshppqqqtgxzswce",
                        help="Authentication API key for accessing the online model service (format: 'Bearer <token>')")
    parser.add_argument("--grpc_max_len", default= 100* 1024* 1024,
                        help="Sampling temperature for model output (0.0-1.0, lower = more deterministic)")
    parser.add_argument("--temperature", default=0.6,
                        help="Sampling temperature for model output (0.0-1.0, lower = more deterministic)")
    parser.add_argument("--max_token", default= 1024,
                        help="Maximum number of tokens allowed in the model's generated output")
    parser.add_argument("--top_p", default=0.9,
                        help="Maximum number of tokens allowed in the model's generated output")
    parser.add_argument("--repetition_penalty", default=1.1,
                        help="Maximum number of tokens allowed in the model's generated output")    
    parser.add_argument("--use_online_model", type= str_to_bool, default= False,
                        help= "use online model or no use")
    parser.add_argument("--vlm_batch_size", type= int, default= 8,
                        help="Maximum number of tokens allowed in the model's generated output")
    parser.add_argument("--text_batch_size", type= int, default= 8,
                        help="Maximum number of tokens allowed in the model's generated output")    
    parser.add_argument("--task_exec_time_csv_path", default= "src/baseline/autogen/results/task_exec_time.csv")
    parser.add_argument("--task_exec_result_jsonl_path", default= "src/baseline/autogen/results/task_exec_result.jsonl")
    args = parser.parse_args()
    args.agent_pools = json.loads(args.agent_pools)
    args.model_folder= os.path.join(args.proj_path, args.model_folder)
    args.task_exec_time_csv_path= os.path.join(args.proj_path, args.task_exec_time_csv_path)
    args.task_exec_result_jsonl_path= os.path.join(args.proj_path, args.task_exec_result_jsonl_path)       
    return args

args = parse_arguments()

class RoundRobinRouter:
    def __init__(self, agent_pools: Dict[str, List[str]]):
        self._agent_pools = agent_pools
        self._next_agent_iterators = {
            workflow_type: cycle(agents)
            for workflow_type, agents in agent_pools.items()
        }
    def get_next_agent(self, workflow_type: str) -> Optional[str]:
        iterator = self._next_agent_iterators.get(workflow_type)
        return next(iterator) if iterator else None

class TaskLoader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.query_loader_factory = {"gaia": GaiaLoader, "tbench": TBenchLoader, "openagi": OpenAGILoader}
    def load_workflow_message(self, dag_id: str, dag_type: str, dag_source: str, supplementary_files: List[str], sub_time: float) -> DAGMessage:
        loader_class = self.query_loader_factory.get(dag_source)
        if not loader_class: raise ValueError(f"未找到 '{dag_source}' 对应的数据加载器。")
        loader = loader_class(args= self.args, dag_id= dag_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time=sub_time)
        question = loader.question
        print(f"😊 load question: {question}")
        file_contents_base64 = {}
        file_paths_map = loader.get_supplementary_files()
        if file_paths_map:
            for filename, file_path in file_paths_map.items():
                with open(file_path, 'rb') as f:
                    file_contents_base64[filename] = base64.b64encode(f.read()).decode('utf-8')
        arg_src = {"supplementary_files": file_contents_base64, "args": json.dumps(vars(self.args))}
        return DAGMessage(dag_id=dag_id, uuid=str(uuid.uuid4()), type= f"{dag_source}_{dag_type}", question=question, arrival_time= loader.arrival_time, sub_time= loader.sub_time, arg_src= arg_src)

@default_subscription
class MasterDispatcherAgent(RoutedAgent):
    """一个常驻在Host端的总管Agent，负责任务的路由和状态追踪。"""
    # MODIFIED: __init__ 现在接收状态字典的引用
    def __init__(self, id: str, agent_pools: Dict[str, List[str]], local_runtime: GrpcWorkerAgentRuntime, active_tasks: Dict, final_results: Dict):
        super().__init__(id)
        self.router = RoundRobinRouter(agent_pools)
        self.local_runtime = local_runtime
        # 直接使用从 main 函数传入的字典引用
        self.active_tasks = active_tasks
        self.final_results = final_results
        print(f"✅ MasterDispatcherAgent '{id}' 已初始化。")

    @message_handler
    async def process_task_message(self, message: DAGMessage, ctx: MessageContext)-> AckMessage:
        """统一的消息处理器，用于分发新任务或接收已完成的结果。"""
        # Case 1: 这是一个已完成任务的返回结果
        # print(f"😀 Master received message: {message}")
        if message.result is not None:
            print(f"✅ [Master] 收到任务 {message.uuid} 的最终结果。")
            if message.uuid in self.active_tasks:
                self.active_tasks[message.uuid].update({"status": "Finished"})
            final_result= json.loads(message.result)
            final_result["leave_time"]= time.time()
            self.final_results[message.uuid] = {"final_result": final_result}
            # 写文件
            try:
                # 'a'模式表示追加写入
                with open(args.task_exec_time_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row= [
                        message.dag_id,
                        message.uuid,
                        message.sub_time,
                        message.arrival_time,
                        message.start_time,
                        message.end_time,
                        message.end_time- message.start_time,
                        final_result["leave_time"],
                        message.end_time- message.arrival_time,
                        final_result["leave_time"]- message.sub_time
                    ]
                    writer.writerow(row)
                print(f"DAG {final_result['dag_id']} execution times have been successfully logged.")
            except Exception as e:
                print(f"Error logging DAG {final_result['dag_id']} to CSV: {e}")
            # 结果写文件
            try:
                # 以追加模式写入数据
                with open(args.task_exec_result_jsonl_path, 'a', encoding= 'utf-8') as f:
                    json_str= json.dumps(final_result, ensure_ascii= False)
                    f.write(json_str + '\n')  # 添加换行符符合JSONL格式
                print(f"💠 任务结果数据已成功追加到: {args.task_exec_result_jsonl_path}")
            except Exception as e:
                print(f"写入文件失败: {e}")
            # 
            return AckMessage(comment=f"Final result for {message.uuid} received.")
        else:
            # Case 2: 这是一个需要分发的新任务
            if message.uuid in self.active_tasks:
                self.active_tasks[message.uuid].update({"status": "Dispatching"})
            
            target_agent_name = self.router.get_next_agent(message.type)
            print(f"🧡 message.type: {message.type} to target_agent_name: {target_agent_name}")
            if not target_agent_name:
                error_msg = f"未找到处理 '{message.type}' 类型的Agent池。"
                self.active_tasks[message.uuid].update({"status": "Failed", "error": error_msg})
                return AckMessage(status="error", comment=error_msg)
            else:
                print(f"🚀 [Master] 正在路由任务 {message.uuid} -> 发送给Agent '{target_agent_name}'")
                try:
                    # 使用正确的API，以“发射后不管”的方式发送
                    ack: AckMessage= await self.local_runtime.send_message(message, recipient= AgentId(target_agent_name, "default"))
                    if ack and ack.status == "ok":
                        print(f"✅ [Master] 收到来自 {target_agent_name} 的回执: {ack.comment}")
                        self.active_tasks[message.uuid].update({"status": "Running", "dispatched_to": target_agent_name})
                    else:
                        raise Exception(f"Worker returned a non-ok acknowledgement: {ack}")
                except Exception as e:
                    self.active_tasks[message.uuid].update({"status": "Failed", "error": str(e)})
                    print(f"任务发送异常, error: {str(e)}")
                return 

def create_flask_app(loop: asyncio.AbstractEventLoop, task_loader: TaskLoader, active_tasks: Dict, final_results: Dict, local_runtime: GrpcWorkerAgentRuntime) -> Flask:
    app = Flask(__name__)
    
    @app.route("/submit_dag", methods=["POST"])
    def submit_dag() -> Any:
        payload = request.get_json()
        dag_ids, dag_types, sources, files_list, sub_time = payload.get("dag_ids"), payload.get("dag_types"), payload.get("dag_sources"), payload.get("dag_supplementary_files"), payload.get("sub_time")
        if not all([dag_ids, dag_types, sources, files_list]):
            return jsonify({"error": "请求体缺少必要字段。"}), 400

        results = []
        for dag_id, dag_type, source, files in zip(dag_ids, dag_types, sources, files_list):
            try:
                message = task_loader.load_workflow_message(dag_id, dag_type, source, files, sub_time)
                active_tasks[message.uuid] = {"status": "Queued", "dag_id": dag_id, "progress": "0/1"}
                # print(f'😊 message has loaded successfully, message: {message}')
                # NEW: 直接、线程安全地调用 send_message，将任务发给 Master Agent
                coro= local_runtime.send_message(message, recipient=AgentId("master_dispatcher", "default"))
                asyncio.run_coroutine_threadsafe(coro, loop)

                results.append({"dag_id": dag_id, "uuid": message.uuid, "status": "scheduled"})
            except Exception as e:
                results.append({"dag_id": dag_id, "error": str(e)})
        return jsonify({"submitted": results}), 200

    @app.route("/dag_status/<dag_uuid>", methods=["GET"])
    def dag_status(dag_uuid: str) -> Any:
        return jsonify(active_tasks.get(dag_uuid, {"error": "DAG not found"}))

    @app.route("/get_final_result/<dag_uuid>", methods=["GET"])
    def get_final_result(dag_uuid: str) -> Any:
        return jsonify(final_results.get(dag_uuid, {"error": "Result not found"}))
    return app

# MODIFIED: 大幅简化的 main 函数
async def main():
    
    csv_head = ['dag_id', 'uuid', 'sub_time', 'arrival_time', 'start_exec_time', 'finish_exec_time', 'exec_time', 'leave_time', 'completion_time', 'response_time']
    if not os.path.exists(args.task_exec_time_csv_path):
        with open(args.task_exec_time_csv_path, 'w', newline='') as f:
            writer= csv.writer(f)
            writer.writerow(csv_head)
        print(f"'{args.task_exec_time_csv_path}' created with headers.")
    if not os.path.exists(args.task_exec_result_jsonl_path):
        with open(args.task_exec_result_jsonl_path, 'w', newline='') as f:
            pass
        print(f"'{args.task_exec_result_jsonl_path}' created with headers.")

    main_event_loop = asyncio.get_running_loop()
    extra_grpc_config = [
        ("grpc.max_send_message_length", args.grpc_max_len),
        ("grpc.max_receive_message_length", args.grpc_max_len),
    ]

    active_tasks: Dict[str, Any] = {}
    final_results: Dict[str, Any] = {}

    host = GrpcWorkerAgentRuntimeHost(address=args.host_addr, extra_grpc_config=extra_grpc_config)
    host.start()
    print(f"✅ AutoGen Host 服务已在 {args.host_addr} 上启动")

    local_runtime = GrpcWorkerAgentRuntime(host_address=args.host_addr, extra_grpc_config=extra_grpc_config)
    await local_runtime.start()
    print("   -> 正在注册消息序列化器...")
    local_runtime.add_message_serializer(try_get_known_serializers_for_type(DAGMessage))
    local_runtime.add_message_serializer(try_get_known_serializers_for_type(AckMessage))
    print("   -> 序列化器注册完毕。")
    await MasterDispatcherAgent.register(
        local_runtime,
        "master_dispatcher",
        lambda: MasterDispatcherAgent(
            id="master_dispatcher", 
            agent_pools=args.agent_pools,
            local_runtime=local_runtime,
            active_tasks=active_tasks,
            final_results=final_results
        )
    )
    print("✅ MasterDispatcherAgent 已成功注册。")
    
    task_loader = TaskLoader(args)
    # MODIFIED: 将 local_runtime 传递给 Flask App
    app = create_flask_app(main_event_loop, task_loader, active_tasks, final_results, local_runtime)
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=args.flask_port, debug=False), daemon=True)
    flask_thread.start()
    print(f"✅ Flask API 服务已在 http://0.0.0.0:{args.flask_port} 上启动")

    # REMOVED: 不再需要 master_agent_work_loop
    try:
        await asyncio.Event().wait()
    finally:
        print("服务正在关闭...")
        host.shutdown()

if __name__ == "__main__":
    asyncio.run(main())