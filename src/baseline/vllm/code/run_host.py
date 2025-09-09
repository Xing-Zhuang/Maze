# run_host.py (重写优化版)
import os
import csv
import json
import time
import uuid
import base64
import argparse
import asyncio
import threading
from flask import Flask, request, jsonify

# 依赖于您项目中的现有文件
from run_worker import DAGWorker
from baseline.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader

# --- 全局共享资源 ---
dag_queue: asyncio.Queue = None
dag_statuses: dict = {}
dag_results: dict = {}
loop: asyncio.AbstractEventLoop = None
app = Flask(__name__)
args: argparse.Namespace = None
task_loader: 'TaskLoader' = None

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

# --- 参数解析 (与原文件一致) ---
def parse_host_arguments():
    """解析Host运行所需的所有参数。"""
    parser = argparse.ArgumentParser(description="VLLM-based Host Server")
    # 路径和端口参数
    parser.add_argument("--proj_path", type=str, default="/root/workspace/d23oa7cp420c73acue30/AgentOS", help="项目根目录的绝对路径。")
    parser.add_argument("--data_path", type=str, default="data/", help="数据目录相对路径。")
    parser.add_argument("--dag_path", type=str, default="src/baseline/workflows", help="DAG 定义目录相对路径。")
    parser.add_argument("--flask_port", type=int, default=5002, help="Flask服务器监听的端口。")
    parser.add_argument("--model_folder",  default="model_cache", 
                        help="Directory for caching downloaded models and intermediate results")
    parser.add_argument("--api_url", default="https://api.siliconflow.cn/v1/chat/completions",
                        help="API endpoint URL for online model inference requests")
    parser.add_argument("--api_key", default="Bearer sk-jbkxfkvrtluiezhqcvflmvenetulbluzpshppqqqtgxzswce",
                        help="Authentication API key for accessing the online model service (format: 'Bearer <token>')")
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
    # VLLM配置文件路径参数
    parser.add_argument("--vllm_endpoints_json", default="src/baseline/vllm/code/vllm_endpoints.json", help="VLLM服务配置文件的路径（相对项目路径）。")
    
    # 日志文件路径参数
    parser.add_argument("--task_exec_time_csv_path", default="src/baseline/vllm/results/task_exec_time.csv", help="任务执行时间记录的CSV文件路径。")
    parser.add_argument("--task_exec_result_jsonl_path", default="src/baseline/vllm/results/task_exec_result.jsonl", help="任务最终结果记录的JSONL文件路径。")
    
    parsed_args, _ = parser.parse_known_args()

    # 将相对路径拼接成绝对路径
    parsed_args.model_folder= os.path.join(parsed_args.proj_path, parsed_args.model_folder)
    parsed_args.data_path = os.path.join(parsed_args.proj_path, parsed_args.data_path)
    parsed_args.dag_path = os.path.join(parsed_args.proj_path, parsed_args.dag_path)
    parsed_args.vllm_endpoints_json = os.path.join(parsed_args.proj_path, parsed_args.vllm_endpoints_json)
    parsed_args.task_exec_time_csv_path = os.path.join(parsed_args.proj_path, parsed_args.task_exec_time_csv_path)
    parsed_args.task_exec_result_jsonl_path = os.path.join(parsed_args.proj_path, parsed_args.task_exec_result_jsonl_path)
    
    return parsed_args

# --- TaskLoader (与原文件一致) ---
class TaskLoader:
    """负责将任务元数据，转换成包含所有实际数据的完整任务包。"""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.query_loader_factory = {"gaia": GaiaLoader, "tbench": TBenchLoader, "openagi": OpenAGILoader}

    def load_task_package(self, task_data: dict, sub_time: float) -> dict:
        """加载单个任务的所有数据。"""
        dag_source = task_data.get("dag_source")
        loader_class = self.query_loader_factory.get(dag_source)
        if not loader_class:
            raise ValueError(f"未找到 '{dag_source}' 对应的数据加载器。")
        
        loader = loader_class(args=self.args, dag_id=task_data.get("dag_id"), dag_type=task_data.get("dag_type"), dag_source=dag_source, supplementary_files=task_data.get("supplementary_files"), sub_time=sub_time)
        
        question = loader.question
        file_paths_map = loader.get_supplementary_files()
        
        file_contents_base64 = {}
        if file_paths_map:
            for filename, file_path in file_paths_map.items():
                try:
                    with open(file_path, 'rb') as f:
                        file_contents_base64[filename] = base64.b64encode(f.read()).decode('utf-8')
                except FileNotFoundError:
                     print(f"⚠️ 文件未找到，跳过: {file_path}")

        return {"dag_id": task_data.get("dag_id"), "dag_type": task_data.get("dag_type"), "dag_source": dag_source, "question": question, "supplementary_files": file_contents_base64, "sub_time": sub_time}

# --- 日志记录核心函数 (与原文件一致) ---
log_lock = threading.Lock()
def ensure_log_files_exist():
    os.makedirs(os.path.dirname(args.task_exec_time_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.task_exec_result_jsonl_path), exist_ok=True)
    if not os.path.exists(args.task_exec_time_csv_path):
        with open(args.task_exec_time_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['dag_id', 'uuid', 'sub_time', 'arrival_time', 'start_exec_time', 'finish_exec_time', 'exec_time', 'leave_time', 'completion_time', 'response_time'])
def log_to_csv(log_data: dict):
    with log_lock:
        with open(args.task_exec_time_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([log_data.get(k) for k in ['dag_id', 'uuid', 'sub_time', 'arrival_time', 'start_exec_time', 'finish_exec_time', 'exec_time', 'leave_time', 'completion_time', 'response_time']])
def log_to_jsonl(log_data: dict):
    with log_lock:
        with open(args.task_exec_result_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

# --- API 端点定义 (参考 Autogen 版重写) ---
@app.route("/submit_dag", methods=["POST"])
def submit_dag_endpoint():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400

    dag_ids = payload.get("dag_ids", [])
    dag_sources = payload.get("dag_sources", [])
    dag_types = payload.get("dag_types", [])
    dag_supplementary_files = payload.get("dag_supplementary_files", [])
    sub_time = payload.get("sub_time")

    if not (len(dag_ids) == len(dag_sources) == len(dag_types) == len(dag_supplementary_files)):
        return jsonify({"error": "Input lists (dag_ids, dag_sources, etc.) must have the same length."}), 400

    submitted_results = []
    # 使用 zip 迭代，处理每个任务
    for dag_id, source, dag_type, files in zip(dag_ids, dag_sources, dag_types, dag_supplementary_files):
        try:
            # 1. 同步准备任务数据
            task_data_for_loader = {
                "dag_id": dag_id,
                "dag_source": source,
                "dag_type": dag_type,
                "supplementary_files": files
            }
            full_task_package = task_loader.load_task_package(task_data_for_loader, sub_time)

            dag_uuid = str(uuid.uuid4())
            final_package_to_queue = {
                "uuid": dag_uuid,
                "arrival_time": time.time(),
                "task_body": full_task_package
            }

            # 2. 将准备好的任务放入后台队列
            # 这是非阻塞操作，会立即返回
            asyncio.run_coroutine_threadsafe(dag_queue.put(final_package_to_queue), loop)
            
            # 3. 更新状态并记录成功信息
            dag_statuses[dag_uuid] = {"status": "queued", "submitted_at": time.time(), "dag_id": dag_id}
            submitted_results.append({"dag_id": dag_id, "uuid": dag_uuid})

        except Exception as e:
            import traceback
            print(f"❌ Error submitting DAG '{dag_id}': {e}\n{traceback.format_exc()}")
            submitted_results.append({"dag_id": dag_id, "error": str(e)})
    
    # 4. 返回处理回执，键名为 "submitted" 以匹配 dispatch_task.py
    return jsonify({"message": "DAG submission request processed.", "submitted": submitted_results}), 202

@app.route("/dag_status/<dag_uuid>", methods=["GET"])
def get_status_endpoint(dag_uuid: str):
    return jsonify(dag_statuses.get(dag_uuid, {"error": "DAG not found."}))

@app.route("/get_final_result/<dag_uuid>", methods=["GET"])
def get_final_result_endpoint(dag_uuid: str):
    status_info = dag_statuses.get(dag_uuid, {})
    if status_info.get("status") not in ["finished", "error"]:
        return jsonify({"status": status_info.get("status", "unknown"), "message": "DAG is still processing."})
    return jsonify(dag_results.get(dag_uuid, {"error": "Result not found for this DAG."}))


# --- 应用启动和后台循环设置 (与原文件一致) ---
def run_asyncio_loop(loop_to_run):
    asyncio.set_event_loop(loop_to_run)
    loop_to_run.run_forever()

async def initialize_and_run_worker(queue, statuses, results, csv_logger, jsonl_logger, worker_args, endpoints_data):
    # 此处传递 vllm_endpoints_data 给 Worker
    worker = DAGWorker(queue, statuses, results, csv_logger, jsonl_logger, worker_args, endpoints_data)
    print("✅ DAGWorker 初始化成功。")
    asyncio.create_task(worker.consume_tasks())
    print("✅ DAGWorker 任务消费者已在后台启动。")

if __name__ == "__main__":
    args = parse_host_arguments()
    ensure_log_files_exist()
    
    vllm_endpoints_data = {}
    try:
        with open(args.vllm_endpoints_json, 'r') as f:
            vllm_endpoints_data = json.load(f)
        print(f"✅ VLLM 配置文件 '{args.vllm_endpoints_json}' 加载成功。")
    except Exception as e:
        print(f"❌ 严重错误: 无法加载VLLM配置文件 '{args.vllm_endpoints_json}'. 错误: {e}")
        exit(1)
    
    task_loader = TaskLoader(args)
    
    dag_queue = asyncio.Queue()
    dag_statuses = {}
    dag_results = {}
    
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    loop_thread.start()
    print("🌀 后台Asyncio事件循环已启动。")

    # 将解析好的 vllm_endpoints_data 传递下去
    asyncio.run_coroutine_threadsafe(
        initialize_and_run_worker(
            dag_queue, 
            dag_statuses, 
            dag_results, 
            log_to_csv, 
            log_to_jsonl, 
            args, 
            vllm_endpoints_data
        ),
        loop
    )

    print(f"🚀 Flask Host服务已启动，监听 http://0.0.0.0:{args.flask_port}")
    # 在生产环境中建议使用 Gunicorn 或 uWSGI
    app.run(host='0.0.0.0', port=args.flask_port)