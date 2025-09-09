import os
import math
import json
import random
import time
import requests
import argparse
import sys
import threading
import queue
from collections import Counter
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# =================================================================================
# Part 1: Query Pool Construction Logic (来自异质率代码1)
# =================================================================================

DATASET_HETEROGENEITY_DICT= {
    "gaia": {
        "file": {"cpu": 1, "gpu": 3, "io": 0},
        "reason": {"cpu": 0, "gpu": 3, "io": 1},
        "speech": {"cpu": 1, "gpu": 4, "io": 0},
        "vision": {"cpu": 1, "gpu": 1, "io": 1},
    },
    "openagi": {
        "document_qa": {"cpu": 4, "gpu": 4, "io": 4},
        "image_captioning_complex": {"cpu": 3, "gpu": 6, "io": 2},
        "multimodal_vqa_complex": {"cpu": 2, "gpu": 4, "io": 3},
        "text_processing_multilingual": {"cpu": 2, "gpu": 6, "io": 4},
    },
    "tbench": {
        "airline_book": {"cpu": 5, "gpu": 2, "io": 0},
        "airline_cancel": {"cpu": 3, "gpu": 2, "io": 2},
        "retail_cancel": {"cpu": 1, "gpu": 2, "io": 2},
        "retail_cancel_modify": {"cpu": 3, "gpu": 1, "io": 2},
        "retail_modify": {"cpu": 3, "gpu": 1, "io": 2},
        "retail_return": {"cpu": 3, "gpu": 1, "io": 2}
    }
}

def calculate_shannon_entropy(task_counts: dict) -> float:
    """根据给定的任务总数字典，计算香农熵。"""
    counts = [count for count in task_counts.values() if count > 0]
    total_tasks = sum(counts)
    if total_tasks == 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        probability = count / total_tasks
        entropy -= probability * math.log2(probability)
    return entropy

def analyze_query_prototypes(compositions: dict) -> dict:
    """计算每个独立Query原型的资源构成和异构性得分。"""
    prototype_profiles = {}
    for dataset, queries in compositions.items():
        for query_name, tasks in queries.items():
            full_name = f"{dataset}_{query_name}"
            heterogeneity = calculate_shannon_entropy(tasks)
            prototype_profiles[full_name] = {
                "composition": tasks,
                "heterogeneity": heterogeneity
            }
    return prototype_profiles

def construct_greedy_batch(prototypes: dict, target_heterogeneity: float, batch_size: int, seed: int):
    """
    使用贪心算法构造一个接近目标异构性的Query批次。
    返回Query类型的列表。
    """
    random.seed(seed)
    batch_types = []
    current_composition = Counter()
    
    # 随机选择一个起点，增加多样性
    start_query = random.choice(list(prototypes.keys()))
    batch_types.append(start_query)
    current_composition.update(prototypes[start_query]["composition"])

    while len(batch_types) < batch_size:
        best_next_query = None
        smallest_diff = float('inf')

        # 遍历所有原型，找到能让当前批次最接近目标异构性的那一个
        for query_name, profile in prototypes.items():
            temp_composition = current_composition + Counter(profile["composition"])
            new_heterogeneity = calculate_shannon_entropy(temp_composition)
            diff = abs(new_heterogeneity - target_heterogeneity)

            if diff < smallest_diff:
                smallest_diff = diff
                best_next_query = query_name

        batch_types.append(best_next_query)
        current_composition.update(prototypes[best_next_query]["composition"])

    final_heterogeneity = calculate_shannon_entropy(current_composition)
    return batch_types, final_heterogeneity


# =================================================================================
# Part 2: Benchmarking Infrastructure (来自连续请求代码)
# =================================================================================

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def rich_print(text: str):
    color_map = {"RED": "\033[91m", "GREEN": "\033[92m", "YELLOW": "\033[93m", "BLUE": "\033[94m", "MAGENTA": "\033[95m", "CYAN": "\033[96m", "WHITE": "\033[97m", "BOLD": "\033[1m", "END": "\033[0m"}
    for key, value in color_map.items(): text = text.replace(key, value)
    print(text + color_map["END"])

def load_dags_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    dag_definitions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        dag_definitions.append(json.loads(line))
                    except json.JSONDecodeError: pass
        return dag_definitions
    except FileNotFoundError:
        return []

class TaskLevelClient:

    """封装了您自研框架 (Task-level) 的API调用逻辑。"""
    def __init__(self, master_addr: str):
        self.base_url = f"http://{master_addr}"
        rich_print(f"✅  WHITE Initialized client for OUR SYSTEM at {self.base_url} WHITE ")
    def submit_dag(self, query: Dict, sub_time: float) -> Optional[Dict]:
        payload = { "dag_ids": [query["dag_id"]], "dag_sources": [query["dag_source"]], "dag_types": [query["dag_type"]], "dag_supplementary_files": [query["dag_supplementary_files"]], "sub_time": sub_time }
        url = f"{self.base_url}/dag/"
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            res = response.json()
            submitted = [dag for dag in res.get("data", []) if "error" not in dag]
            return submitted[0] if submitted else None
        except requests.exceptions.RequestException:
            return None
    def get_status_text(self, handle: Dict) -> str:
        run_id = handle['run_id']
        url = f"{self.base_url}/status/"
        try:
            response = requests.post(url, json={"run_id": run_id}, timeout=2)
            if response.status_code == 200:
                status_dict = response.json().get("data", {}).get("dag_status")
                if status_dict:
                    total = len(status_dict)
                    completed = sum(1 for task in status_dict.values() if task.get("status") != "unfinished")
                    if total > 0 and completed == total:
                        return "Finished"
                    return f"{completed}/{total} tasks completed"
            return "Polling status..."
        except requests.exceptions.RequestException:
            return "Connection error..."
    def release(self, handle: Dict):
        url = f"{self.base_url}/release/"
        try:
            payload = {"run_id": handle['run_id'], "dag_id": handle['dag_id'], "task2id": handle['task2id']}
            requests.post(url, json=payload, timeout=10)
        except requests.exceptions.RequestException: pass
    def get_and_print_results(self, handle: Dict):
        run_id = handle['run_id']
        dag_id = handle['dag_id']
        task2id = handle['task2id']
        url = f"{self.base_url}/get/"
        results = {}
        rich_print(f"\n BOLD MAGENTA 🔧 正在获取 DAG '{dag_id}' (Run: {run_id}) 的最终结果... BOLD MAGENTA ")
        for task_name, task_id in task2id.items():
            payload = {"run_id": run_id, "func_name": task_name, "task_id": task_id}
            try:
                response = requests.post(url, json=payload, timeout=20)
                if response.status_code == 200:
                    res = response.json()
                    if res.get("data") and "task_ret_data" in res["data"]:
                        try: results[task_name] = json.loads(res['data']['task_ret_data'])
                        except (json.JSONDecodeError, TypeError): results[task_name] = res['data']['task_ret_data']
                    else: results[task_name] = f"Error: {res.get('msg')}"
                else: results[task_name] = f"Error: Status {response.status_code}"
            except requests.exceptions.RequestException as e:
                results[task_name] = f"Error: {e}"
        pretty_result = json.dumps(results, indent=4, ensure_ascii=False)
        rich_print(f" BOLD GREEN --- 最终输出 for DAG '{dag_id}' --- BOLD GREEN \n{pretty_result}\n BOLD GREEN ----------------------------------------- BOLD GREEN ")

class AgentLevelClient:

    """封装了 AutoGen / AgentScope (Agent-level) 的API调用逻辑。"""
    def __init__(self, master_addr: str):
        self.base_url = f"http://{master_addr}"
        rich_print(f"✅  WHITE Initialized client for AUTOGEN/AGENTSCOPE at {self.base_url} WHITE ")
    def submit_dag(self, query: Dict, sub_time: float) -> Optional[str]:
        payload = { "dag_ids": [query["dag_id"]], "dag_sources": [query["dag_source"]], "dag_types": [query["dag_type"]], "dag_supplementary_files": [query["dag_supplementary_files"]], "sub_time": sub_time }
        url = f"{self.base_url}/submit_dag"
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            res = response.json()
            submitted = res.get("submitted", [])
            if submitted and "error" not in submitted[0]:
                return submitted[0].get("uuid")
            return None
        except requests.exceptions.RequestException:
            return None
    def get_status_text(self, handle: str) -> str:
        uuid = handle
        try:
            response = requests.get(f"{self.base_url}/dag_status/{uuid}", timeout=2)
            if response.status_code == 200:
                return response.json().get("status", "Unknown")
            return "Polling status..."
        except requests.exceptions.RequestException:
            return "Connection error..."
    def get_and_print_results(self, handle: str):
        uuid = handle
        rich_print(f"\n BOLD MAGENTA 🔧 正在获取 DAG (UUID: {uuid}) 的最终结果... BOLD MAGENTA ")
        try:
            response = requests.get(f"{self.base_url}/get_final_result/{uuid}", timeout=20)
            if response.status_code == 200:
                result_data = response.json()
                pretty_result = json.dumps(result_data, indent=4, ensure_ascii=False)
                rich_print(f" BOLD GREEN --- 最终输出 for DAG (UUID: {uuid}) --- BOLD GREEN \n{pretty_result}\n BOLD GREEN ----------------------------------------- BOLD GREEN ")
            else:
                rich_print(f"  -> ⚠️  YELLOW 获取结果失败 for {uuid}: Status {response.status_code} YELLOW ")
        except requests.exceptions.RequestException as e:
            rich_print(f"❌  RED 获取结果 {uuid} 时网络连接失败: {e} RED ")
    def release(self, handle: Any):
        pass


class BenchmarkRunner:

    def __init__(self, client: Any, query_pool: List[Dict], load_profile: List[tuple], num_workers: int, random_seed: int= 42):
        self.client = client
        self.query_pool = query_pool
        self.load_profile = load_profile
        self.num_workers = num_workers
        self.request_queue = queue.Queue()
        self.results_lock = threading.Lock()
        self.results = []
        self.stop_event = threading.Event()
        self.start_time = 0
        self.live_dags_status = {}
        self.live_dags_lock = threading.Lock()
        self.completed_count = 0
        self.submitted_count = 0
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.current_phase_info = "Initializing..."
        rich_print(f" BOLD BLUE 🔮 随机种子已设置: {random_seed} BOLD BLUE ")
    def _request_generator(self):
        local_random = random.Random(self.random_seed)
        local_np_random = np.random.RandomState(self.random_seed)
        rich_print(" BOLD CYAN 🚀 Request generator started... BOLD CYAN ")
        self.start_time = time.time()
        for i, (duration, lambda_rate) in enumerate(self.load_profile):
            if self.stop_event.is_set(): break
            phase_text = f"Phase {i+1}/{len(self.load_profile)}: Duration={duration}s, Rate={lambda_rate:.4f} req/s"
            with self.live_dags_lock:
                self.current_phase_info = phase_text
            rich_print(f"---  YELLOW {phase_text}  YELLOW  ---")
            stage_end_time = time.time() + duration
            while time.time() < stage_end_time:
                if self.stop_event.is_set(): break
                if lambda_rate > 0:
                    wait_time = local_np_random.exponential(1.0 / lambda_rate)
                    time.sleep(wait_time)
                else:
                    time.sleep(1)
                if time.time() >= stage_end_time: break
                self.request_queue.put(local_random.choice(self.query_pool))
        with self.live_dags_lock:
            self.current_phase_info = "Generation finished."
        # Don't set stop_event here, let workers finish the queue
        rich_print(f" BOLD CYAN 🏁 Request generator finished. All requests queued. BOLD CYAN ")
    def _monitoring_loop(self):
        while not self.stop_event.is_set():
            clear_console()
            rich_print("="*25 + "  BOLD WHITE 实时任务监控仪表盘 WHITE  BOLD  " + "="*25)
            with self.live_dags_lock:
                elapsed_runtime = time.time() - self.start_time if self.start_time > 0 else 0.0
                running_count = len(self.live_dags_status)
                phase_info = self.current_phase_info
                rich_print(f" WHITE --------------------------------- [ General ] ---------------------------------- WHITE ")
                rich_print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 运行时长: {elapsed_runtime:.1f}s | WHITE 提交: {self.submitted_count} | 运行中: {running_count} | 已完成: {self.completed_count} WHITE ")
                rich_print(f" YELLOW Current Phase: {phase_info} YELLOW ")
                rich_print(f" WHITE --------------------------------- [ Live DAGs ] ---------------------------------- WHITE ")
                for i, (unique_id, data) in enumerate(self.live_dags_status.items()):
                    if i >= 10:
                        rich_print(f"  ... and {running_count - 10} more running ...")
                        break
                    dag_id, status_text, submit_time = data['dag_id'], data['status_text'], data['submit_time']
                    elapsed = time.time() - submit_time
                    rich_print(f"  CYAN DAG: {dag_id:<30} WHITE |  CYAN UUID: {unique_id:<37} WHITE |  CYAN 状态: {status_text:<25} ({elapsed:.1f}s) WHITE ")
            time.sleep(1)
    def _worker(self, worker_id: int):
        while True:
            handle = None
            unique_id = None
            try:
                # When the generator is done and queue is empty, worker can exit.
                if self.stop_event.is_set() and self.request_queue.empty():
                    break
                query = self.request_queue.get(timeout=1)
                submission_time = time.time()
                handle = self.client.submit_dag(query, submission_time)
                if not handle:
                    self._log_result(query["dag_id"], "SUBMIT_FAILED", submission_time, time.time())
                    continue
                unique_id = handle['run_id'] if isinstance(handle, dict) else handle
                with self.live_dags_lock:
                    self.submitted_count += 1
                    self.live_dags_status[unique_id] = {
                        'dag_id': query['dag_id'],
                        'status_text': 'Submitted...',
                        'submit_time': submission_time
                    }
                while True: # Polling loop
                    status_text = self.client.get_status_text(handle)
                    with self.live_dags_lock:
                        if unique_id in self.live_dags_status:
                            self.live_dags_status[unique_id]['status_text'] = status_text
                    if status_text.lower() == "finished":
                        self._log_result(query["dag_id"], "SUCCESS", submission_time, time.time())
                        self.client.get_and_print_results(handle) # Optionally print results during run
                        break
                    time.sleep(10) # Poll every 10 seconds
            except queue.Empty:
                continue
            except Exception as e:
                rich_print(f" RED [Worker {worker_id}] Error: {e} RED ")
            finally:
                if unique_id:
                    with self.live_dags_lock:
                        if unique_id in self.live_dags_status:
                            del self.live_dags_status[unique_id]
                            self.completed_count += 1
                if handle:
                    self.client.release(handle)
    def _log_result(self, dag_id: str, status: str, start_time: float, end_time: float):
        latency = end_time - start_time
        with self.results_lock:
            self.results.append({"dag_id": dag_id, "status": status, "latency": latency, "end_time": end_time, "start_time": start_time})
    def run(self):
        generator_thread = threading.Thread(target=self._request_generator)
        worker_threads = [threading.Thread(target=self._worker, args=(i,)) for i in range(self.num_workers)]
        monitor_thread = threading.Thread(target=self._monitoring_loop)
        
        generator_thread.start()
        for t in worker_threads: t.start()
        monitor_thread.start()
        
        generator_thread.join()
        self.stop_event.set() # Signal workers to stop after emptying the queue
        
        for t in worker_threads: t.join()
        
        # Stop monitor after workers are done
        monitor_thread.join()
        return self.results
    def save_results(self, filename="results.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        rich_print(f" BOLD GREEN ✅ 实验结果已保存到: {filename} BOLD GREEN ")


# =================================================================================
# Part 3: Main Execution Block (整合与驱动)
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    一个统一的、可配置的压测脚本，用于测试异构工作负载下的系统吞吐量。
    """)
    # --- 系统与路径参数 ---
    parser.add_argument("--target_system", type=str, required=True, choices=['ours', 'autogen', 'agentscope', 'vllm'], help="要测试的目标系统。")
    parser.add_argument("--proj_path", default="/root/workspace/d23oa7cp420c73acue30/AgentOS", help="AgentOS项目根目录。")
    # --- 负载生成参数 ---
    parser.add_argument("--heterogeneity", type=float, default= 0.8, help="目标Query池的异质率 (香农熵)。")
    parser.add_argument("--pool_size", type=int, default=100, help="要构造的Query池的大小。")
    # --- 压测参数 ---
    parser.add_argument("--avg_dag_time", type=int, default=45, help="预估的单个DAG平均完成时间，用于计算请求率。")
    parser.add_argument("--workers", type=int, default=40, help="并发压测的工作线程数。")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子，用于复现。")
    
    args = parser.parse_args()

    load_profile = [
        (1800, (1/ args.avg_dag_time))
        ]    

    # 1. 构造Query池
    rich_print(" BOLD BLUE 📚 Part 1: 正在构造指定异质率的Query池... BOLD BLUE ")
    query_prototypes = analyze_query_prototypes(DATASET_HETEROGENEITY_DICT)
    
    generated_query_types, achieved_heterogeneity = construct_greedy_batch(
        prototypes=query_prototypes,
        target_heterogeneity=args.heterogeneity,
        batch_size=args.pool_size,
        seed=args.seed
    )
    rich_print(f" BOLD GREEN ✅ Query池构造完成! BOLD GREEN ")
    rich_print(f"   - 目标异质率: {args.heterogeneity:.4f}")
    rich_print(f"   - 实际达成: {achieved_heterogeneity:.4f}")
    rich_print(f"   - 池中Query类型构成: {dict(Counter(generated_query_types))}")

    # 2. 加载完整的Query定义
    rich_print("\n BOLD BLUE 📂 Part 2: 正在从文件加载完整的Query定义... BOLD BLUE ")
    all_dags_by_type = {}
    dataset_paths = ["data/gaia/gaia_query.jsonl", "data/tbench/tbench_query.jsonl", "data/openagi/openagi_query.jsonl"]
    for rel_path in dataset_paths:
        full_path = os.path.join(args.proj_path, rel_path)
        if os.path.exists(full_path):
            for dag in load_dags_from_jsonl(full_path):
                dag_type = dag.get("dag_type")
                if dag_type:
                    # 将 'gaia_file' 这种格式与我们的原型名称对齐
                    full_type_name = f"{dag.get('dag_source', '').split('/')[-1]}_{dag_type}"
                    if full_type_name not in all_dags_by_type:
                        all_dags_by_type[full_type_name] = []
                    all_dags_by_type[full_type_name].append(dag)

    # 3. 根据构造的池子类型，创建最终的请求池
    final_query_pool = []
    for query_type in generated_query_types:
        if query_type in all_dags_by_type and all_dags_by_type[query_type]:
            # 从该类型的所有可用DAG中随机选择一个
            final_query_pool.append(random.choice(all_dags_by_type[query_type]))

    if not final_query_pool:
        rich_print(" RED ❌ 错误: 最终的Query池为空！请检查文件路径和Query类型名称是否匹配。 RED ")
        sys.exit(1)
    rich_print(f" BOLD GREEN ✅ 最终请求池准备完毕，总大小: {len(final_query_pool)}。 BOLD GREEN ")

    # 4. 设置客户端和负载配置
    server_addresses = { 'ours': '127.0.0.1:6382', 'autogen': 'localhost:5002', 'agentscope': 'localhost:5002', 'vllm': 'localhost:5002'}
    target_addr = server_addresses.get(args.target_system)
    if not target_addr:
        rich_print(f" RED 错误: 未知的目标系统 '{args.target_system}' RED ")
        sys.exit(1)
    client = TaskLevelClient(master_addr=target_addr) if args.target_system == 'ours' else AgentLevelClient(master_addr=target_addr)

    # 5. 运行压测
    rich_print("\n BOLD BLUE 🚀 Part 3: 启动压测... BOLD BLUE ")
    benchmark = BenchmarkRunner(
        client=client, 
        query_pool=final_query_pool, 
        load_profile=load_profile,
        num_workers=args.workers, 
        random_seed=args.seed
    )
    results = benchmark.run()