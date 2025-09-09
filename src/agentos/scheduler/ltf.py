import os
import sys
import json
import time
import heapq
import redis
import queue
import requests
import threading
import traceback
import cloudpickle
import networkx as nx
import importlib.util
from agentos.utils.exec_time_pred import DAGTaskPredictor
from agentos.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader

query_loader_factory = {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader,
}

def dag_manager_ltf(args, dag_que, dag_status_dict):
    dags = {}
    # --- 架构核心：为每种资源创建一个独立的优先队列 ---
    submit_queues = {
        "cpu": queue.PriorityQueue(),
        "gpu": queue.PriorityQueue(),
        "io": queue.PriorityQueue(),
    }
    predictor = DAGTaskPredictor(args.redis_ip, args.redis_port, args.time_pred_model_path,
                                 args.min_sample4train, args.min_sample4incremental)
    redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=False)
    task_enqueue_index = 0
    task_enqueue_lock = threading.Lock()

    def enqueue_task(run_id, func_name, is_initial=False):
        nonlocal task_enqueue_index
        dag = dags[run_id]
        node_info = dag.nodes[func_name]
        task_type = node_info.get('type', 'cpu')
        
        if is_initial:
            pred_time = -1.0
        else:
            pred_task_ids = [dag.nodes[p].get("task_id") for p in dag.predecessors(func_name)]
            pred_time = predictor.predict(
                succ_func_name=func_name,
                succ_task_type=task_type,
                pred_task_ids=pred_task_ids
            )

        if pred_time <= 1.0:
            defaults = {'io': 2, 'cpu': 5, 'gpu': 30}
            pred_time = defaults.get(task_type, 5)

        arrival_time = dag.graph["arrival_time"]
        with task_enqueue_lock:
            current_index = task_enqueue_index
            task_enqueue_index += 1
        priority_tuple = (arrival_time, -pred_time, current_index)
        item_to_queue = (priority_tuple, run_id, func_name)
        # 放入对应的资源队列
        queue_for_task = submit_queues.get(task_type, submit_queues["cpu"])
        queue_for_task.put(item_to_queue)
        print(f"  -> Enqueued '{func_name}' to [{task_type.upper()}] queue. (DAG Arrival: {arrival_time:.2f}, Pred Time: {pred_time:.4f}s)")

    def dag_creator():
        while True:
            try:
                run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time = dag_que.get()
                query_loader = query_loader_factory.get(dag_source)
                if not query_loader: continue
                loader = query_loader(args= args, dag_id= dag_id, run_id= run_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time= sub_time)
                dag = loader.get_dag(task2id)
                dag.graph['lock'] = threading.Lock()
                dags[run_id] = dag
                print(f"😊 ParallelLTF: DAG '{dag_id}' created.")

                for node, in_degree in dag.in_degree():
                    node_info = dag.nodes[node]
                    node_info["in_degree"] = in_degree
                    if in_degree == 0:
                        enqueue_task(run_id, node_info["func_name"], is_initial=True)
            except Exception as e:
                print(f"❌ [Error] In dag_creator: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)

    def submitter_worker(task_type: str, task_queue: queue.PriorityQueue):
        """一个为特定资源类型工作的提交者线程"""
        print(f"🚀 Submitter for [{task_type.upper()}] queue is running.")
        RETRY_LIMIT= 3
        RETRY_DELAY_SECONDS= 1
        print(f"🚀 Resilient Submitter for [{task_type.upper()}] queue is running.")
        while True:
            try:
                priority_tuple, run_id, func_name = task_queue.get()
                succeeded= False
                for attempt in range(RETRY_LIMIT):
                    try:
                        dag= dags.get(run_id)
                        if not dag:
                            print(f"  -> ❓ [Submitter-{task_type}] Warning: DAG for run_id {run_id} not found. Task '{func_name}' skipped permanently.")
                            succeeded = True 
                            break

                        payload = dict(dag.nodes[func_name])
                        task_id= payload["task_id"]

                        payload["priority"]= priority_tuple
                        payload.update({
                            "run_id": run_id, "dag_id": dag.graph["dag_id"],
                            "question": dag.graph.get("question"), "answer": dag.graph.get("answer"),
                            "supplementary_file_paths": dag.graph.get("supplementary_file_paths"),
                            "dag_func_file": dag.graph.get("dag_func_file"), "arrival_time": dag.graph.get("arrival_time")
                        })

                        if not payload.get("dag_func_file"):
                            succeeded = True
                            break

                        redis_func_key = f"func:{task_id}"
                        spec = importlib.util.spec_from_file_location("dag_module", payload["dag_func_file"])
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        func = getattr(module, func_name)
                        redis_client.set(redis_func_key, cloudpickle.dumps(func))

                        if redis_client.get(redis_func_key) is None:
                            raise ConnectionError(f"Verification GET failed for task {task_id}.")
                        requests.post(f"http://{args.master_addr}/inform", json=payload)
                        print(f"🎁 [{task_type.upper()}] Submitted: '{func_name}' from Run ID {run_id}")
                        succeeded = True
                        break
                    except Exception as e:
                        print(f"  -> ⚠️ [Submitter-{task_type}] Attempt {attempt + 1}/{RETRY_LIMIT} failed for task '{func_name}': {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                        time.sleep(RETRY_DELAY_SECONDS)
                if not succeeded:
                    print(f"  -> ❌ [Submitter-{task_type}] FAILED to submit task '{func_name}' after {RETRY_LIMIT} attempts. Re-queuing.")
                    task_queue.put((priority_tuple, run_id, func_name))
                    time.sleep(RETRY_DELAY_SECONDS) 
            except Exception as e:
                print(f"❌ [CRITICAL ERROR] In {task_type.upper()} submitter main loop: {e}\n{traceback.format_exc()}")
                time.sleep(RETRY_DELAY_SECONDS)
            time.sleep(0.05)

    def monitor():
        redis_monitor_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=True)
        completion_queue_name = "task_completion_queue"
        print(f"💠 ParallelLTF Monitor is listening...")

        while True:
            try:
                _, message = redis_monitor_client.brpop(completion_queue_name)
                notification = json.loads(message)
                run_id, func_name, status = notification["run_id"], notification["func_name"], notification["status"]

                dag = dags.get(run_id)
                if not dag: continue
                
                # Added the missing logic to update the shared status dictionary
                print(f"✅ Monitor: Updating status for '{func_name}'(DAG_ID: {dag.graph['dag_id']}) to '{status}'.")
                task_id = dag.nodes[func_name].get("task_id")
                task_result_raw = redis_monitor_client.get(f"result:{task_id}") if task_id else None
                if task_result_raw:
                    record_json = json.loads(task_result_raw)
                    predictor.collect_data_for_task(
                        task_id=task_id,
                        func_name=func_name,
                        record_json=record_json
                    )
                task_result = json.loads(task_result_raw) if task_result_raw else {}
                current_status = dag_status_dict[run_id]
                current_status[func_name] = {
                    "start_exec_time": task_result.get("start_time", 0.0),
                    "finish_exec_time": task_result.get("end_time", 0.0),
                    "arrival_time": dag.graph.get("arrival_time"),
                    "sub_time": dag.graph.get("sub_time", 0.0),
                    "leave_time": time.time(),
                    "status": status,
                }
                dag_status_dict[run_id] = current_status

                if status == "finished":
                    successors_to_check = list(dag.successors(func_name))
                    if not successors_to_check:
                         print(f"  -> '{func_name}' is a terminal node. No successors to enqueue.")
                    for successor_name in successors_to_check:
                        with dag.graph['lock']:
                            successor_node = dag.nodes[successor_name]
                            print(f"  -> Checking successor '{successor_name}':")
                            print(f"     - In-degree BEFORE decrement: {successor_node.get('in_degree')}")
                            successor_node["in_degree"] -= 1
                            print(f"     - In-degree AFTER decrement: {successor_node.get('in_degree')}")
                            is_ready = (successor_node["in_degree"] == 0)
                        if is_ready:
                            print(f"  -> SUCCESSOR READY! Enqueuing '{successor_name}'...")
                            enqueue_task(run_id, successor_name, is_initial=False)
            except Exception as e:
                print(f"❌ [Error] In monitor: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)


    # --- 启动所有线程 ---
    threads = [
        threading.Thread(target=dag_creator),
        threading.Thread(target=monitor)
    ]
    for task_type, task_queue in submit_queues.items():
        threads.append(threading.Thread(target=submitter_worker, args=(task_type, task_queue)))
        
    for t in threads:
        t.start()
    
    print("✅ Parallel Longest-Task-First (ParallelLTF) Scheduler Started.")

    for t in threads:
        t.join()