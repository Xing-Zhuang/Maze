import os
import sys
import json
import time
import requests
import threading
import traceback
import cloudpickle
import redis
import networkx as nx
import importlib.util
import queue  # 引入线程安全的队列模块

# 假设这些工具类在你的项目路径下
from agentos.utils.exec_time_pred import DAGTaskPredictor
from agentos.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader

query_loader_factory = {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader,
}

def dag_manager_ftc(args, dag_que, dag_status_dict, w1=2.0, w2=1.0):
    """
    FTC (Fairness-Throughput-Criticality) 调度器模块 V4.2 (最终修复版)

    - 修正了 payload['priority'] 的传递方式，直接传递优先级元组。
    - 补全了 monitor 中缺失的 dag_status_dict 状态写回逻辑。
    - 采用生产者-消费者模式，使用线程安全的 PriorityQueue，根除并发问题。
    """
    print(f"🌟 FTC Scheduler (V4.2 - Final Version) starting with weights: w1(Throughput)={w1}, w2(Criticality)={w2}")

    # --- 共享状态 ---
    dags = {}
    dags_lock = threading.Lock()
    status_lock = threading.Lock() # 为 dag_status_dict 增加锁
    submit_queue = queue.PriorityQueue()
    task_enqueue_index = 0
    task_enqueue_lock = threading.Lock()

    # --- 动态学习状态 ---
    dag_type_avg_times = {} 
    task_type_avg_times = {'io': 2.0, 'cpu': 5.0, 'gpu': 30.0}
    max_known_inferences = 10.0

    # --- 辅助模块 ---
    predictor = DAGTaskPredictor(args.redis_ip, args.redis_port, args.time_pred_model_path,
                                 args.min_sample4train, args.min_sample4incremental)
    redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=False)
    
    def enqueue_task(dag, run_id, func_name, is_start_node=False):
        scheduler_start_time = time.time()
        pred_exec_time= None
        pred_start_time, pred_cost_time= None, None
        nonlocal task_enqueue_index
        node_info = dag.nodes[func_name]
        task_type = node_info.get('type', 'cpu')

        # w.o. time pred method
        pred_time= None
        if not is_start_node: # stop time pred
            pred_task_ids = [dag.nodes[p].get("task_id") for p in dag.predecessors(func_name)] # stop time pred
            pred_start_time= time.time() # stop time pred
            pred_time = predictor.predict(succ_func_name=func_name, succ_task_type=task_type, pred_task_ids=pred_task_ids) # stop time pred
            pred_cost_time= time.time()- pred_start_time # stop time pred
            pred_exec_time= pred_time # stop time pred
        if not pred_time:
            pred_time = task_type_avg_times.get(task_type, 5.0)
                
        node_info['pred_time'] = pred_time
        
        remaining_inf = float(dag.graph.get('remaining_inferences', 0))
        score_urgency = 1.0 - (remaining_inf / max_known_inferences)

        dag_type = dag.graph['dag_type']
        expected_dag_time = dag_type_avg_times.get(dag_type, 300.0)
        score_criticality = pred_time / expected_dag_time

        performance_score = w1 * score_urgency + w2 * score_criticality
        
        with task_enqueue_lock:
            current_index = task_enqueue_index
            task_enqueue_index += 1
            
        # arrival_time = dag.graph.get("arrival_time", time.time())
        sub_time = dag.graph.get("sub_time", time.time())
        priority_tuple = (sub_time, -performance_score, current_index)
        scheduler_end_time= time.time()
        item_to_queue = (priority_tuple, run_id, func_name, scheduler_end_time - scheduler_start_time, pred_exec_time, pred_cost_time)
        submit_queue.put(item_to_queue)
        print(f"  -> Enqueued '{func_name}'. (Arrival: {priority_tuple[0]:.2f}, PerfScore: {-priority_tuple[1]:.2f})")

    def dag_creator():
        nonlocal max_known_inferences
        while True:
            try:
                run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time = dag_que.get()
                
                query_loader = query_loader_factory.get(dag_source)
                if not query_loader: continue

                loader = query_loader(args=args, dag_id=dag_id, run_id=run_id, dag_type=dag_type, dag_source=dag_source, supplementary_files=supplementary_files, sub_time=sub_time)
                dag = loader.get_dag(task2id)
                if not dag: continue
                
                dag.graph['lock'] = threading.Lock()
                total_inferences = sum(1 for node in dag.nodes if dag.nodes[node].get('type') == 'gpu')
                dag.graph['total_inferences'] = total_inferences
                dag.graph['remaining_inferences'] = total_inferences
                max_known_inferences = max(max_known_inferences, float(total_inferences))
                
                with dags_lock:
                    dags[run_id] = dag
                
                print(f"😊 FTC: DAG '{dag_id}' (run_id: {run_id[:8]}) created. Total Inferences: {total_inferences}")

                for node, in_degree in dag.in_degree():
                    if in_degree == 0:
                        dag.nodes[node]["in_degree"] = -1
                        enqueue_task(dag, run_id, node, is_start_node=True)
                    else:
                        dag.nodes[node]["in_degree"] = in_degree
            except Exception as e:
                print(f"❌ [Error] In dag_creator: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)

    def scheduler_and_submitter():
        """消费者线程，只管从队列取最优任务并提交。"""
        print("🚀 FTC Submitter (Elegant Priority Version) is running.")
        RETRY_LIMIT, RETRY_DELAY_SECONDS = 3, 1
        
        while True:
            try:
                priority_tuple, run_id, func_name, scheduler_cost_time, pred_exec_time, pred_cost_time = submit_queue.get()
                
                print(f"🏆 Picked '{func_name}' (Arrival: {priority_tuple[0]:.2f}, PerfScore: {-priority_tuple[1]:.2f}).")
                
                succeeded = False
                for attempt in range(RETRY_LIMIT):
                    try:
                        dag = None
                        with dags_lock:
                            dag = dags.get(run_id)
                        if not dag:
                            print(f"  -> ❓ [Submitter] Warning: DAG for run_id {run_id} not found (likely completed). Task '{func_name}' skipped.")
                            succeeded = True 
                            break
                        node_data = dict(dag.nodes[func_name])
                        dag.nodes[func_name]['scheduler_cost_time'] = scheduler_cost_time
                        dag.nodes[func_name]['pred_exec_time'] = pred_exec_time
                        dag.nodes[func_name]['pred_cost_time'] = pred_cost_time
                        # --- 修正一：构建完整的Payload，并直接传递优先级元组 ---
                        payload = {
                            "priority": priority_tuple, # <--- 直接传递元组
                            **node_data,
                            "run_id": run_id, 
                            "dag_id": dag.graph["dag_id"],
                            "dag_func_file": dag.graph.get("dag_func_file"), 
                            "arrival_time": dag.graph.get("arrival_time"),
                            "question": dag.graph.get("question"), 
                            "answer": dag.graph.get("answer"),
                            "supplementary_file_paths": dag.graph.get("supplementary_file_paths")
                        }

                        if not payload.get("dag_func_file"):
                            succeeded = True
                            break
                        
                        task_id = payload["task_id"]
                        redis_func_key = f"func:{task_id}"
                        spec = importlib.util.spec_from_file_location("dag_module", payload["dag_func_file"])
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        func = getattr(module, func_name)
                        redis_client.set(redis_func_key, cloudpickle.dumps(func))

                        if redis_client.get(redis_func_key) is None:
                            raise ConnectionError(f"Redis SET/GET verification failed for task {task_id}.")
                        
                        response = requests.post(f"http://{args.master_addr}/inform", json=payload)
                        response.raise_for_status()

                        print(f"🎁 Submitted: '{func_name}' from Run ID {run_id[:8]}")
                        succeeded = True
                        break
                    except Exception as e:
                        print(f"  -> ⚠️ [Submitter] Attempt {attempt + 1}/{RETRY_LIMIT} failed for '{func_name}': {e}. Retrying...")
                        time.sleep(RETRY_DELAY_SECONDS)
                
                if not succeeded:
                    print(f"  -> ❌ [Submitter] FAILED to submit task '{func_name}'. Re-enqueuing.")
                    with dags_lock:
                        dag = dags.get(run_id)
                    if dag:
                        enqueue_task(dag, run_id, func_name)
            
            except Exception as e:
                print(f"❌ [CRITICAL ERROR] In submitter loop: {e}\n{traceback.format_exc()}")
                time.sleep(1)
            time.sleep(0.05)

    def monitor():
        """生产者线程：当任务完成时，产生新的就绪任务。"""
        print(f"💠 FTC Monitor is listening...")
        redis_monitor_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=True)
        completion_queue_name = "task_completion_queue"
        
        while True:
            try:
                _, message = redis_monitor_client.brpop(completion_queue_name)
                notification = json.loads(message)
                run_id, func_name, status = notification["run_id"], notification["func_name"], notification["status"]

                dag = None
                with dags_lock:
                    dag = dags.get(run_id)
                if not dag: continue
                
                print(f"✅ Monitor: Received '{status}' for '{func_name}' (DAG: {dag.graph['dag_id'][:8]})")
                
                with dag.graph['lock']:
                    node_info = dag.nodes[func_name]
                    if status == "finished":
                        if node_info.get('type') == 'gpu':
                            dag.graph['remaining_inferences'] = max(0, dag.graph.get('remaining_inferences', 1) - 1)
                        
                        task_id = node_info.get("task_id")
                        task_result_raw = redis_monitor_client.get(f"result:{task_id}")
                        if task_result_raw:
                            task_result = json.loads(task_result_raw)
                            start_exec_time = task_result.get("start_time", 0.0)
                            finish_exec_time = task_result.get("end_time", 0.0)
                            actual_exec_time = finish_exec_time - start_exec_time # 修正 typo
                            predictor.collect_data_for_task( # stop time pred
                                task_id=task_id, # stop time pred
                                func_name=func_name, # stop time pred
                                record_json=task_result # stop time pred
                            ) # stop time pred

                            # 更新任务平均耗时
                            if actual_exec_time > 0:
                                task_type = node_info.get('type', 'cpu')
                                current_avg = task_type_avg_times.get(task_type, actual_exec_time)
                                task_type_avg_times[task_type] = 0.8 * current_avg + 0.2 * actual_exec_time

                            # --- 补全状态写回逻辑 ---
                            with status_lock:
                                current_status = dag_status_dict.get(run_id, {})
                                status_entry = current_status.setdefault(func_name, {})
                                status_entry.update({
                                    "start_exec_time": start_exec_time,
                                    "finish_exec_time": finish_exec_time,
                                    "arrival_time": dag.graph.get("arrival_time"),
                                    "sub_time": dag.graph.get("sub_time"),
                                    "leave_time": time.time(),
                                    "status": status,
                                    "scheduler_cost_time": node_info.get('scheduler_cost_time', 0.0),
                                    "pred_exec_time": node_info.get('pred_exec_time', 0.0),
                                    "pred_cost_time": node_info.get('pred_cost_time', 0.0),
                                })
                                dag_status_dict[run_id] = current_status

                    is_dag_complete = True
                    for successor_name in dag.successors(func_name):
                        is_dag_complete = False
                        successor_node = dag.nodes[successor_name]
                        if successor_node.get("in_degree", 0) > 0:
                            successor_node["in_degree"] -= 1
                            if successor_node["in_degree"] == 0:
                                successor_node["in_degree"] = -1
                                enqueue_task(dag, run_id, successor_name)
                    
                    if is_dag_complete:
                        all_nodes_processed = all(d.get('in_degree', -1) == -1 for n, d in dag.nodes.items())
                        if all_nodes_processed:
                            dag_total_time = time.time() - dag.graph['arrival_time']
                            dag_type = dag.graph['dag_type']
                            current_avg_dag_time = dag_type_avg_times.get(dag_type, dag_total_time)
                            dag_type_avg_times[dag_type] = 0.8 * current_avg_dag_time + 0.2 * dag_total_time
                            print(f"🎉 DAG '{dag.graph['dag_id'][:8]}' COMPLETE in {dag_total_time:.2f}s.")
                            with dags_lock:
                                if run_id in dags:
                                    del dags[run_id]
            except Exception as e:
                print(f"❌ [Error] In monitor loop: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)

    # --- 启动所有线程 ---
    threads = [
        threading.Thread(target=dag_creator, daemon=True),
        threading.Thread(target=scheduler_and_submitter, daemon=True),
        threading.Thread(target=monitor, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    print("✅ All FTC Scheduler threads started.")

    for t in threads:
        t.join()