import os
import sys
import json
import time
import threading
import traceback
import networkx as nx
import queue
from typing import Dict
import uuid
import logging

def dag_manager_daps(args, dag_que: queue.Queue, completion_que: queue.Queue, task_scheduler, shutdown_event, w1=2.0, w2=1.0):
    """
    (V3 - 内存通信版)
    - 从内存队列接收工作流。
    - 通过直接方法调用提交任务。
    - 从内存队列接收任务完成通知。
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🌟 FTC Scheduler (V3 - In-Memory Version) starting...")

    dags: Dict[str, nx.DiGraph] = {}
    dags_lock = threading.Lock()
    submit_queue = queue.PriorityQueue()
    task_enqueue_index = 0
    task_enqueue_lock = threading.Lock()

    dag_type_avg_times = {} 
    task_type_avg_times = {'io': 2.0, 'cpu': 5.0, 'gpu': 30.0}
    max_known_inferences = 10.0
    logger = logging.getLogger(__name__)

    def enqueue_task(dag, run_id, node_id, is_start_node=False):
        """将就绪任务放入优先级队列。"""
        scheduler_start_time = time.time()
        nonlocal task_enqueue_index
        node_info = dag.nodes[node_id]
        task_type = node_info.get('resources', {}).get('type', 'cpu')

        # 暂时使用默认平均时间进行预测
        pred_time = task_type_avg_times.get(task_type, 5.0)
        pred_exec_time = pred_time
        pred_cost_time = 0.0
        
        node_info['pred_time'] = pred_time
        
        remaining_inf = float(dag.graph.get('remaining_inferences', 0))
        score_urgency = 1.0 - (remaining_inf / max_known_inferences) if max_known_inferences > 0 else 0

        dag_type = dag.graph['dag_type']
        expected_dag_time = dag_type_avg_times.get(dag_type, 300.0)
        score_criticality = pred_time / expected_dag_time if expected_dag_time > 0 else 0

        performance_score = w1 * score_urgency + w2 * score_criticality
        
        with task_enqueue_lock:
            current_index = task_enqueue_index
            task_enqueue_index += 1
            
        sub_time = dag.graph.get("sub_time", time.time())
        priority_tuple = (sub_time, -performance_score, current_index)
        scheduler_end_time= time.time()
        item_to_queue = (priority_tuple, run_id, node_id, scheduler_end_time - scheduler_start_time, pred_exec_time, pred_cost_time)
        submit_queue.put(item_to_queue)
        logger.info(f"  -> Enqueued '{node_info.get('name', node_id)}'. (Arrival: {priority_tuple[0]:.2f}, PerfScore: {-priority_tuple[1]:.2f})")

    def dag_creator():
        """从提交队列中获取数据包，在内存中重建图，并预注册所有任务。"""
        nonlocal max_known_inferences
        while not shutdown_event.is_set():
            try:
                submission_package = dag_que.get()
                
                if not isinstance(submission_package, dict) or submission_package.get("submission_type") != "dynamic_agent":
                    print(f"⚠️ [dag_creator] 收到未知格式的数据，已跳过: {submission_package}")
                    continue

                run_id = submission_package['run_id']
                server_root_path = submission_package['server_root_path']
                workflow_payload = submission_package['workflow_payload']
                
                print(f"😊 [dag_creator] 开始处理新的动态Agent工作流, run_id: {run_id[:8]}")

                dag = nx.node_link_graph(workflow_payload['graph_definition'])
                
                tasks_definition = workflow_payload['tasks']
                for task_id, task_payload_dict in tasks_definition.items():
                    if task_id in dag.nodes:
                        dag.nodes[task_id].update(task_payload_dict)

                dag.graph['run_id'] = run_id
                dag.graph['dag_id'] = workflow_payload.get('dag_id', str(uuid.uuid4()))
                dag.graph['server_root_path'] = server_root_path
                dag.graph['name'] = workflow_payload['name']
                dag.graph['arrival_time'] = time.time() 
                dag.graph['sub_time'] = time.time()
                dag.graph['dag_type'] = workflow_payload['name']
                dag.graph['lock'] = threading.Lock()
                
                total_inferences = sum(1 for node_id in dag.nodes if dag.nodes[node_id].get('resources', {}).get('type') == 'gpu')
                dag.graph['total_inferences'] = total_inferences
                dag.graph['remaining_inferences'] = total_inferences
                max_known_inferences = max(max_known_inferences, float(total_inferences))
                
                with dags_lock:
                    dags[run_id] = dag
                
                print(f"  - ✅ [dag_creator] 内存图构建完成 for run_id: {run_id[:8]}.")

                # --- 新增逻辑开始 ---
                print(f"  - ⏳ [dag_creator] 预注册工作流中的所有 {len(dag.nodes)} 个任务...")
                for task_id, node_data_view in dag.nodes(data=True):
                    # 复制一份节点数据以避免修改原始图数据
                    node_data = dict(node_data_view)
                    
                    # 构建 TaskStatusManager.add_task 需要的 task_info 字典
                    task_info = {
                        "run_id": run_id, 
                        "dag_id": dag.graph.get("dag_id", ""),
                        "task_id": task_id,
                        "func_name": node_data.get("name"),
                        "serialized_func": node_data.get('serialized_func'),
                        "inputs": node_data.get("inputs", {}),
                        "output_parameters": node_data.get("meta", {}).get("output_parameters", {}),
                        # 从 'resources' 字典中安全地获取资源信息
                        "type": node_data.get("resources", {}).get("type", "cpu"),
                        "cpu_num": node_data.get("resources", {}).get("cpu_num", 1),
                        "mem": node_data.get("resources", {}).get("mem", 1024),
                        "gpu_mem": node_data.get("resources", {}).get("gpu_mem", 0),
                        "server_root_path": server_root_path,
                        "arrival_time": dag.graph['arrival_time']
                    }
                    task_scheduler.status_mgr.add_task(task_info)
                print(f"  - ✅ [dag_creator] 所有任务预注册完成。")
                # --- 新增逻辑结束 ---


                for node_id, in_degree in dag.in_degree():
                    if in_degree == 0:
                        dag.nodes[node_id]["in_degree"] = -1
                        enqueue_task(dag, run_id, node_id, is_start_node=True)
                    else:
                        dag.nodes[node_id]["in_degree"] = in_degree
            
            except Exception as e:
                print(f"❌ [Error] In dag_creator: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)

    def scheduler_and_submitter():
        """通过直接调用 TaskScheduler.submit() 方法提交任务。"""
        logger.info("🚀 FTC Submitter (In-Memory Version) is running.")
        
        while not shutdown_event.is_set():
            try:
                priority_tuple, run_id, task_id, scheduler_cost_time, pred_exec_time, pred_cost_time = submit_queue.get()
                with dags_lock:
                    dag = dags.get(run_id)
                if not dag:
                    logger.warning(f"  -> ❓ [Submitter] Warning: DAG for run_id {run_id} not found. Task '{task_id}' skipped.")
                    continue
                
                node_data = dict(dag.nodes[task_id])
                logger.info(f"🏆 Picked '{node_data.get('name', task_id)}' for submission.")
                dag.nodes[task_id]['scheduler_cost_time'] = scheduler_cost_time
                dag.nodes[task_id]['pred_exec_time'] = pred_exec_time
                dag.nodes[task_id]['pred_cost_time'] = pred_cost_time

                serialized_func = node_data.get('serialized_func')
                if not serialized_func:
                    raise ValueError(f"Serialized function not found in graph for task '{task_id}'")
                server_root_path = dag.graph.get("server_root_path")
                payload = {
                    "priority": priority_tuple,
                    "run_id": run_id, 
                    "dag_id": dag.graph.get("dag_id", ""),
                    "task_id": task_id,
                    "func_name": node_data.get("name"),
                    "serialized_func": serialized_func,
                    "inputs": node_data.get("inputs", {}),
                    "output_parameters": node_data.get("meta", {}).get("output_parameters", {}),
                    "type": node_data.get("resources", {}).get("type", "cpu"),
                    "cpu_num": node_data.get("resources", {}).get("cpu_num", 1),
                    "mem": node_data.get("resources", {}).get("mem", 1024),
                    "gpu_mem": node_data.get("resources", {}).get("gpu_mem", 0),
                    "server_root_path": server_root_path,
                }
                task_scheduler.submit(payload)
                logger.info(f"🎁 Submitted: '{node_data.get('name')}' from Run ID {run_id[:8]}")

            except Exception as e:
                logger.error(f"❌ [CRITICAL ERROR] In submitter loop: {e}\n{traceback.format_exc()}")
            time.sleep(0.05)

    def monitor():
        """从内存队列获取任务完成通知，并记录时间。"""
        print(f"💠 FTC Monitor (In-Memory Version) is listening...")
        
        while not shutdown_event.is_set():
            try:
                notification = completion_que.get()
                
                run_id = notification.get("run_id")
                task_id = notification.get("task_id")
                status = notification.get("status")

                # --- 新增：记录任务时间 ---
                if status in ["finished", "failed"]:
                    start_time = notification.get("worker_start_exec_time")
                    end_time = notification.get("worker_end_time")
                    if start_time and end_time:
                        timing_data = {
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                        task_scheduler.status_mgr.record_task_completion(run_id, task_id, timing_data)
                # --- 新增结束 ---

                with dags_lock:
                    dag = dags.get(run_id)
                if not dag: continue
                
                node_info = dag.nodes.get(task_id, {})
                func_name = node_info.get("name", "unknown_task")
                
                print(f"✅ Monitor: Received '{status}' for '{func_name}' (DAG: {dag.graph.get('name', 'N/A')[:8]})")
                
                with dag.graph['lock']:
                    if status == "finished":
                        if node_info.get('resources', {}).get('type') == 'gpu':
                            dag.graph['remaining_inferences'] = max(0, dag.graph.get('remaining_inferences', 1) - 1)
                        
                        node_info.update(notification)
                    
                    is_dag_complete = True
                    successors = list(dag.successors(task_id)) if dag.has_node(task_id) else []
                    for successor_id in successors:
                        is_dag_complete = False
                        successor_node = dag.nodes[successor_id]
                        if successor_node.get("in_degree", 0) > 0:
                            successor_node["in_degree"] -= 1
                            if successor_node["in_degree"] == 0:
                                successor_node["in_degree"] = -1
                                enqueue_task(dag, run_id, successor_id)
                    
                    # 检查是否所有任务都完成了（包括那些没有后继的任务）
                    all_nodes_finished = all(d.get('in_degree', -1) == -1 for n, d in dag.nodes.items())

                    if is_dag_complete and all_nodes_finished:
                        dag_total_time = time.time() - dag.graph['arrival_time']
                        # --- 新增：记录工作流总时间 ---
                        task_scheduler.status_mgr.record_run_completion(run_id, dag_total_time)
                        
                        print(f"🎉 DAG '{dag.graph.get('name', '')}' COMPLETE in {dag_total_time:.2f}s.")
                        with dags_lock:
                            if run_id in dags:
                                del dags[run_id]

            except Exception as e:
                print(f"❌ [Error] In monitor loop: {e}\n{traceback.format_exc()}")
            time.sleep(0.01)

    # --- 启动所有线程 ---
    threads = [
        threading.Thread(target=dag_creator, daemon=True, name="DAGCreator"),
        threading.Thread(target=scheduler_and_submitter, daemon=True, name="Submitter"),
        threading.Thread(target=monitor, daemon=True, name="Monitor")
    ]
    
    for t in threads:
        t.start()

    logger.info("🌟 DAPS Scheduler threads started.")
    shutdown_event.wait()
    logger.info("🌟 DAPS Scheduler has received shutdown signal and is exiting.")