import queue
import threading
import time
import requests
import networkx as nx
import cloudpickle
import importlib.util
import json
import redis
import traceback
import random

# CPOP需要同时用到ranku和rankd
from agentos.utils.scheduler_algorithm_utils import compute_ranku, compute_rankd, TASK_TYPE_DEFAULT_EXEC_TIMES
from agentos.utils.query_loader import GaiaLoader, TBenchLoader, OpenAGILoader

query_loader_factory = {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader
}

def dag_manager_cpop(args, dag_que, dag_status_dict):
    """
    一个严格遵循CPOP算法思想的动态调度管理器。
    - 核心思想: 
        1. 使用 ranku + rankd 作为任务优先级。
        2. 识别关键路径(Critical Path)，并为其指派一个最优处理器。
        3. 关键路径上的任务优先调度到指定处理器，其他任务则在所有处理器中寻找最优EFT。
    - 遵循原始Paper:
        - 本实现确保任务只有在“真实时间”达到其“预估开始时间(EST)”之后才会被派发。
        - 适用于需要严格遵循算法定义的科研和实验场景。
    """
    
    # --- 状态维护 ---
    dags_data = {}
    
    try:
        res = requests.post(f"http://{args.master_addr}/resource")
        res.raise_for_status()
        available_nodes = list(res.json().keys())
        if not available_nodes:
            raise ValueError("No available compute nodes found from master.")
        print(f"✅ CPOP Manager: Successfully discovered {len(available_nodes)} nodes: {available_nodes}")
    except Exception as e:
        print(f"❌ [FATAL] Failed to get available nodes from master: {e}. Exiting.")
        return

    proc_schedules = {node: [] for node in available_nodes}
    task_to_node_map = {} 
    
    ready_list = []
    ready_list_lock = threading.Lock()
    
    monitor_que = queue.Queue()
    
    exec_time_db = {}
    redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=False)
    print("💠 CPOP manager (Strict Paper Version) initialized.")


    def find_insert_time_slot(schedule_list, ready_time, exec_time):
        """在节点的调度列表上为任务找到最早的可用时间插槽。"""
        if not schedule_list:
            return ready_time

        if ready_time + exec_time <= schedule_list[0][0]:
            return ready_time

        for i in range(len(schedule_list) - 1):
            prev_end = schedule_list[i][1]
            next_start = schedule_list[i + 1][0]
            start_candidate = max(ready_time, prev_end)
            if start_candidate + exec_time <= next_start:
                return start_candidate

        return max(ready_time, schedule_list[-1][1])

    def find_critical_path(dag: nx.DiGraph, priority: dict) -> list:
        """根据优先级寻找图中的关键路径。"""
        entry_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
        if not entry_nodes: return []
        entry_node = entry_nodes[0]
        
        path = [entry_node]
        current = entry_node
        while dag.out_degree(current) > 0:
            successors = list(dag.successors(current))
            if not successors: break
            next_node = max(successors, key=lambda n: priority.get(n, 0))
            path.append(next_node)
            current = next_node
        return path

    def select_cp_processor(cp_tasks: list) -> str:
        """为关键路径选择处理器，同质情况下。"""
        return random.choice(available_nodes)

    def dag_creator():
        """
        线程函数：接收新DAG，执行完整的CPOP调度算法生成计划，然后将初始任务放入就绪列表。
        """
        while True:
            run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time = dag_que.get()
            print(f"💠 CPOP Creator: Received new DAG '{dag_id}'")
            try:
                query_loader = query_loader_factory.get(dag_source)
                loader = query_loader(args= args, dag_id= dag_id, run_id= run_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time= sub_time)
                dag_graph = loader.get_dag(task2id)
                dags_data[run_id] = dag_graph
                
                # --- 核心CPOP调度逻辑 ---
                print(f"💠 CPOP Scheduler: Planning for DAG '{dag_id}'.")
                
                # 1. 计算 ranku, rankd, 和最终优先级 (ranku + rankd)
                ranku_values = compute_ranku(dag_graph, exec_time_db, run_id)
                rankd_values = compute_rankd(dag_graph, exec_time_db, run_id)
                priorities = {
                    task: ranku_values.get(task, 0) + rankd_values.get(task, 0)
                    for task in dag_graph.nodes()
                }

                # 2. 识别关键路径并为其选择处理器
                critical_path = find_critical_path(dag_graph, priorities)
                cp_processor = select_cp_processor(critical_path)
                print(f"  -> Identified Critical Path: {' -> '.join(critical_path)}")
                print(f"  -> Critical Path assigned to Processor: '{cp_processor}'")

                # 3. 记录调度“0时刻”
                dag_graph.graph['schedule_start_time'] = time.time()
                
                # 4. 按优先级顺序调度所有任务
                # 使用一个模拟的就绪队列来进行拓扑排序
                scheduling_ready_queue = [n for n in dag_graph.nodes if dag_graph.in_degree(n) == 0]
                scheduled_tasks = set()

                while len(scheduled_tasks) < len(dag_graph.nodes()):
                    if not scheduling_ready_queue:
                        print(f"❌ [SCHEDULER_FATAL] Scheduling ready queue is empty but tasks remain. Check for cycles.")
                        break

                    # 按优先级排序，选择优先级最高的任务
                    scheduling_ready_queue.sort(key=lambda task: priorities.get(task, 0), reverse=True)
                    task_name = scheduling_ready_queue.pop(0)

                    # 计算任务的就绪时间
                    ready_time = 0.0
                    for pred in dag_graph.predecessors(task_name):
                        ready_time = max(ready_time, dag_graph.nodes[pred].get('eft', 0.0))
                    
                    exec_time = exec_time_db.get((run_id, task_name))
                    if exec_time is None:
                        task_type = dag_graph.nodes[task_name].get('type')
                        exec_time = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])
                    # 为任务选择最佳节点和时间
                    # 如果任务在关键路径上，只在CP处理器上调度
                    candidate_nodes = [cp_processor] if task_name in critical_path else available_nodes
                    
                    best_node, best_eft, best_est = None, float('inf'), 0.0
                    for node_id in candidate_nodes:
                        est = find_insert_time_slot(proc_schedules[node_id], ready_time, exec_time)
                        eft = est + exec_time
                        if eft < best_eft:
                            best_eft, best_est, best_node = eft, est, node_id

                    # 存储调度决策
                    print(f"  -> Task '{task_name}' scheduled on node '{best_node}' at EST: {best_est:.2f}, finishes at EFT: {best_eft:.2f}")
                    dag_graph.nodes[task_name]['est'] = best_est
                    dag_graph.nodes[task_name]['eft'] = best_eft
                    dag_graph.nodes[task_name]['node'] = best_node
                    
                    proc_schedules[best_node].append((best_est, best_eft, (run_id, task_name)))
                    proc_schedules[best_node].sort(key=lambda x: x[0])
                    task_to_node_map[(run_id, task_name)] = best_node
                    scheduled_tasks.add(task_name)

                    # 更新调度就绪队列
                    for successor in dag_graph.successors(task_name):
                        if all(p in scheduled_tasks for p in dag_graph.predecessors(successor)):
                            if successor not in scheduling_ready_queue:
                                scheduling_ready_queue.append(successor)
                
                # 5. 将真正的初始任务放入执行就绪列表
                with ready_list_lock:
                    for node in dag_graph.nodes():
                        if dag_graph.in_degree(node) == 0:
                            ready_list.append((run_id, node))
                print(f"  -> Initial ready tasks for DAG '{dag_id}' have been queued for submission.")
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [Error] Failed during CPOP DAG creation/scheduling for '{dag_id}': {e}")
                print(traceback.format_exc())


    def submitter():
        """
        线程函数：从就绪列表中取出任务，检查是否到达其EST，然后提交。
        (此逻辑与HEFT版本完全相同，实现了时间感知的派发)
        """
        task_order= 1
        while True:
            task_to_submit = None
            
            with ready_list_lock:
                for run_id, func_name in ready_list:
                    dag_graph = dags_data[run_id]
                    task_node = dag_graph.nodes[func_name]
                    
                    task_est = task_node.get('est', 0.0)
                    schedule_start_time = dag_graph.graph.get('schedule_start_time', time.time())
                    
                    current_elapsed_time = time.time() - schedule_start_time
                    
                    if current_elapsed_time >= task_est:
                        task_to_submit = (run_id, func_name)
                        print(f"🎁 CPOP Submitter: Task '{func_name}' EST of {task_est:.2f} has been reached (current elapsed: {current_elapsed_time:.2f}). Picking for submission.")
                        ready_list.remove(task_to_submit)
                        break 
            
            if task_to_submit:
                run_id, func_name = task_to_submit
                dag_graph = dags_data[run_id]
                task_info = dict(dag_graph.nodes[func_name])
                dag_id= dag_graph.graph["dag_id"]
                task_info['node_id'] = task_to_node_map.get((run_id, func_name))
                if not task_info.get('node_id'):
                    print(f"❌ [FATAL] Could not find scheduled node for task '{func_name}'.")
                    continue

                task_info["dag_id"] = dag_id
                task_info["run_id"] = run_id                
                task_info["question"] = dag_graph.graph.get("question", "")
                task_info["answer"] = dag_graph.graph.get("answer", "")
                task_info["supplementary_file_paths"] = dag_graph.graph.get("supplementary_file_paths", {})
                task_info["dag_func_file"] = dag_graph.graph.get("dag_func_file", "")
                task_info["arrival_time"] = dag_graph.graph.get("arrival_time", time.time())
                task_info["priority"]= task_order
                task_order+= 1
                try:
                    spec = importlib.util.spec_from_file_location("dag_module", task_info["dag_func_file"])
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, func_name)
                    serialized_func = cloudpickle.dumps(func)
                    redis_func_key = f"func:{task_info['task_id']}"
                    redis_client.set(redis_func_key, serialized_func)
                    
                    print(f"  -> Submitting task '{func_name}' to the master scheduler, AFFINITY to node '{task_info['node_id']}'.")
                    requests.post(f"http://{args.master_addr}/inform", json=task_info)
                    monitor_que.put((run_id, func_name))
                except Exception as e:
                    print(f"❌ [Error] Failed to submit task '{func_name}': {e}")
            time.sleep(0.01)

    def monitor():
        """
        线程函数：通过监听Redis消息队列来高效地获取任务完成通知，
        并触发后续依赖任务。
        """
        # --- MODIFICATION START ---
        # This thread is completely rewritten for event-driven notifications.
        
        # 1. Create a Redis client for this specific thread
        redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=True)
        completion_queue_name = "task_completion_queue"
        print(f"💠 HEFT Monitor is now listening on Redis queue: '{completion_queue_name}'")

        while True:
            try:
                # 2. Efficiently block and wait for a message. No more polling.
                # The timeout is a fail-safe, it will wait indefinitely if set to 0.
                message = redis_client.brpop(completion_queue_name, timeout=0)
                if not message:
                    continue

                # message is a tuple (queue_name, data), we need the data part.
                notification = json.loads(message[1])
                dag_id = notification["dag_id"]
                run_id = notification["run_id"]
                task_id = notification["task_id"]
                func_name = notification["func_name"]
                status = notification["status"]
                
                print(f"✅ HEFT Monitor (Event-Driven): Received completion for task '{func_name}' with status '{status}'.")

                dag_graph = dags_data.get(run_id)
                if not dag_graph:
                    print(f"⚠️ Warning: Received notification for an unknown DAG ID '{dag_id}'. Skipping.")
                    continue
                
                # 3. Update the detailed execution stats in the shared dictionary.
                # This logic is still useful for the final /release call.
                try:
                    task_result_raw = redis_client.get(f"result:{task_id}") # The runner still stores detailed results here
                    task_result = json.loads(task_result_raw) if task_result_raw else {}
                    
                    start_exec_time = task_result.get("start_time", 0.0)
                    finish_exec_time = task_result.get("end_time", 0.0)
                    arrival_time = dag_graph.graph.get("arrival_time", 0.0)
                    sub_time = dag_graph.graph.get("sub_time", 0.0)
                    leave_time = time.time()

                    current_dag_status = dict(dag_status_dict[run_id])
                    task_status_info = current_dag_status.get(func_name, {})
                    task_status_info['status'] = status
                    task_status_info['start_exec_time'] = start_exec_time
                    task_status_info['finish_exec_time'] = finish_exec_time
                    task_status_info['sub_time']= sub_time
                    task_status_info['arrival_time'] = arrival_time
                    task_status_info['leave_time'] = leave_time
                    current_dag_status[func_name] = task_status_info
                    dag_status_dict[run_id] = current_dag_status
                except Exception as e:
                    print(f"❌ [Error] Failed to update dag_status_dict for '{func_name}': {e}")
                    # Even if stats update fails, we must mark status to unblock dependents
                    current_dag_status = dict(dag_status_dict[run_id])
                    current_dag_status.setdefault(func_name, {})['status'] = status
                    dag_status_dict[run_id] = current_dag_status

                if status != "finished":
                    continue

                # 4. Trigger dependent tasks (the core logic remains the same)
                dag_graph.nodes[func_name]['status'] = 'finished'
                for successor in dag_graph.successors(func_name):
                    all_preds_done = all(
                        dag_graph.nodes[pred].get('status') == 'finished' 
                        for pred in dag_graph.predecessors(successor)
                    )
                    
                    if all_preds_done:
                        with ready_list_lock:
                            if (run_id, successor) not in ready_list:
                                print(f"  -> Dependency met for '{successor}'. Added to ready list.")
                                ready_list.append((run_id, successor))
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [FATAL] An error occurred in the HEFT monitor thread: {e}")
                print(traceback.format_exc())
                time.sleep(5) # Avoid rapid-fire errors
    
    # --- 主函数逻辑 ---
    print("🚀 Starting CPOP scheduler manager (Strict Paper Version)...")
    creator_thread = threading.Thread(target=dag_creator, daemon=True)
    submitter_thread = threading.Thread(target=submitter, daemon=True)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    
    creator_thread.start()
    submitter_thread.start()
    monitor_thread.start()
    
    print("✅ CPOP scheduler manager and its worker threads are running.")
    creator_thread.join()