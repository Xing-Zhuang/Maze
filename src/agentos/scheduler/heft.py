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

from agentos.utils.scheduler_algorithm_utils import compute_ranku, TASK_TYPE_DEFAULT_EXEC_TIMES
from agentos.utils.query_loader import GaiaLoader, TBenchLoader,OpenAGILoader

query_loader_factory = {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader
}

def dag_manager_heft(args, dag_que, dag_status_dict):
    """
    一个严格遵循原始论文思想的HEFT算法动态调度管理器。
    - 核心思想: 
        1. 使用ranku值对任务进行优先级排序。
        2. 对于每个任务，遍历所有计算节点，找到能使其最早完成(EFT)的时间点和对应的最早开始时间(EST)。
        3. 将任务调度到最优节点。
    - 遵循原始Paper:
        - 本实现通过改造submitter线程，确保任务只有在“真实时间”达到其“预估开始时间(EST)”之后才会被派发。
        - 这种方式严格模拟了静态调度中的时间表，适用于科研和算法验证场景。
    """
    
    # --- 状态维护 ---
    dags_data = {}
    
    try:
        res = requests.post(f"http://{args.master_addr}/resource")
        res.raise_for_status()
        available_nodes = list(res.json().keys())
        if not available_nodes:
            raise ValueError("No available compute nodes found from master.")
        print(f"✅ HEFT Manager: Successfully discovered {len(available_nodes)} nodes: {available_nodes}")
    except Exception as e:
        print(f"❌ [FATAL] Failed to get available nodes from master: {e}. Exiting.")
        return

    proc_schedules = {node: [] for node in available_nodes}
    task_to_node_map = {} 
    
    # 就绪任务列表 (任务的数据依赖已满足)
    ready_list = []
    ready_list_lock = threading.Lock()
    
    monitor_que = queue.Queue()
    
    exec_time_db = {}
    redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=False)
    print("💠 HEFT manager (Strict Paper Version) initialized.")


    def find_insert_time_slot(schedule_list, ready_time, exec_time):
        """
        在节点的调度列表上为任务找到最早的可用时间插槽。
        (此函数无需修改，逻辑与原文一致)
        """
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

    def schedule_dag(run_id):
        """
        对单个DAG执行完整的HEFT调度算法，生成调度计划并填充初始就绪列表。
        (此函数无需修改，逻辑与原文一致)
        """
        dag_graph = dags_data[run_id]
        dag_id= dag_graph.graph["dag_id"]
        ranku_values = compute_ranku(dag_graph, exec_time_db, run_id)
        
        task_list = sorted(dag_graph.nodes(), key=lambda task: ranku_values.get(task, 0), reverse=True)
        
        print(f"💠 HEFT Scheduler: Planning for DAG '{dag_id}'. Task order: {task_list}")

        for task_name in task_list:
            ready_time = 0.0
            for pred in dag_graph.predecessors(task_name):
                pred_info = dag_graph.nodes[pred]
                ready_time = max(ready_time, pred_info.get('eft', 0.0))

            exec_time = exec_time_db.get((run_id, task_name))
            if exec_time is None:
                task_type = dag_graph.nodes[task_name].get('type')
                exec_time = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])

            best_node, best_eft, best_est = None, float('inf'), 0.0

            for node_id in proc_schedules.keys():
                schedule_on_node = proc_schedules[node_id]
                est = find_insert_time_slot(schedule_on_node, ready_time, exec_time)
                eft = est + exec_time
                
                if eft < best_eft:
                    best_eft, best_est, best_node = eft, est, node_id
            
            print(f"  -> Task '{task_name}' scheduled on node '{best_node}' at EST: {best_est:.2f}, finishes at EFT: {best_eft:.2f}")
            dag_graph.nodes[task_name]['est'] = best_est
            dag_graph.nodes[task_name]['eft'] = best_eft
            dag_graph.nodes[task_name]['node'] = best_node
            
            proc_schedules[best_node].append((best_est, best_eft, (run_id, task_name)))
            proc_schedules[best_node].sort(key=lambda x: x[0])
            task_to_node_map[(run_id, task_name)] = best_node

        with ready_list_lock:
            for node in dag_graph.nodes():
                if dag_graph.in_degree(node) == 0:
                    ready_list.append((run_id, node))
        print(f"  -> Initial ready tasks for DAG '{dag_id}' have been queued for submission.")

    def dag_creator():
        """
        线程函数：接收新DAG，为其生成调度计划，并记录调度开始时间。
        """
        while True:
            run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time = dag_que.get()
            print(f"💠 HEFT Creator: Received new DAG '{dag_id}'")

            try:
                query_loader = query_loader_factory.get(dag_source)
                loader = query_loader(args= args, dag_id= dag_id, run_id= run_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time= sub_time)
                dag_graph = loader.get_dag(task2id)
                dags_data[run_id] = dag_graph
                
                # --- MODIFICATION START ---
                # 记录调度过程的“0时刻”，用于后续计算真实等待时间
                dag_graph.graph['schedule_start_time'] = time.time()
                print(f"  -> Set schedule start time for DAG '{dag_id}' to {dag_graph.graph['schedule_start_time']:.2f}")
                # --- MODIFICATION END ---
                schedule_dag(run_id)
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [Error] Failed during DAG creation/scheduling for '{dag_id}': {e}")
                print(traceback.format_exc())

    def submitter():
        """
        线程函数：从就绪列表中取出任务，检查是否到达其EST，然后提交。
        """
        # --- MODIFICATION START ---
        # 重写整个submitter的逻辑
        task_order= 1
        while True:
            task_to_submit = None
            
            with ready_list_lock:
                # 遍历就绪列表，寻找可以提交的任务
                for run_id, func_name in ready_list:
                    dag_graph = dags_data[run_id]
                    task_node = dag_graph.nodes[func_name]
                    
                    # 获取预估的开始时间(EST)和调度参考的0时刻
                    task_est = task_node.get('est', 0.0)
                    schedule_start_time = dag_graph.graph.get('schedule_start_time', time.time())
                    
                    # 计算从调度开始到当前过去了多少真实时间
                    current_elapsed_time = time.time() - schedule_start_time
                    
                    # 只有当真实流逝时间 >= 预估开始时间，任务才被派发
                    if current_elapsed_time >= task_est:
                        task_to_submit = (run_id, func_name)
                        print(f"🎁 HEFT Submitter: Task '{func_name}' EST of {task_est:.2f} has been reached (current elapsed: {current_elapsed_time:.2f}). Picking for submission.")
                        ready_list.remove(task_to_submit)
                        break # 一次只提交一个，保持循环简单
            
            if task_to_submit:
                run_id, func_name = task_to_submit
                dag_graph = dags_data[run_id]
                task_info = dict(dag_graph.nodes[func_name])
                
                task_info['node_id'] = task_to_node_map.get((run_id, func_name))
                if not task_info.get('node_id'):
                    print(f"❌ [FATAL] Could not find scheduled node for task '{func_name}'.")
                    continue

                task_info["run_id"] = run_id
                task_info["dag_id"] = dag_graph.graph["dag_id"]
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
                    
                    redis_client.set(f"func:{task_info['task_id']}", serialized_func)
                    
                    print(f"  -> Submitting task '{func_name}' to the master scheduler, AFFINITY to node '{task_info['node_id']}'.")
                    requests.post(f"http://{args.master_addr}/inform", json=task_info)
                    monitor_que.put((run_id, func_name))
                except Exception as e:
                    print(f"❌ [Error] Failed to submit task '{func_name}': {e}")
            # 如果没有任务到达EST，短暂休眠，避免CPU空转
            time.sleep(0.01)
        # --- MODIFICATION END ---

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
                run_id= notification["run_id"]
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
                    sub_time= dag_graph.graph.get("sub_time", 0.0)
                    leave_time = time.time()

                    current_dag_status = dict(dag_status_dict[run_id])
                    task_status_info = current_dag_status.get(func_name, {})
                    task_status_info['status'] = status
                    task_status_info['sub_time']= sub_time
                    task_status_info['start_exec_time'] = start_exec_time
                    task_status_info['finish_exec_time'] = finish_exec_time
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
    print("🚀 Starting HEFT scheduler manager (Strict Paper Version)...")
    creator_thread = threading.Thread(target=dag_creator, daemon=True)
    submitter_thread = threading.Thread(target=submitter, daemon=True)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    
    creator_thread.start()
    submitter_thread.start()
    monitor_thread.start()
    
    print("✅ HEFT scheduler manager and its worker threads are running.")
    creator_thread.join()