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
import numpy as np # 引入numpy用于计算平均值

from agentos.utils.scheduler_algorithm_utils import TASK_TYPE_DEFAULT_EXEC_TIMES
from agentos.utils.query_loader import GaiaLoader, TBenchLoader,OpenAGILoader

# query_loader_factory 的定义保持不变
query_loader_factory = {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader
}

def dag_manager_peft(args, dag_que, dag_status_dict):
    """
    一个严格遵循原始论文思想的PEFT算法动态调度管理器。
    - 核心思想: 
        1. 首先计算乐观成本表 (Optimistic Cost Table, OCT)。
        2. 使用基于OCT计算出的rank_oct值对任务进行优先级排序。
        3. 对于每个任务，遍历所有计算节点，找到能使其乐观最早完成时间(O_EFT = EFT + OCT)最小的节点。
        4. 将任务调度到最优节点。
    - 遵循原始Paper:
        - 本实现通过改造submitter线程，确保任务只有在“真实时间”达到其“预估开始时间(EST)”之后才会被派发。
        - 这种方式严格模拟了静态调度中的时间表，适用于科研和算法验证场景。
    """
    
    # --- 状态维护 (与HEFT相同) ---
    dags_data = {}
    
    try:
        res = requests.post(f"http://{args.master_addr}/resource")
        res.raise_for_status()
        available_nodes = list(res.json().keys())
        if not available_nodes:
            raise ValueError("No available compute nodes found from master.")
        print(f"✅ PEFT Manager: Successfully discovered {len(available_nodes)} nodes: {available_nodes}")
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
    print("💠 PEFT manager (Strict Paper Version) initialized.")


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

    # --- PEFT核心改动开始 ---

    def compute_oct_and_rank_oct(dag_graph, run_id):
        """
        计算乐观成本表 (OCT) 和 rank_oct。
        这是PEFT算法的核心。
        """
        oct_table = {task: {proc: 0.0 for proc in available_nodes} for task in dag_graph.nodes()}
        rank_oct_values = {task: 0.0 for task in dag_graph.nodes()}

        # 按照反向拓扑顺序遍历DAG
        for task_i in reversed(list(nx.topological_sort(dag_graph))):
            # 如果是出口任务，其OCT值为0
            if dag_graph.out_degree(task_i) == 0:
                continue

            # 对于每个处理器pk，计算task_i的OCT值
            for pk in available_nodes:
                max_succ_val = 0
                # 遍历task_i的所有后继任务
                for task_j in dag_graph.successors(task_i):
                    min_child_val = float('inf')
                    # 遍历所有可能的处理器pw来执行后继任务task_j
                    for pw in available_nodes:
                        # 获取task_j在pw上的执行时间。注意：PEFT原文假设w(tj,pw)已知。
                        # 这里我们简化为使用平均执行时间，与HEFT实现保持一致。
                        # 在一个更复杂的模型中，这里应该查询一个 processor-specific 的成本矩阵。
                        w_ij = exec_time_db.get((run_id, task_j))
                        if w_ij is None:
                            task_type = dag_graph.nodes[task_j].get('type')
                            w_ij = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])
                        # OCT(tj, pw) + w(tj, pw)
                        # 注意：原文中的通信成本c_ij在这里被简化了，因为它在现有框架中未明确建模。
                        # 如果pw=pk，通信成本为0。这个逻辑隐含在ready_time计算中。
                        current_val = oct_table[task_j][pw] + w_ij
                        min_child_val = min(min_child_val, current_val)
                    
                    max_succ_val = max(max_succ_val, min_child_val)
                
                oct_table[task_i][pk] = max_succ_val
            
            # 计算 rank_oct，即该任务在所有处理器上OCT值的平均值
            rank_oct_values[task_i] = np.mean(list(oct_table[task_i].values()))
            
        return oct_table, rank_oct_values

    def schedule_dag(run_id):
        """
        对单个DAG执行完整的PEFT调度算法，生成调度计划并填充初始就绪列表。
        """
        dag_graph = dags_data[run_id]
        dag_id = dag_graph.graph["dag_id"]

        # 1. 计算OCT和rank_oct，替换HEFT中的ranku
        oct_table, rank_oct_values = compute_oct_and_rank_oct(dag_graph, run_id)
        
        # 2. 根据rank_oct对任务进行降序排序
        task_list = sorted(dag_graph.nodes(), key=lambda task: rank_oct_values.get(task, 0), reverse=True)
        
        print(f"💠 PEFT Scheduler: Planning for DAG '{dag_id}'. Task order: {task_list}")

        for task_name in task_list:
            ready_time = 0.0
            for pred in dag_graph.predecessors(task_name):
                pred_info = dag_graph.nodes[pred]
                ready_time = max(ready_time, pred_info.get('eft', 0.0))

            exec_time = exec_time_db.get((run_id, task_name))
            if exec_time is None:
                task_type = dag_graph.nodes[task_name].get('type')
                exec_time = TASK_TYPE_DEFAULT_EXEC_TIMES.get(task_type, TASK_TYPE_DEFAULT_EXEC_TIMES['default'])
            # 3. 寻找最小的 O_EFT (Optimistic EFT)，而不是EFT
            best_node, best_eft, best_est = None, float('inf'), 0.0
            best_o_eft = float('inf')

            for node_id in proc_schedules.keys():
                schedule_on_node = proc_schedules[node_id]
                est = find_insert_time_slot(schedule_on_node, ready_time, exec_time)
                eft = est + exec_time
                
                # 计算 O_EFT = EFT + OCT
                o_eft = eft + oct_table[task_name][node_id]
                
                # 决策基于 O_EFT
                if o_eft < best_o_eft:
                    best_o_eft = o_eft
                    best_eft, best_est, best_node = eft, est, node_id
            
            print(f"   -> Task '{task_name}' scheduled on node '{best_node}' at EST: {best_est:.2f}, EFT: {best_eft:.2f} (O_EFT: {best_o_eft:.2f})")
            dag_graph.nodes[task_name]['est'] = best_est
            dag_graph.nodes[task_name]['eft'] = best_eft
            dag_graph.nodes[task_name]['node'] = best_node
            
            proc_schedules[best_node].append((best_est, best_eft, (run_id, task_name)))
            proc_schedules[best_node].sort(key=lambda x: x[0])
            task_to_node_map[(run_id, task_name)] = best_node

        # 将入度为0的任务加入就绪队列 (逻辑不变)
        with ready_list_lock:
            for node in dag_graph.nodes():
                if dag_graph.in_degree(node) == 0:
                    ready_list.append((run_id, node))
        print(f"   -> Initial ready tasks for DAG '{dag_id}' have been queued for submission.")

    # --- PEFT核心改动结束 ---


    # --- dag_creator, submitter, monitor 线程保持不变 ---
    # 它们负责动态执行由 schedule_dag 生成的静态计划
    # 只需要将print语句中的 "HEFT" 改为 "PEFT" 即可。

    def dag_creator():
        """
        线程函数：接收新DAG，为其生成调度计划，并记录调度开始时间。
        """
        while True:
            run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time = dag_que.get()
            print(f"💠 PEFT Creator: Received new DAG '{dag_id}'")

            try:
                query_loader = query_loader_factory.get(dag_source)
                loader = query_loader(args= args, dag_id= dag_id, run_id= run_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time= sub_time)
                dag_graph = loader.get_dag(task2id)
                dags_data[run_id] = dag_graph
                
                dag_graph.graph['schedule_start_time'] = time.time()
                print(f"   -> Set schedule start time for DAG '{dag_id}' to {dag_graph.graph['schedule_start_time']:.2f}")

                schedule_dag(run_id)
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [Error] Failed during DAG creation/scheduling for '{dag_id}': {e}")
                print(traceback.format_exc())

    def submitter():
        """
        线程函数：从就绪列表中取出任务，检查是否到达其EST，然后提交。
        (逻辑与HEFT版本完全相同)
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
                        print(f"🎁 PEFT Submitter: Task '{func_name}' EST of {task_est:.2f} has been reached (current elapsed: {current_elapsed_time:.2f}). Picking for submission.")
                        ready_list.remove(task_to_submit)
                        break 
            
            if task_to_submit:
                run_id, func_name = task_to_submit
                dag_graph = dags_data[run_id]
                task_info = dict(dag_graph.nodes[func_name])
                
                task_info['node_id'] = task_to_node_map.get((run_id, func_name))
                if not task_info.get('node_id'):
                    print(f"❌ [FATAL] Could not find scheduled node for task '{func_name}'.")
                    continue

                # ... (后续提交逻辑与HEFT完全一致，此处省略以保持简洁)
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
                    
                    print(f"   -> Submitting task '{func_name}' to the master scheduler, AFFINITY to node '{task_info['node_id']}'.")
                    requests.post(f"http://{args.master_addr}/inform", json=task_info)
                    monitor_que.put((run_id, func_name))
                except Exception as e:
                    print(f"❌ [Error] Failed to submit task '{func_name}': {e}")

            time.sleep(0.01)

    def monitor():
        """
        线程函数：监听任务完成，并更新依赖。
        (逻辑与HEFT版本完全相同)
        """
        redis_client = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=True)
        completion_queue_name = "task_completion_queue"
        print(f"💠 PEFT Monitor is now listening on Redis queue: '{completion_queue_name}'")

        while True:
            try:
                message = redis_client.brpop(completion_queue_name, timeout=0)
                if not message:
                    continue

                notification = json.loads(message[1])
                dag_id = notification["dag_id"]
                run_id = notification["run_id"]
                task_id = notification["task_id"]
                func_name = notification["func_name"]
                status = notification["status"]
                
                print(f"✅ PEFT Monitor (Event-Driven): Received completion for task '{func_name}' with status '{status}'.")

                dag_graph = dags_data.get(run_id)
                if not dag_graph:
                    print(f"⚠️ Warning: Received notification for an unknown DAG ID '{dag_id}'. Skipping.")
                    continue
                
                # ... (后续状态更新和依赖解锁逻辑与HEFT完全一致，此处省略以保持简洁)
                try:
                    task_result_raw = redis_client.get(f"result:{task_id}")
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
                    current_dag_status = dict(dag_status_dict[run_id])
                    current_dag_status.setdefault(func_name, {})['status'] = status
                    dag_status_dict[run_id] = current_dag_status

                if status != "finished":
                    continue

                dag_graph.nodes[func_name]['status'] = 'finished'
                for successor in dag_graph.successors(func_name):
                    all_preds_done = all(
                        dag_graph.nodes[pred].get('status') == 'finished' 
                        for pred in dag_graph.predecessors(successor)
                    )
                    
                    if all_preds_done:
                        with ready_list_lock:
                            if (run_id, successor) not in ready_list:
                                print(f"   -> Dependency met for '{successor}'. Added to ready list.")
                                ready_list.append((run_id, successor))
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [FATAL] An error occurred in the PEFT monitor thread: {e}")
                print(traceback.format_exc())
                time.sleep(5) 
    
    # --- 主函数逻辑 (启动线程) ---
    print("🚀 Starting PEFT scheduler manager (Strict Paper Version)...")
    creator_thread = threading.Thread(target=dag_creator, daemon=True)
    submitter_thread = threading.Thread(target=submitter, daemon=True)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    
    creator_thread.start()
    submitter_thread.start()
    monitor_thread.start()
    
    print("✅ PEFT scheduler manager and its worker threads are running.")
    creator_thread.join()