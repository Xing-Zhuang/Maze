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
from agentos.utils.query_loader import GaiaLoader
from agentos.utils.query_loader import TBenchLoader
from agentos.utils.query_loader import OpenAGILoader
query_loader_factory= {
    "gaia": GaiaLoader,
    "tbench": TBenchLoader,
    "openagi": OpenAGILoader
}

class PriorityQueue:
    def __init__(self):
        self._queue= []
        self._lock= threading.Lock()
        self._index= 0  # 用于稳定排序

    def put(self, time, dag_id, func_name):
        with self._lock:
            heapq.heappush(self._queue, (-time, self._index, dag_id, func_name))
            self._index+= 1

    def get(self):
        with self._lock:
            if not self._queue:
                raise IndexError("pop from an empty priority queue")
            pop_ele= heapq.heappop(self._queue)
            return (pop_ele[0], pop_ele[2], pop_ele[3])

    def peek(self):
        with self._lock:
            if not self._queue:
                raise IndexError("peek from an empty priority queue")
            pop_ele= self._queue[0]
            return (pop_ele[0], pop_ele[2], pop_ele[3])
    
    def empty(self):
        with self._lock:
            length= len(self._queue)
        return  length== 0
    
        
class MLQ:
    def __init__(self)-> None:
        self.mlq= []
        self.lock= threading.Lock()
        
    def put(self, run_id: str, func_name: str, time: float, layer: int):
        if(layer+ 1> len(self.mlq)):
            with self.lock:
                self.mlq.append(PriorityQueue())
        self.mlq[layer].put(time,run_id,func_name)

def dag_manager_mlq(args, dag_que, dag_status_dict):
      
    dags= {}
    submit_queue= MLQ()
    predictor= DAGTaskPredictor(args.redis_ip, args.redis_port, args.time_pred_model_path, args.min_sample4train, args.min_sample4incremental)
    redis_client= redis.Redis(host= args.redis_ip, port= args.redis_port, decode_responses= False)
    monitor_que= queue.Queue()

    def dag_creator():
        def calculate_layer(dag): # 根据拓扑排序计算任务所在DAG层数
            # 拓扑排序
            topo_order= list(nx.topological_sort(dag))
            for node in topo_order:
                # 如果节点没有前驱，则其层次为0
                if dag.in_degree(node)== 0:
                    dag.nodes[node]['layer']= 0
                else:
                    # 否则，其层次为其所有前驱节点的最大层次加1
                    layer= max(
                        dag.nodes[pred]['layer']
                        for pred in dag.predecessors(node)
                    )+ 1  
                    dag.nodes[node]['layer']= layer
        
        while True:
            run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time= dag_que.get()
            try:
                query_loader= query_loader_factory.get(dag_source)
                if not query_loader:
                    print(f"❌ [Error] No loader found for dag_source: {dag_source}")
                    continue
                loader = query_loader(args= args, dag_id= dag_id, run_id= run_id, dag_type= dag_type, dag_source= dag_source, supplementary_files= supplementary_files, sub_time= sub_time)
                dag= loader.get_dag(task2id)
                dags[run_id]= dag
                calculate_layer(dag)
                print(f"😊 Putting task into queue...")
                #计算每个node的入度信息，并将start node放入mlq的第0级队列
                for node, in_degree in dag.in_degree():
                    node_info= dag.nodes[node]
                    node_info["in_degree"]= in_degree
                    node_info["arrival_time"]= dag.graph['arrival_time']
                    if in_degree== 0:
                        submit_queue.put(run_id= run_id, func_name= node_info["func_name"] , time= -1, layer= 0)
                        print(f"😊 Putting dag_id: {dag_id}(run_id: {run_id})'s func_name: {node_info['func_name']} into queue...")
            except Exception as e:
                print(f"❌ [Error] Failed to create DAG for {dag_id}(Run: {run_id}): {e}")
            time.sleep(0.01)
            
    def submitter():
        #创建 Redis 连接，序列化函数存入redis
        task_order= 1
        while True:
            for pq in submit_queue.mlq:  #找到第一个非空的队列，提交其队首的任务（以这个操作为最小粒度，一直重复这个过程）
                if not pq.empty():
                    _, run_id, func_name= pq.get()
                    dag= dags[run_id]
                    if "question" not in dag.graph or "dag_func_file" not in dag.graph:
                        raise ValueError(
                            f"The essential key 'question' and 'dag_func_file' was not found in the DAG's metadata (dag.graph). "
                            f"Please ensure that the corresponding dataset loader class (e.g., GaiaLoader) "
                            f"provides an implementation to populate this value."
                        )
                    node1dag= dict(dag.nodes[func_name])
                    node1dag["run_id"] = run_id
                    node1dag["dag_id"] = dag.graph["dag_id"] # 保留静态ID
                    node1dag["question"]= dag.graph["question"]
                    node1dag["answer"]= dag.graph["answer"]
                    node1dag["supplementary_file_paths"]= dag.graph["supplementary_file_paths"]
                    node1dag["dag_func_file"]= dag.graph["dag_func_file"]
                    node1dag["arrival_time"]= dag.graph["arrival_time"]
                    node1dag["priority"]= task_order
                    task_order+= 1
                    print(f"🎁 提交任务: {func_name}, run_id:{run_id}, task_id:{node1dag['task_id']}")
                    # 函数序列化
                    spec = importlib.util.spec_from_file_location("dag", node1dag["dag_func_file"])
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, func_name)
                    serialized_func = cloudpickle.dumps(func)
                    redis_client.set(f"func:{node1dag['task_id']}", serialized_func)
                    print(f"✅ Function '{func_name}' serialized to Redis.")
                    #通知master
                    try:
                        requests.post(f"http://{args.master_addr}/inform", json= dict(node1dag))
                    except requests.exceptions.RequestException as e:
                        print(f"❌ 通知master失败: {e}")
                    # 放入monitor_que
                    monitor_que.put((run_id, func_name))
                    break # 每次只提交一个任务
            time.sleep(0.01) # 如果所有队列都为空，则短暂休眠
    
    def monitor():
        redis_client_monitor = redis.Redis(host=args.redis_ip, port=args.redis_port, decode_responses=True)
        completion_queue_name = "task_completion_queue"
        print(f"💠 Monitor is now listening on Redis queue: '{completion_queue_name}'")

        while True:
            try:
                # 1. 等待任何任务完成的通知
                _, message = redis_client_monitor.brpop(completion_queue_name)
                if not message:
                    continue

                notification = json.loads(message)
                run_id = notification["run_id"]
                func_name = notification["func_name"]
                status = notification["status"]

                print(f"✅ Monitor: Received completion for '{func_name}' (Run ID: {run_id}) with status '{status}'.")

                dag = dags.get(run_id)
                if not dag: continue
                
                task_id = dag.nodes[func_name].get("task_id")
                if not task_id: continue

                # 2. 立即为刚刚完成的任务收集数据 (无论它是不是叶子节点)
                task_result_raw = redis_client_monitor.get(f"result:{task_id}")
                if task_result_raw:
                    record_json = json.loads(task_result_raw)
                    predictor.collect_data_for_task(
                        task_id=task_id,
                        func_name=func_name,
                        record_json=record_json
                    )

                # 3. Update the shared dag_status_dict with detailed stats
                try:
                    task_result = json.loads(task_result_raw) if task_result_raw else {}
                    start_exec_time = task_result.get("start_time", 0.0)
                    finish_exec_time = task_result.get("end_time", 0.0)
                    current_status = dict(dag_status_dict.get(run_id, {}))
                    status_entry = current_status.setdefault(func_name, {})
                    status_entry["start_exec_time"] = start_exec_time
                    status_entry["finish_exec_time"] = finish_exec_time
                    status_entry["arrival_time"] = dag.graph["arrival_time"]
                    status_entry["sub_time"] = dag.graph["sub_time"]
                    status_entry["leave_time"] = time.time()
                    status_entry["status"] = status
                    dag_status_dict[run_id] = current_status
                except Exception as e:
                    print(f"❌ Error updating dag_status_dict for '{func_name}': {e}")

                if status != "finished":
                    continue

                # 4. Handle successor unlocking using MLQ's specific logic
                successors = list(dag.successors(func_name))
                if not successors:
                    print(f"  -> Task '{func_name}' is a leaf node. No successors to predict.")
                    continue
                for successor_id in successors:
                    successor = dag.nodes[successor_id]
                    successor["in_degree"] -= 1
                    if successor["in_degree"] == 0:
                        print(f"  -> Successor task '{successor_id}' is now ready.")
                        succ_task_type= successor.get('type', 'cpu')
                        # 4. 只为已就绪的后继任务调用预测方法
                        pred_task_ids = [dag.nodes[p].get("task_id") for p in dag.predecessors(successor_id)]
                        pred_time = predictor.predict(
                            succ_func_name=successor_id,
                            succ_task_type= succ_task_type,
                            pred_task_ids=pred_task_ids
                        )
                        
                        submit_queue.put(
                            run_id=run_id,
                            func_name=successor_id,
                            time=pred_time,
                            layer=dag.nodes[func_name]["layer"] + 1
                        )
                        print(f"  -> Added '{successor_id}' to MLQ with predicted priority.")
                time.sleep(0.01)
            except Exception as e:
                print(f"❌ [FATAL] An error occurred in the monitor thread: {e}")
                import traceback
                print(f"❌ monitor have met an error: {traceback.format_exc()}")
                time.sleep(5)

    create_dag_thread= threading.Thread(target= dag_creator)
    create_dag_thread.start()
    print(f"😊 create_dag_thread running...")
    monitor_thread= threading.Thread(target= monitor) 
    monitor_thread.start()
    print(f"😊 monitor running...")    
    submitter_thread= threading.Thread(target= submitter) 
    submitter_thread.start()
    print(f"😊 submitter running...")