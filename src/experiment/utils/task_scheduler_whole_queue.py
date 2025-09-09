import os
import ray
import time
import queue
import redis
import random
import threading
from typing import Optional, Dict, List, Tuple
import tracemalloc
import requests
from agentos.utils.execution_backend import VLLMBackend, HuggingFaceBackend
import heapq
import gc
from collections import deque, defaultdict
import json

def write_log(dag_id,run_id,task_id,dag_func_file,func_name,status):
    log_dir = './log'
    log_file = os.path.join(log_dir, 'log.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as file:
        file.write(f"{dag_id},{run_id},{task_id},{dag_func_file},{func_name},{status},{time.time()}\n")

@ray.remote(num_cpus=0, max_calls=1)
def remote_task_runner(serialized_func: bytes, ctx_actor: object, task_id: str, redis_host:str, redis_port:int, task_type:str, gpu_indices: Optional[List[int]])-> Dict[str, str]:
    try:
        import os
        import json
        import redis
        import cloudpickle
        import torch
        func= cloudpickle.loads(serialized_func)
       
        if task_type == "gpu" and gpu_indices:
            visible_devices = ",".join(map(str, gpu_indices))
            print(f"  -> Setting CUDA_VISIBLE_DEVICES='{visible_devices}' for task {task_id}.")
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        tracemalloc.start()
        result = func(ctx_actor)
        torch.cuda.empty_cache()
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{task_id}: peak mem {peak / (1024 ** 2)} MB")
        
        if result:
            r = redis.Redis(host= redis_host, port= redis_port)
            r.set(f"result:{task_id}", result)
           
        return {"status": "finished"}
    except Exception as e:
        error_message= str(e)
        print(f"[FAILED] Task {task_id} failed with error: {error_message}")
        return {"status": "failed", "err_msg": error_message}

class TaskScheduler:
    def __init__(self, resource_mgr: object, status_mgr: object, dag_ctx_mgr: object, redis_ip:str="127.0.0.1",redis_port: int= 6379, proj_path: str= "", model_folder= "model_cache", models_config_path: str= "")-> None:
        self.master_node_id = ray.get_runtime_context().get_node_id()
        self.resource_mgr = resource_mgr
        self.status_mgr = status_mgr
        self.dag_ctx_mgr = dag_ctx_mgr
        self.models_config = {}
        self.models_config_path = os.path.join(proj_path, models_config_path)
        try:
            with open(self.models_config_path, 'r') as f:
                self.models_config = json.load(f)
            print(f"✅ TaskScheduler: Model configuration loaded from '{self.models_config_path}'.")
        except FileNotFoundError:
            print(f"⚠️ TaskScheduler: models_config_path '{self.models_config_path}' not found. No model configs loaded.")
        except json.JSONDecodeError:
            print(f"❌ TaskScheduler: Error decoding JSON from '{self.models_config_path}'.")
        self.backends = {
            "vllm": VLLMBackend(
                resource_manager=self.resource_mgr,
                proj_path= proj_path,
                model_folder= model_folder,
                models_config_path= self.models_config_path
            ),
            "huggingface": HuggingFaceBackend()
        }
        
        # --- ABLATION MODIFICATION START ---
        # Original multi-queues are commented out
        # self.task_queue_cpu = queue.PriorityQueue()
        # self.task_queue_gpu = queue.PriorityQueue()
        # self.task_queue_io = queue.PriorityQueue()
        
        # A single, unified queue for all task types
        self.unified_task_queue = queue.PriorityQueue()
        print("✅ [Ablation Study] Initialized with a SINGLE UNIFIED task queue.")
        # --- ABLATION MODIFICATION END ---

        self.vllm_waiting_queue= deque()
        self.vllm_replica_load = defaultdict(int)
        self.VLLM_RESERVED_FREE_GPUS = 1
        self.VLLM_LOAD_THRESHOLD = 5 # vLLM副本负载阈值
        self.VLLM_EVICTION_GRACE_PERIOD = 60
        self.Check_VLLM_Interval = 10 # seconds
        self.redis_ip = redis_ip
        self.redis_port = redis_port
        self.redis_client = redis.Redis(host= redis_ip, port= redis_port)
        self.running_tasks = []
        self.running_tasks_lock = threading.Lock()
        self.resource_lock = threading.Lock()

        self.SINGLE_GPU_MEM_THRESHOLD = self._calculate_min_gpu_memory_threshold()
        print(f"✅ Dynamically calculated single GPU memory threshold: {self.SINGLE_GPU_MEM_THRESHOLD} MiB")

        # --- ABLATION MODIFICATION START ---
        # Start the new unified scheduler loop instead of the old three
        self.start_unified_scheduler_loop()
        # self.start_cpu_scheduler_loop()
        # self.start_io_scheduler_loop()
        # self.start_gpu_scheduler_loop()
        # --- ABLATION MODIFICATION END ---

        self.start_result_monitor()
        self.start_vllm_monitor_loop()
        self.bug_out_control= 0

    def submit(self, task_info, priority= None):
        """
        Submits a task to the scheduler. In this ablated version, all tasks go
        into the single unified queue.
        """
        self.status_mgr.add_task(task_info)
        if priority is None:
            priority= time.time()
        if isinstance(priority, list):
            priority = tuple(priority)
        elif isinstance(priority, (int, float)):
            priority = (priority,)
        
        print(f"💮 now receive submit from dag_id: {task_info['dag_id']}, task_id: {task_info['task_id']}")
        item_to_queue= (priority, task_info)
        
        # --- ABLATION MODIFICATION: All tasks go to the unified queue ---
        self.unified_task_queue.put(item_to_queue)

    def start_unified_scheduler_loop(self):
        """
        A single, unified scheduler loop that replaces the three separate loops.
        It dequeues the highest-priority task from the unified queue and delegates
        it to the appropriate handler based on its type.
        """
        def loop():
            while True:
                # --- VLLM Waiting Queue Logic (Integrated into the main loop) ---
                # This part is crucial: we must first check if a waiting vLLM task can be activated.
                # If so, it gets re-queued with high priority into our unified queue.
                if self.vllm_waiting_queue:
                    _, task_info_peek = self.vllm_waiting_queue[0]
                    model_name_peek = task_info_peek.get("model_name")
                    ready_replicas = self.resource_mgr.find_gpus_by_model(
                        model_name_peek, "vllm", status="OCCUPIED"
                    )
                    available_replicas = [r for r in ready_replicas if not self._is_vllm_replica_full(r.get("request_api_url"))]
                    if available_replicas:
                        available_replica = min(
                            available_replicas,
                            key=lambda r: self.vllm_replica_load.get(r.get("request_api_url"), 0)
                        )
                        priority, task_info = self.vllm_waiting_queue.popleft()
                        target_api_url = available_replica.get('request_api_url')
                        task_info[f'{task_info["func_name"]}_request_api_url'] = target_api_url
                        self.vllm_replica_load[target_api_url] += 1
                        model_mem = self.models_config.get(task_info.get("model_name"), {}).get("gpu_mem", 80000)
                        
                        # Re-queue the activated task into the main unified queue
                        remaining_gpu_mem = task_info.get('gpu_mem', 0) - model_mem
                        task_info['gpu_mem'] = remaining_gpu_mem
                        # Give it the highest possible priority to be processed next
                        self.unified_task_queue.put(((float('-inf'),) + priority, task_info)) 
                        print(f"✅ Activated waiting task '{task_info['func_name']}' for model '{model_name_peek}'. Re-enqueued with high priority.")
                        continue # Restart loop to process the newly activated task

                if self.unified_task_queue.empty():
                    time.sleep(0.05)
                    continue
                
                priority, task_info = self.unified_task_queue.get()
                task_type = task_info.get("type", "cpu")
                
                placement_found = False
                if task_type == "gpu":
                    placement_found = self._handle_gpu_task(priority, task_info)
                elif task_type == "cpu":
                    placement_found = self._handle_cpu_task(priority, task_info)
                elif task_type == "io":
                    placement_found = self._handle_io_task(priority, task_info)

                if not placement_found:
                    # If placement was not found (e.g., no resources), put it back in the queue
                    self.unified_task_queue.put((priority, task_info))
                
                time.sleep(0.01) # A small sleep to prevent busy-waiting
            
        threading.Thread(target=loop, daemon=True, name="UnifiedScheduler").start()
        print("✅ [Ablation Study] Unified Scheduler Loop Started.")

    def _handle_gpu_task(self, priority, task_info) -> bool:
        """
        Contains the scheduling logic specifically for GPU tasks.
        This is the refactored logic from the original start_gpu_scheduler_loop.
        Returns True if the task was placed and dispatched, False otherwise.
        """
        run_id, task_id, pre_scheduled_node_id, func_name, model_name, backend_type= task_info["run_id"], task_info["task_id"], task_info.get("node_id"), task_info.get("func_name"), task_info.get("model_name"), task_info.get("backend", "huggingface")
        request_api_url= task_info.get(f"{func_name}_request_api_url", None)
        backend = self.backends[backend_type]
        gpu_indices_for_dispatch = []
        dag_ctx= self.dag_ctx_mgr.get_context(run_id)
        placement_found, selected_node_id, target_api_url= False, None, None

        with self.resource_lock:
            if backend_type == 'vllm' and not request_api_url:
                replicas = self.resource_mgr.find_gpus_by_model(model_name, "vllm", "OCCUPIED")
                ready_replicas = [r for r in replicas if not self._is_vllm_replica_full(r.get("request_api_url"))]
                model_mem = self.models_config.get(model_name, {}).get("gpu_mem", 80000)
                if ready_replicas:
                    available_replica = min(ready_replicas, key=lambda r: self.vllm_replica_load.get(r.get("request_api_url"), 0))
                    target_api_url= available_replica['request_api_url']
                    task_info[f"{func_name}_request_api_url"]= target_api_url
                    self.vllm_replica_load[target_api_url]+= 1
                    self.choice_cpu_gpu_queue_according_resource(task_info.get('gpu_mem', 0) - model_mem, priority, task_info)
                    print(f"✅ Found existing replica for '{model_name}' at {target_api_url}.")
                    return True # Task was re-queued, so it's "handled" for this loop iteration
                else:
                    should_deploy_model= False
                    num_requests_waiting= 1+ sum(1 for _, t_info in self.vllm_waiting_queue if t_info.get("model_name")== model_name)
                    ready_replicas_count= len(replicas)
                    deploying_replicas_count= len(set(gpu.get("runner_key") for gpu in self.resource_mgr.find_all_gpus_by_state("DEPLOYING") if gpu.get("model_name") == model_name))
                    replicas_count= ready_replicas_count+ deploying_replicas_count
                    should_deploy_model= True if replicas_count== 0 or num_requests_waiting* 1.0/ replicas_count> self.VLLM_LOAD_THRESHOLD else False
                    if should_deploy_model:
                        num_free_gpus = len(self.resource_mgr.find_all_gpus_by_state("FREE"))
                        if num_free_gpus> self.VLLM_RESERVED_FREE_GPUS:
                            deployment_node_id, deployment_indices, runner_key= None, None, None
                            for node_id_search in self.resource_mgr.node2avai_resources.keys():
                                can_deploy, indices = self._find_gpu_placement_on_node(node_id_search, {"gpu_mem": model_mem})
                                if can_deploy:
                                    deployment_node_id, deployment_indices = node_id_search, indices
                                    runner_key= (deployment_node_id, frozenset(deployment_indices))
                                    break
                            if deployment_node_id:
                                print(f"💡 Triggering deployment for '{model_name}' on Node {deployment_node_id[:6]}, GPU(s) {deployment_indices}.")
                                backend.deploy(deployment_node_id, deployment_indices, model_name)
                                self.resource_mgr.update_gpu_state(deployment_node_id, deployment_indices, {"status": "DEPLOYING", "model_name": model_name, "backend": "vllm", "runner_key": runner_key})
                                self.resource_mgr.reduce_node_resource(deployment_node_id, deployment_indices, {"mem": 0, "type": "gpu", "gpu_mem": model_mem})
                                self.vllm_waiting_queue.append((priority, task_info))
                                return True # Task moved to waiting queue
                    else:
                        self.vllm_waiting_queue.append((priority, task_info))
                        return True # Task moved to waiting queue
            else:
                if pre_scheduled_node_id:
                    can_run, indices = self._find_gpu_placement_on_node(pre_scheduled_node_id, task_info)
                    if can_run:
                        placement_found, selected_node_id, gpu_indices_for_dispatch = True, pre_scheduled_node_id, indices
                        if not dag_ctx:
                            node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                            dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                            self._prepare_dag_context(task_info, dag_ctx)
                else:
                    if dag_ctx:
                        affinity_node_id= self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                        can_run, indices= self._find_gpu_placement_on_node(affinity_node_id, task_info)
                        if can_run:
                            placement_found, selected_node_id, gpu_indices_for_dispatch = True, affinity_node_id, indices
                    if not placement_found:
                        affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                        candidate_nodes = [node_id for node_id in self.resource_mgr.node2avai_resources.keys() if node_id != affinity_node_to_exclude and self._find_gpu_placement_on_node(node_id, task_info)[0]]
                        if candidate_nodes:
                            best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                            _, indices = self._find_gpu_placement_on_node(best_node_id, task_info)
                            selected_node_id, gpu_indices_for_dispatch, placement_found = best_node_id, indices, True
                        if not dag_ctx and placement_found:
                            node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                            dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                            self._prepare_dag_context(task_info, dag_ctx)
        
        if placement_found:
            self.resource_mgr.reduce_node_resource(selected_node_id, gpu_indices_for_dispatch, task_info)
            dag_ctx = self.dag_ctx_mgr.get_context(run_id)
            if request_api_url: ray.get(dag_ctx.put.remote(f"{func_name}_request_api_url", request_api_url))
            print(f"🚀 Dispatching GPU task '{task_info['func_name']}' to node '{selected_node_id}' on GPUs {gpu_indices_for_dispatch}.")
            self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
            self.status_mgr.set_status(run_id, task_id, "running")
            self._dispatch_task(run_id, task_id, func_name, self.redis_ip, self.redis_port, "gpu", gpu_indices_for_dispatch)
            return True
        else: # No placement found, try eviction
            idle_candidates = [gpu for gpu in self.resource_mgr.find_all_gpus_by_state("OCCUPIED") if gpu.get("backend") == "vllm" and self.vllm_replica_load.get(gpu.get("request_api_url"), 0) == 0 and (time.time() - gpu.get('deployment_finish_time', 0) > self.VLLM_EVICTION_GRACE_PERIOD)]
            if idle_candidates:
                lru_runner_info= self.resource_mgr.find_lru_runner(idle_candidates)
                if lru_runner_info:
                    node_id_to_evict, indices_to_evict, model_to_evict = lru_runner_info
                    print(f"💡 Evicting idle model '{model_to_evict}' on Node {node_id_to_evict[:6]}/GPUs {indices_to_evict} to free up resources.")
                    model_mem = self.models_config.get(model_to_evict, {}).get("gpu_mem", 80000)
                    self.resource_mgr.add_node_resource(node_id_to_evict, indices_to_evict, {"mem": 0, "type": "gpu", "gpu_mem": model_mem})
                    self.backends["vllm"].undeploy(node_id_to_evict, indices_to_evict)
            return False # Task was not dispatched, needs re-queuing
    
    def _handle_cpu_task(self, priority, task_info) -> bool:
        """
        Contains the scheduling logic specifically for CPU tasks.
        Returns True if the task was placed and dispatched, False otherwise.
        """
        run_id, task_id, pre_scheduled_node_id = task_info["run_id"], task_info["task_id"], task_info.get("node_id")
        selected_node_id = None
        dag_ctx = self.dag_ctx_mgr.get_context(run_id)
        placement_found = False                    

        with self.resource_lock:
            if pre_scheduled_node_id and self._find_cpu_io_placement_on_node(pre_scheduled_node_id, task_info):
                placement_found, selected_node_id = True, pre_scheduled_node_id
                if not dag_ctx:
                    node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                    dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                    self._prepare_dag_context(task_info, dag_ctx)
            else:
                if dag_ctx:
                    affinity_node_id = self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                    if self._find_cpu_io_placement_on_node(affinity_node_id, task_info):
                        placement_found, selected_node_id = True, affinity_node_id
                if not placement_found:
                    affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                    candidate_nodes = [node_id for node_id in self.resource_mgr.node2avai_resources.keys() if node_id != affinity_node_to_exclude and self._find_cpu_io_placement_on_node(node_id, task_info)]
                    if candidate_nodes:
                        best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                        selected_node_id, placement_found = best_node_id, True
                    if not dag_ctx and placement_found:
                        node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                        dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                        self._prepare_dag_context(task_info, dag_ctx)
                            
        if placement_found:
            self.resource_mgr.reduce_node_resource(selected_node_id, None, task_info)
            print(f"🚀 Dispatching CPU task '{task_info['func_name']}' to node '{selected_node_id}'.")
            self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
            self.status_mgr.set_status(run_id, task_id, "running")
            self._dispatch_task(run_id, task_id, task_info['func_name'], self.redis_ip, self.redis_port, "cpu", None)
            return True
        return False

    def _handle_io_task(self, priority, task_info) -> bool:
        """
        Contains the scheduling logic specifically for IO tasks.
        Returns True if the task was placed and dispatched, False otherwise.
        """
        run_id, task_id, pre_scheduled_node_id = task_info["run_id"], task_info["task_id"], task_info.get("node_id")
        selected_node_id = None
        dag_ctx = self.dag_ctx_mgr.get_context(run_id)
        placement_found = False

        with self.resource_lock:
            if pre_scheduled_node_id and self._find_cpu_io_placement_on_node(pre_scheduled_node_id, task_info):
                placement_found, selected_node_id = True, pre_scheduled_node_id
                if not dag_ctx:
                    node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                    dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                    self._prepare_dag_context(task_info, dag_ctx)
            else:
                if dag_ctx:
                    affinity_node_id = self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                    if self._find_cpu_io_placement_on_node(affinity_node_id, task_info):
                        placement_found, selected_node_id = True, affinity_node_id
                if not placement_found:
                    affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                    candidate_nodes = [node_id for node_id in self.resource_mgr.node2avai_resources.keys() if node_id != affinity_node_to_exclude and self._find_cpu_io_placement_on_node(node_id, task_info)]
                    if candidate_nodes:
                        best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                        selected_node_id, placement_found = best_node_id, True
                    if not dag_ctx and placement_found:
                        node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                        dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                        self._prepare_dag_context(task_info, dag_ctx)

        if placement_found:
            self.resource_mgr.reduce_node_resource(selected_node_id, None, task_info)
            print(f"🚀 Dispatching IO task '{task_info['func_name']}' to node '{selected_node_id}'.")
            self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
            self.status_mgr.set_status(run_id, task_id, "running")
            self._dispatch_task(run_id, task_id, task_info['func_name'], self.redis_ip, self.redis_port, "io", None)
            return True
        return False

    def _dispatch_task(self, run_id: str, task_id: str, func_name:str, redis_host: str, redis_port: str, task_type:str, gpu_indices: Optional[List[int]]):
        try:
            redis_func_key= f"func:{task_id}"
            serialized_func= None
            for attempt in range(5):
                serialized_func = self.redis_client.get(redis_func_key)
                if serialized_func:break
                print(f"  -> [Attempt {attempt+1}/5] Function for task {task_id} not yet in Redis. Retrying in 0.1s...")
                time.sleep(0.05* attempt)
            if not serialized_func: raise ValueError(f"Function for task {task_id} not found in Redis.")
            self.redis_client.delete(redis_func_key)

            node_id = self.status_mgr.get_selected_node(run_id, task_id)
            ctx = self.dag_ctx_mgr.get_context(run_id)

            gpus_to_request = 0
            if gpu_indices and task_type == 'gpu':
                if len(gpu_indices) > 1:
                    gpus_to_request = len(gpu_indices)
                    print(f"  -> Dispatching multi-GPU task, requesting {gpus_to_request} full GPUs.")
                else:
                    total_gpu_mem = self.resource_mgr.node2avai_resources[node_id]["gpu_info"][gpu_indices[0]]["gpu_mem_total"]
                    task_info = self.status_mgr.get_task_info(run_id, task_id)
                    requested_mem = task_info.get("gpu_mem", 2048)
                    gpus_to_request = max(0.001, requested_mem/ total_gpu_mem)
                    print(f"  -> Dispatching single-GPU task, requesting fraction {gpus_to_request:.4f} of a GPU.")

            result_ref = remote_task_runner.options(
                num_gpus= gpus_to_request if gpu_indices and task_type == 'gpu' else 0,
                scheduling_strategy= ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(serialized_func, ctx, task_id, redis_host, redis_port, task_type, gpu_indices)
          
            with self.running_tasks_lock:
                task_info = self.status_mgr.task_info_map.get((run_id, task_id), {})
                task_to_run = {
                    "run_id": run_id, "dag_id": task_info.get('dag_id'), "task_id": task_id,
                    "func_name": func_name, "node_id": node_id, "ref": result_ref, "type": task_type,
                    "gpu_indices": gpu_indices,
                    "gpu_mem": task_info.get("gpu_mem", 0),
                    "cpu_num": task_info.get("cpu_num", 0), "mem": task_info.get("mem", 0),
                    "backend": task_info.get("backend"),
                    f"{func_name}_request_api_url": task_info.get(f"{func_name}_request_api_url")
                }
                self.running_tasks.append(task_to_run)
        except Exception as e:
            print(f"[DISPATCH FAILED] Task {task_id} failed to dispatch: {e}")
            self.status_mgr.set_status(run_id, task_id, "failed", err_msg= str(e))
            
    # --- Helper methods (no changes needed) ---
    def start_vllm_monitor_loop(self):
        def loop():
            while True:
                time.sleep(self.Check_VLLM_Interval)
                with self.resource_lock:
                    deploying_gpus = self.resource_mgr.find_all_gpus_by_state("DEPLOYING")
                    if not deploying_gpus: continue
                    print(f"🩺 [vLLM Monitor] Checking {len(deploying_gpus)} deploying GPU(s)...")
                    checked_runners = set() 
                    for gpu_info in deploying_gpus:
                        runner_key = self.resource_mgr.find_runner_key_for_gpu(gpu_info['node_id'], gpu_info['index'])
                        if runner_key and runner_key not in checked_runners:
                            node_id, gpu_indices_set = runner_key
                            gpu_indices = list(gpu_indices_set)
                            is_ready, api_url = self.backends["vllm"].is_server_ready(node_id, gpu_indices)
                            if is_ready:
                                print(f"✅ [vLLM Monitor] Model on {node_id[:6]}/GPUs {gpu_indices} is now ready!")
                                self.resource_mgr.update_gpu_state(node_id, gpu_indices, {"status": "OCCUPIED", "request_api_url": api_url, "runner_key": runner_key, "backend": "vllm", "deployment_finish_time": time.time()})
                            checked_runners.add(runner_key)
        threading.Thread(target=loop, daemon=True, name="VLLMMonitor").start()

    def _calculate_min_gpu_memory_threshold(self) -> int:
        min_mem = float('inf')
        gpus_found = False
        if not self.resource_mgr or not self.resource_mgr.node2avai_resources: return 24000
        for node_info in self.resource_mgr.node2avai_resources.values():
            for gpu_info in node_info.get("gpu_info", []):
                total_mem = gpu_info.get("gpu_mem_total")
                if total_mem is not None:
                    gpus_found, min_mem = True, min(min_mem, total_mem)
        return int(min_mem) if gpus_found else 24000

    def _prepare_dag_context(self, task_info: Dict, dag_ctx: ray.actor.ActorHandle) -> None:
        run_id, dag_id = task_info.get('run_id'), task_info.get('dag_id')
        context_data = {'run_id': run_id, "dag_id": dag_id, "question": task_info.get("question", ""), "answer": task_info.get("answer", "")}
        if task_info.get('supplementary_file_paths', {}):
            file_contents = {}
            for filename, file_path in task_info['supplementary_file_paths'].items():
                try:
                    with open(file_path, 'rb') as f: file_contents[filename] = f.read()
                except Exception as e: print(f"❌ [Error] Failed to read file {file_path}: {e}")
            context_data["supplementary_files"] = file_contents
        ray.get([dag_ctx.put.remote(k, v) for k, v in context_data.items() if v is not None])
        print(f"✅ Context for DAG {dag_id} (run_id, {run_id}) is ready.")

    def _is_vllm_replica_full(self, api_url: str) -> bool:
        return not api_url or self.vllm_replica_load.get(api_url, 0) >= self.VLLM_LOAD_THRESHOLD

    def _find_gpu_placement_on_node(self, node_id: str, task_info: dict) -> Tuple[bool, List[int]]:
        requested_mem, node_res = float(task_info.get("gpu_mem", 0)), self.resource_mgr.node2avai_resources.get(node_id)
        if not node_res: return False, []
        is_large_task, available_gpus = requested_mem > self.SINGLE_GPU_MEM_THRESHOLD, [gpu for gpu in node_res.get("gpu_info", []) if gpu.get("status", 'FREE') == 'FREE']
        if is_large_task:
            if sum(gpu['gpu_mem'] for gpu in available_gpus) < requested_mem: return False, []
            sorted_gpus, mem_sum, selected_indices = sorted(available_gpus, key=lambda g: g['gpu_mem'], reverse=True), 0, []
            for gpu in sorted_gpus:
                mem_sum, _ = mem_sum + gpu['gpu_mem'], selected_indices.append(gpu['index'])
                if mem_sum >= requested_mem: return True, selected_indices
            return False, []
        else:
            candidate_gpus = [gpu for gpu in available_gpus if gpu['gpu_mem'] >= requested_mem]
            if candidate_gpus: return True, [max(candidate_gpus, key=lambda g: g['gpu_mem'])['index']]
            return False, []
    
    def choice_cpu_gpu_queue_according_resource(self, remaining_gpu_mem, priority, task_info):
        task_info['gpu_mem'] = remaining_gpu_mem
        self.unified_task_queue.put(((float('-inf'),) + priority, task_info)) # Modified for unified queue
        print(f"✅ Activated task '{task_info['func_name']}'. Re-enqueued as normal GPU task.")
    
    def start_result_monitor(self):
        def monitor():
            redis_client = redis.Redis(host=self.redis_ip, port=self.redis_port, decode_responses=True)
            completion_queue_name = "task_completion_queue"
            while True:
                if not self.running_tasks:
                    time.sleep(0.05)
                    continue
                with self.running_tasks_lock:
                    ready_refs, _ = ray.wait([t["ref"] for t in self.running_tasks], timeout=0.05)
                    if not ready_refs: continue
                    
                    ref_to_task = {t["ref"]: t for t in self.running_tasks}
                    for ref in ready_refs:
                        task = ref_to_task.pop(ref)
                        result = ray.get(ref)
                        with self.resource_lock:
                            if task.get("backend") == "vllm":
                                api_url = task.get(f"{task['func_name']}_request_api_url")
                                if api_url in self.vllm_replica_load:
                                    self.vllm_replica_load[api_url] = max(0, self.vllm_replica_load[api_url] - 1)
                            self.resource_mgr.add_node_resource(task["node_id"], task.get("gpu_indices"), task)
                        
                        run_id, task_id, func_name = task["run_id"], task["task_id"], task["func_name"]
                        final_status = "failed" if result.get("status") == "failed" else "finished"
                        self.status_mgr.set_status(run_id, task_id, final_status, err_msg=result.get("err_msg"))
                        print(f"{'✅' if final_status == 'finished' else '❌'} [{final_status.upper()}] Task {task_id} ('{func_name}') completed.")
                        
                        notification = {"dag_id": task.get("dag_id"), "run_id": run_id, "task_id": task_id, "func_name": func_name, "status": final_status}
                        redis_client.lpush(completion_queue_name, json.dumps(notification))
                    self.running_tasks = list(ref_to_task.values())
        threading.Thread(target=monitor, daemon=True).start()

    def _dispatch_task(self, run_id: str, task_id: str, func_name:str, redis_host: str, redis_port: str, task_type:str, gpu_indices: Optional[List[int]]):
        try:
            redis_func_key = f"func:{task_id}"
            serialized_func = self.redis_client.get(redis_func_key)
            if not serialized_func: raise ValueError(f"Function for task {task_id} not found in Redis.")
            self.redis_client.delete(redis_func_key)

            node_id, ctx = self.status_mgr.get_selected_node(run_id, task_id), self.dag_ctx_mgr.get_context(run_id)
            gpus_to_request = 0
            if gpu_indices and task_type == 'gpu':
                if len(gpu_indices) > 1:
                    gpus_to_request = len(gpu_indices)
                else:
                    task_info = self.status_mgr.get_task_info(run_id, task_id)
                    total_gpu_mem = self.resource_mgr.node2avai_resources[node_id]["gpu_info"][gpu_indices[0]]["gpu_mem_total"]
                    gpus_to_request = max(0.001, task_info.get("gpu_mem", 2048) / total_gpu_mem)

            result_ref = remote_task_runner.options(
                num_gpus=gpus_to_request,
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(serialized_func, ctx, task_id, redis_host, redis_port, task_type, gpu_indices)
          
            with self.running_tasks_lock:
                task_info = self.status_mgr.task_info_map.get((run_id, task_id), {})
                task_to_run = {**task_info, "node_id": node_id, "ref": result_ref, "gpu_indices": gpu_indices}
                self.running_tasks.append(task_to_run)
        except Exception as e:
            print(f"[DISPATCH FAILED] Task {task_id} failed to dispatch: {e}")
            self.status_mgr.set_status(run_id, task_id, "failed", err_msg=str(e))
            
    def _find_cpu_io_placement_on_node(self, node_id: str, task_info: dict) -> bool:
        node_res = self.resource_mgr.node2avai_resources.get(node_id)
        if not node_res: return False
        task_type, mem_ok = task_info.get("type"), node_res.get("mem", 0) >= float(task_info.get("mem", 0))
        if task_type == "cpu": return node_res.get("cpu_num", 0) >= float(task_info.get("cpu_num", 1)) and mem_ok
        elif task_type == "io": return node_res.get("io_task", 0) > 0 and mem_ok
        return False