import os
import ray
import time
import queue
import torch
import random
import threading
import hashlib
import zipfile
import io     
import cloudpickle
from typing import Optional, Dict, List, Tuple, Any
import tracemalloc
import requests
from maze.utils.execution_backend import VLLMBackend, HuggingFaceBackend
import heapq
import gc
import re
from collections import deque, defaultdict
import json
import inspect
from maze.agent.config import config
from maze.core.worker.executor import remote_task_runner
import logging

class TaskScheduler:
    def __init__(self, resource_mgr: object, status_mgr: object, dag_ctx_mgr: object, completion_queue: queue.Queue, master_addr: str, proj_path: str= "", model_folder= "model_cache", models_config_path: str= "")-> None:
        self.master_addr = master_addr
        self.master_node_id = ray.get_runtime_context().get_node_id()
        self.resource_mgr = resource_mgr
        self.status_mgr = status_mgr
        self.dag_ctx_mgr = dag_ctx_mgr
        self.models_config = {}
        self.completion_queue = completion_queue
        self.models_config_path = os.path.join(proj_path, models_config_path)
        self.logger = logging.getLogger(__name__)
        try:
            with open(self.models_config_path, 'r') as f:
                self.models_config = json.load(f)
            self.logger.info(f"✅ TaskScheduler: Model configuration loaded from '{self.models_config_path}'.")
        except FileNotFoundError:
            self.logger.warning(f"⚠️ TaskScheduler: models_config_path '{self.models_config_path}' not found. No model configs loaded.")
        except json.JSONDecodeError:
            self.logger.error(f"❌ TaskScheduler: Error decoding JSON from '{self.models_config_path}'.")
        self.backends = {
            "vllm": VLLMBackend(
                resource_manager=self.resource_mgr,
                proj_path= proj_path,
                model_folder= model_folder,
                models_config_path= self.models_config_path
            ),
            "huggingface": HuggingFaceBackend()
        }
        self.vllm_waiting_queue= deque()
        self.task_queue_cpu = queue.PriorityQueue()
        self.task_queue_gpu = queue.PriorityQueue()
        self.task_queue_io = queue.PriorityQueue()
        self.vllm_replica_load = defaultdict(int)
        self.VLLM_RESERVED_FREE_GPUS = 1
        self.VLLM_LOAD_THRESHOLD = 5 # vLLM副本负载阈值
        self.VLLM_EVICTION_GRACE_PERIOD = 60
        self.Check_VLLM_Interval = 10 # seconds
        self.running_tasks = []
        self.running_tasks_lock = threading.Lock()
        self.resource_lock = threading.Lock()

        self.SINGLE_GPU_MEM_THRESHOLD = self._calculate_min_gpu_memory_threshold()
        self.logger.info(f"✅ Dynamically calculated single GPU memory threshold: {self.SINGLE_GPU_MEM_THRESHOLD} MiB")

        self.start_cpu_scheduler_loop()
        self.start_io_scheduler_loop()
        self.start_gpu_scheduler_loop()
        self.start_result_monitor()
        self.start_vllm_monitor_loop()
        self.bug_out_control= 0

    def start_vllm_monitor_loop(self):
        """
        一个后台线程，专门用于监控vLLM的部署状态。
        """
        def loop():
            while True:
                time.sleep(self.Check_VLLM_Interval) # 每10秒检查一次
                with self.resource_lock:
                    # 查找所有正在部署的GPU
                    deploying_gpus = self.resource_mgr.find_all_gpus_by_state("DEPLOYING")
                    if not deploying_gpus:
                        continue
                    
                    self.logger.debug(f"🩺 [vLLM Monitor] Checking {len(deploying_gpus)} deploying GPU(s)...")
                    checked_runners = set() 
                    for gpu_info in deploying_gpus:
                        # 从单个GPU信息反查它属于哪个runner（哪个多卡部署任务）
                        runner_key = self.resource_mgr.find_runner_key_for_gpu(gpu_info['node_id'], gpu_info['index']) # 根据GPU确定模型部署信息
                        if runner_key and runner_key not in checked_runners: # 确保还没有检查过
                            node_id, gpu_indices_set = runner_key # 
                            gpu_indices = list(gpu_indices_set)
                            is_ready, api_url = self.backends["vllm"].is_server_ready(node_id, gpu_indices) # 看看对应vllm部署情况如何
                            if is_ready:
                                self.logger.debug(f"✅ [vLLM Monitor] Model on {node_id[:6]}/GPUs {gpu_indices} is now ready!")
                                # 状态切换！
                                self.resource_mgr.update_gpu_state(node_id, gpu_indices, {
                                                                    "status": "OCCUPIED",
                                                                    "request_api_url": api_url,
                                                                    "runner_key": runner_key,
                                                                    "backend": "vllm",
                                                                    "deployment_finish_time": time.time()
                                                                })
                            checked_runners.add(runner_key)
        
        threading.Thread(target=loop, daemon=True, name="VLLMMonitor").start()

    def _calculate_min_gpu_memory_threshold(self) -> int:
        min_mem = float('inf')
        gpus_found = False
        if not self.resource_mgr or not self.resource_mgr.node2avai_resources:
            self.logger.warning("⚠️ Resource manager not ready, using default threshold 24000 MiB.")
            return 24000
        for node_info in self.resource_mgr.node2avai_resources.values():
            for gpu_info in node_info.get("gpu_info", []):
                total_mem = gpu_info.get("gpu_mem_total")
                if total_mem is not None:
                    gpus_found = True
                    if total_mem < min_mem: min_mem = total_mem
        if not gpus_found:
            self.logger.warning("⚠️ No GPUs found in the cluster, using default threshold 24000 MiB.")
            return 24000
        return int(min_mem)

    def _prepare_dag_context(self, task_info: Dict, dag_ctx: ray.actor.ActorHandle) -> None:
        run_id, dag_id = task_info.get('run_id'), task_info.get('dag_id')
        self.logger.debug(f"💾 Preparing context for DAG {run_id}...")
        context_data = {'run_id': run_id, "dag_id": dag_id, "question": task_info.get("question", ""), "answer": task_info.get("answer", "")}
        supplementary_file_paths = task_info.get('supplementary_file_paths', {})
        if supplementary_file_paths:
            file_contents = {}
            for filename, file_path in supplementary_file_paths.items():
                try:
                    with open(file_path, 'rb') as f: content = f.read()
                    file_contents[filename] = content
                except Exception as e: self.logger.error(f"❌ [Error] Failed to read file {file_path}: {e}")
            context_data["supplementary_files"] = file_contents
        futures = [dag_ctx.put.remote(k, v) for k, v in context_data.items() if v is not None]
        self.logger.debug(f"✅ Context for DAG {dag_id} (run_id, {run_id}) is ready.")

    def _is_vllm_replica_full(self, api_url: str) -> bool:
        """通过vLLM的API检查其请求队列是否为空。"""
        if not api_url: return True # 如果没有URL，则认为不可用
        if self.vllm_replica_load[api_url]>= self.VLLM_LOAD_THRESHOLD:
            return True

    def _find_gpu_placement_on_node(self, node_id: str, task_info: dict) -> Tuple[bool, List[int]]:
        requested_mem = float(task_info.get("gpu_mem", 0))
        node_res = self.resource_mgr.node2avai_resources.get(node_id)
        if not node_res: return False, []

        is_large_task = requested_mem > self.SINGLE_GPU_MEM_THRESHOLD
        available_gpus = [gpu for gpu in node_res.get("gpu_info", []) if gpu.get("status", 'FREE') == 'FREE']
        
        if is_large_task:
            total_available_mem = sum(gpu['gpu_mem'] for gpu in available_gpus)
            if total_available_mem < requested_mem: return False, []
            
            sorted_gpus = sorted(available_gpus, key=lambda g: g['gpu_mem'], reverse=True)
            mem_sum = 0
            selected_indices = []
            for gpu in sorted_gpus:
                mem_sum += gpu['gpu_mem']
                selected_indices.append(gpu['index'])
                if mem_sum >= requested_mem: return True, selected_indices
            return False, []
        else:
            candidate_gpus = [gpu for gpu in available_gpus if gpu['gpu_mem'] >= requested_mem]
            if candidate_gpus:
                best_gpu = max(candidate_gpus, key=lambda g: g['gpu_mem'])
                return True, [best_gpu['index']]
            else:
                return False, []

    def choice_cpu_gpu_queue_according_resource(self, remaining_gpu_mem, priority, task_info):
        # if remaining_gpu_mem <= 0:
            # 仅走API，设置为CPU任务
            # task_info['type'] = 'cpu'
            # task_info.pop('gpu_mem', None)
            # self.task_queue_cpu.put((priority, task_info))
            # print(f"✅ Activated and Demoted task '{task_info['func_name']}' to CPU queue.")
        # else:
        # 仍然是GPU任务，但需求减少
        task_info['gpu_mem'] = remaining_gpu_mem
        self.task_queue_gpu.put(((float('-inf'),) + priority, task_info))
        self.logger.debug(f"✅ Activated task '{task_info['func_name']}'. Re-enqueued as normal GPU task.")

    def start_gpu_scheduler_loop(self):
        def loop():
            while True:
                if self.vllm_waiting_queue:
                    _, task_info_peek = self.vllm_waiting_queue[0]
                    model_name_peek = task_info_peek.get("model_name")
                    ready_replicas = self.resource_mgr.find_gpus_by_model(
                        model_name_peek, "vllm", status="OCCUPIED"
                    )
                    available_replicas = [r for r in ready_replicas if not self._is_vllm_replica_full(r.get("request_api_url"))] # 筛选出所有“可用”（负载未满）的副本
                    if available_replicas: # 如果有可用模型副本
                        available_replica = min(
                            available_replicas,
                            key=lambda r: self.vllm_replica_load.get(r.get("request_api_url"), 0)
                        )
                        priority, task_info = self.vllm_waiting_queue.popleft() # 正式弹出任务
                        target_api_url = available_replica.get('request_api_url') # 获得请求api_url
                        task_info[f'{task_info["func_name"]}_request_api_url'] = target_api_url # 设置url
                        # 创建一个极高优先级，确保它被立刻处理
                        self.vllm_replica_load[target_api_url]+= 1 # 模型负载+ 1
                        model_mem = self.models_config.get(task_info.get("model_name"), {}).get("gpu_mem", 80000)
                        self.choice_cpu_gpu_queue_according_resource(task_info.get('gpu_mem', 0)- model_mem, priority, task_info)
                        self.logger.debug(f"✅ Activated waiting task '{task_info['func_name']}' for model '{model_name_peek}'.")
                        # 激活一个后，立即进入主循环处理，避免一次性激活太多导致队列拥堵
                        continue
                # 如果没有等待队列，继续处理主任务队列
                if self.task_queue_gpu.empty(): 
                    time.sleep(0.05)
                    continue

                priority, task_info = self.task_queue_gpu.get()
                run_id, task_id, pre_scheduled_node_id, func_name, model_name, backend_type= task_info["run_id"], task_info["task_id"], task_info.get("node_id"), task_info.get("func_name"), task_info.get("model_name"), task_info.get("backend", "huggingface")
                request_api_url= task_info.get(f"{func_name}_request_api_url", None)
                backend = self.backends[backend_type]
                gpu_indices_for_dispatch = []
                dag_ctx= self.dag_ctx_mgr.get_context(run_id)
                placement_found, selected_node_id, target_api_url= False, None, None

                with self.resource_lock:
                    # 场景A: vLLM任务，有特殊的查找逻辑
                    if backend_type == 'vllm' and not request_api_url: # vllm任务且没有部署模型与之适配，当前版本每次部署一台，效率不高，可以通过统计任务数量来决定是否多部署几个，从而增加系统吞吐量
                        replicas = self.resource_mgr.find_gpus_by_model(model_name, "vllm", "OCCUPIED") # 找副本模型
                        ready_replicas = [r for r in replicas if not self._is_vllm_replica_full(r.get("request_api_url"))]
                        model_mem = self.models_config.get(model_name, {}).get("gpu_mem", 80000) # 获取模型对应的GPU显存开销
                        if ready_replicas: # 有可用模型副本且不忙
                            available_replica = min( # 负载均衡
                                ready_replicas,
                                key=lambda r: self.vllm_replica_load.get(r.get("request_api_url"), 0)
                            )
                            target_api_url= available_replica['request_api_url'] # 
                            task_info[f"{func_name}_request_api_url"]= target_api_url # 任务与已有模型进行一个绑定
                            self.vllm_replica_load[target_api_url]+= 1 # 模型负载+ 1
                            self.choice_cpu_gpu_queue_according_resource(task_info.get('gpu_mem', 0) - model_mem, priority, task_info) # 放入cpu还是gpu队列
                            self.logger.debug(f"✅ Found existing replica for '{model_name}' at {target_api_url}.")
                            continue
                        else: # 没找到可用副本，需要找个空闲GPU来部署
                            should_deploy_model= False
                            num_requests_waiting= 1+ sum(1 for _, t_info in self.vllm_waiting_queue if t_info.get("model_name")== model_name)
                            ready_replicas_count= len(replicas) # 已部署的副本模型
                            deploying_replicas_count= len(set(gpu.get("runner_key") for gpu in self.resource_mgr.find_all_gpus_by_state("DEPLOYING") if gpu.get("model_name") == model_name)) # 正在部署的模型副本
                            replicas_count= ready_replicas_count+ deploying_replicas_count
                            should_deploy_model= True if replicas_count== 0 or num_requests_waiting* 1.0/ replicas_count> self.VLLM_LOAD_THRESHOLD else False
                            if should_deploy_model:
                                # --- 在部署前检查资源保留池 ---
                                num_free_gpus = len(self.resource_mgr.find_all_gpus_by_state("FREE"))
                                if num_free_gpus> self.VLLM_RESERVED_FREE_GPUS:
                                    deployment_node_id, deployment_indices, runner_key= None, None, None
                                    for node_id_search in self.resource_mgr.node2avai_resources.keys(): # 找可部署的节点
                                        can_deploy, indices = self._find_gpu_placement_on_node(node_id_search, {"gpu_mem": model_mem})
                                        if can_deploy:
                                            deployment_node_id, deployment_indices = node_id_search, indices
                                            runner_key= (deployment_node_id, frozenset(deployment_indices)) # 部署组
                                            break
                                    if deployment_node_id: # 有可部署的节点
                                        self.logger.debug(f"💡 Triggering deployment for '{model_name}' on Node {deployment_node_id[:6]}, GPU(s) {deployment_indices}.")
                                        backend.deploy(deployment_node_id, deployment_indices, model_name)
                                        self.resource_mgr.update_gpu_state(deployment_node_id, deployment_indices, {"status": "DEPLOYING", "model_name": model_name, "backend": "vllm", "runner_key": runner_key}) # 更新GPU状态
                                        self.resource_mgr.reduce_node_resource(deployment_node_id, deployment_indices, {"mem": 0, "type": "gpu", "gpu_mem": model_mem}) # 减扣资源
                                        self.vllm_waiting_queue.append((priority, task_info)) # 放入等待队列中
                                        continue
                            else:
                                self.vllm_waiting_queue.append((priority, task_info))
                                continue
                    else:
                        if pre_scheduled_node_id:
                            can_run, indices = self._find_gpu_placement_on_node(pre_scheduled_node_id, task_info)
                            if can_run:
                                placement_found= True
                                self.logger.debug(f"✅ Accepting pre-scheduled placement on node '{pre_scheduled_node_id}' for GPUs {indices}.")
                                selected_node_id = pre_scheduled_node_id
                                gpu_indices_for_dispatch = indices
                                if not dag_ctx:
                                    node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                    dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                                    self._prepare_dag_context(task_info, dag_ctx)
                        else:
                            if dag_ctx:
                                affinity_node_id= self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                                # print(f"🔍 Checking data-affinity node '{affinity_node_id}'...")
                                can_run, indices= self._find_gpu_placement_on_node(affinity_node_id, task_info)
                                if can_run:
                                    placement_found= True
                                    self.logger.debug(f"  -> ✅ Affinity node has resources. Placing on GPUs {indices}.")
                                    selected_node_id = affinity_node_id
                                    gpu_indices_for_dispatch = indices

                            if not placement_found:
                                candidate_nodes= []
                                affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                                for node_id_search in self.resource_mgr.node2avai_resources.keys():
                                    if node_id_search== affinity_node_to_exclude:
                                        continue
                                    can_run, indices = self._find_gpu_placement_on_node(node_id_search, task_info)
                                    # print(f"  -> 🌀 Affinity_node_to_exclude: {affinity_node_to_exclude}, node resource_mgr.node2avai_resources: {self.resource_mgr.node2avai_resources}")
                                    # print(f"  -> 🆒 Task id: {task_id}, checking node '{node_id_search[:6]}' for GPU placement: {can_run}, indices: {indices}.")
                                    if can_run:
                                        candidate_nodes.append(node_id_search)
                                if candidate_nodes:
                                    best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                                    # print(f"  -> 😄Candidate nodes with resources: {candidate_nodes}. Choosing least loaded: '{best_node_id}'.")
                                    _, indices = self._find_gpu_placement_on_node(best_node_id, task_info)
                                    selected_node_id = best_node_id
                                    gpu_indices_for_dispatch = indices
                                    placement_found = True

                                if not dag_ctx and placement_found:
                                    node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                    dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                                    self._prepare_dag_context(task_info, dag_ctx)
                
                if placement_found:
                    self.resource_mgr.reduce_node_resource(selected_node_id, gpu_indices_for_dispatch, task_info)
                    self.logger.debug(f"[{time.strftime('%H:%M:%S')}] 🧠 GPU Scheduler Loop: Dequeued Task")
                    self.logger.debug(f"  -> Task to Schedule: '{func_name}' (run_id: {run_id}) Priority: {priority}")
                    dag_ctx = self.dag_ctx_mgr.get_context(run_id)
                    if request_api_url:ray.get(dag_ctx.put.remote(f"{func_name}_request_api_url", request_api_url)) # 设置API URL到DAG上下文
                    self.logger.info(f"🚀 Dispatching GPU task '{task_info['func_name']}' to node '{selected_node_id}' on GPUs {gpu_indices_for_dispatch}.")
                    self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
                    self.status_mgr.set_status(run_id, task_id, "running")
                    self._dispatch_task(run_id, task_id, task_info['func_name'], "gpu", gpu_indices_for_dispatch)
                else:
                    # 优先级4: 驱逐
                    idle_candidates = [gpu for gpu in self.resource_mgr.find_all_gpus_by_state("OCCUPIED") 
                                       if gpu.get("backend") == "vllm" 
                                       and self.vllm_replica_load.get(gpu.get("request_api_url"), 0) == 0
                                       and (time.time() - gpu.get('deployment_finish_time', 0) > self.VLLM_EVICTION_GRACE_PERIOD)] # 刚部署好的模型给一定的保护期
                    if idle_candidates:
                        lru_runner_info= self.resource_mgr.find_lru_runner(idle_candidates)
                        if lru_runner_info:
                            node_id_to_evict, indices_to_evict, model_to_evict = lru_runner_info
                            self.logger.info(f"💡 Evicting idle model '{model_to_evict}' on Node {node_id_to_evict[:6]}/GPUs {indices_to_evict} to free up resources.")
                            # 返还整个模型占用的资源
                            model_mem = self.models_config.get(model_to_evict, {}).get("gpu_mem", 80000)
                            self.resource_mgr.add_node_resource(node_id_to_evict, indices_to_evict, 
                                                            {"mem": 0, "type": "gpu", "gpu_mem": model_mem})
                            # 命令后台卸载整个Runner
                            self.backends["vllm"].undeploy(node_id_to_evict, indices_to_evict)
                    self.task_queue_gpu.put((priority, task_info))
                time.sleep(0.05)
        threading.Thread(target=loop, daemon=True).start()

    def _find_cpu_io_placement_on_node(self, node_id: str, task_info: dict) -> bool:
        node_res = self.resource_mgr.node2avai_resources.get(node_id)
        if not node_res: return False
        task_type = task_info.get("type")
        if task_type == "cpu":
            cpu_ok = node_res.get("cpu_num", 0) >= float(task_info.get("cpu_num", 1))
            mem_ok = node_res.get("mem", 0) >= float(task_info.get("mem", 0))
            return cpu_ok and mem_ok
        elif task_type == "io":
            io_ok = node_res.get("io_task", 0) > 0
            mem_ok = node_res.get("mem", 0) >= float(task_info.get("mem", 0))
            return io_ok and mem_ok
        return False

    def start_cpu_scheduler_loop(self):
        def loop():
            while True:
                if self.task_queue_cpu.empty(): 
                    time.sleep(0.05)
                    continue

                priority, task_info= self.task_queue_cpu.get()
                dag_id, run_id, task_id, pre_scheduled_node_id, arrival_time = task_info["dag_id"], task_info["run_id"], task_info["task_id"], task_info.get("node_id"), task_info.get("arrival_time")
                selected_node_id = None
                dag_ctx = self.dag_ctx_mgr.get_context(run_id)
                placement_found = False                    

                with self.resource_lock:
                    # 优先级1：预调度节点
                    if pre_scheduled_node_id:
                        can_run = self._find_cpu_io_placement_on_node(pre_scheduled_node_id, task_info)
                        if can_run:
                            placement_found = True
                            self.logger.debug(f"✅ Accepting pre-scheduled placement on node '{pre_scheduled_node_id}' for CPU task.")
                            selected_node_id = pre_scheduled_node_id
                            # 如果是新DAG，需要确保上下文被创建
                            if not dag_ctx:
                                node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                                self._prepare_dag_context(task_info, dag_ctx)
                    # 优先级2：数据亲和节点
                    else:
                        if dag_ctx:
                            affinity_node_id = self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                            can_run = self._find_cpu_io_placement_on_node(affinity_node_id, task_info)
                            if can_run:
                                placement_found = True
                                self.logger.debug(f"  -> ✅ Affinity node has resources.")
                                selected_node_id = affinity_node_id
                        
                        # 优先级3：全局搜索
                        if not placement_found:
                            # print(f"🌀 No pre-schedule or affinity placement found for CPU task. Starting global search...")
                            # 避免重复检查亲和节点
                            affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                            candidate_nodes = []
                            for node_id_search in self.resource_mgr.node2avai_resources.keys():
                                if node_id_search == affinity_node_to_exclude:
                                    continue
                                if self._find_cpu_io_placement_on_node(node_id_search, task_info):
                                    candidate_nodes.append(node_id_search)

                            if candidate_nodes:
                                best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                                self.logger.debug(f"  -> Candidate nodes: {candidate_nodes}. Choosing least loaded: '{best_node_id}'.")
                                selected_node_id = best_node_id
                                placement_found = True
                        
                            # 如果是全局搜索找到的，且是新DAG，则创建上下文
                            if not dag_ctx and placement_found:
                                node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                                self._prepare_dag_context(task_info, dag_ctx)
                                    
                # 统一的任务派发
                if placement_found:
                    self.resource_mgr.reduce_node_resource(selected_node_id, None, task_info)
                    self.logger.info(f"🚀 Dispatching CPU task '{task_info['func_name']}' to node '{selected_node_id}'.")
                    self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
                    self.status_mgr.set_status(run_id, task_id, "running")
                    self._dispatch_task(run_id, task_id, task_info['func_name'], "cpu", None)
                else:
                    self.task_queue_cpu.put((priority, task_info))
                time.sleep(0.05)
        threading.Thread(target=loop, daemon=True).start()

    def start_io_scheduler_loop(self):
        def loop():
            while True:
                if self.task_queue_io.empty(): 
                    time.sleep(0.05)
                    continue
                priority, task_info = self.task_queue_io.get()
                dag_id, run_id, task_id, pre_scheduled_node_id, arrival_time = task_info["dag_id"], task_info["run_id"], task_info["task_id"], task_info.get("node_id"), task_info.get("arrival_time")
                selected_node_id = None
                dag_ctx = self.dag_ctx_mgr.get_context(run_id)
                placement_found = False

                with self.resource_lock:
                    # 优先级1：预调度节点
                    if pre_scheduled_node_id:
                        can_run = self._find_cpu_io_placement_on_node(pre_scheduled_node_id, task_info)
                        if can_run:
                            placement_found = True
                            self.logger.debug(f"✅ Accepting pre-scheduled placement on node '{pre_scheduled_node_id}' for IO task.")
                            selected_node_id = pre_scheduled_node_id
                            if not dag_ctx:
                                node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                dag_ctx = self.dag_ctx_mgr.create_context(pre_scheduled_node_id, node_ip, task_info)
                                self._prepare_dag_context(task_info, dag_ctx)
                    # 优先级2：数据亲和节点
                    else:
                        if dag_ctx:
                            affinity_node_id = self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                            # print(f"🔍 Checking data-affinity node '{affinity_node_id}' for IO task...")
                            can_run = self._find_cpu_io_placement_on_node(affinity_node_id, task_info)
                            if can_run:
                                placement_found = True
                                self.logger.debug(f"  -> ✅ Affinity node has resources.")
                                selected_node_id = affinity_node_id
                        
                        # 优先级3：全局搜索
                        if not placement_found:
                            # print(f"🌀 No pre-schedule or affinity placement found for IO task. Starting global search...")
                            affinity_node_to_exclude = dag_ctx and self.dag_ctx_mgr.ctx2id.get(dag_ctx)
                            candidate_nodes = []
                            for node_id_search in self.resource_mgr.node2avai_resources.keys():
                                if node_id_search == affinity_node_to_exclude:
                                    continue
                                if self._find_cpu_io_placement_on_node(node_id_search, task_info):
                                    candidate_nodes.append(node_id_search)
                            if candidate_nodes:
                                best_node_id = self.dag_ctx_mgr.get_least_loaded_node(candidate_nodes)
                                self.logger.debug(f"  -> Candidate nodes: {candidate_nodes}. Choosing least loaded: '{best_node_id}'.")
                                selected_node_id = best_node_id
                                placement_found = True
                                
                            if not dag_ctx and placement_found:
                                node_ip = self.resource_mgr.id2ip.get(selected_node_id, "")
                                dag_ctx = self.dag_ctx_mgr.create_context(selected_node_id, node_ip, task_info)
                                self._prepare_dag_context(task_info, dag_ctx)

                if placement_found:
                    self.resource_mgr.reduce_node_resource(selected_node_id, None, task_info)
                    self.logger.info(f"🚀 Dispatching IO task '{task_info['func_name']}' to node '{selected_node_id}'.")
                    self.status_mgr.set_selected_node(run_id, task_id, selected_node_id)
                    self.status_mgr.set_status(run_id, task_id, "running")
                    self._dispatch_task(run_id, task_id, task_info['func_name'], "io", None)
                else:
                    self.task_queue_io.put((priority, task_info))
                time.sleep(0.05)
        threading.Thread(target=loop, daemon=True).start()

    def start_result_monitor(self):
        def monitor():
            while True:
                if not self.running_tasks:
                    time.sleep(0.05)
                    continue
                with self.running_tasks_lock:
                    ready_refs = [task["ref"] for task in self.running_tasks]
                    ready, _ = ray.wait(ready_refs, num_returns=len(ready_refs), timeout= 0.05)
                    if not ready: continue
                    ref_to_task = {task["ref"]: task for task in self.running_tasks}
                    for ref in ready:
                        task = ref_to_task.pop(ref)
                        run_id, task_id, func_name = task["run_id"], task["task_id"], task["func_name"]
                        result = ray.get(ref)
                        receive_task_result_time= time.time()
                        with self.resource_lock:
                            if task.get("backend") == "vllm":
                                api_url = task.get(f"{func_name}_request_api_url")
                                if api_url and api_url in self.vllm_replica_load:
                                    self.vllm_replica_load[api_url] = max(0, self.vllm_replica_load[api_url] - 1)
                                    self.logger.debug(f"  -> 📉 Decremented load for {api_url}. New load: {self.vllm_replica_load[api_url]}")
                            self.resource_mgr.add_node_resource(task["node_id"], task.get("gpu_indices"), task)

                        final_status = "finished" if result.get("status") == "finished" else "failed"
                        if final_status == "finished":
                            self.status_mgr.set_status(run_id, task_id, "finished")
                            self.logger.info(f"✅ [SUCCESS] Task {task_id} ('{func_name}') completed.")
                        else:
                            self.status_mgr.set_status(run_id, task_id, "failed", err_msg= result.get("err_msg"))
                            self.logger.error(f"❌ [FAILED] Task {task_id} ('{func_name}') failed: {result.get('err_msg')}")
                        
                        # **通知逻辑**
                        notification = {
                            "dag_id": task.get("dag_id"), "run_id": run_id, "task_id": task_id,
                            "func_name": func_name, "status": final_status, "dispatch_task_time": task["dispatch_task_time"],
                            "receive_task_result_time": receive_task_result_time, 
                            "worker_start_exec_time": result.get("worker_start_exec_time"),
                            "worker_end_time": result.get("end_time") # <--- 新增此行
                        }
                        try:
                            # 放入内存队列，而不是 Redis
                            self.completion_queue.put(notification)
                            self.logger.debug(f"  -> 📬 Notified completion of '{func_name}' via in-memory queue.")
                        except Exception as e:
                            self.logger.error(f"❌ [Error] Failed to send completion notification for '{func_name}': {e}")
                    self.running_tasks = list(ref_to_task.values())
                time.sleep(0.05)
        threading.Thread(target=monitor, daemon=True).start()

    def submit(self, task_info: Dict):
        """接收来自 daps 的任务字典，并放入相应的优先级队列。"""
        self.status_mgr.add_task(task_info)
        task_type = task_info.get("type", "cpu")
        priority = task_info.get("priority", time.time())
        
        if isinstance(priority, list):
            priority = tuple(priority)
        elif isinstance(priority, (int, float)):
            priority = (priority,)
        
        self.logger.info(f"💮 TaskScheduler received task: {task_info['task_id']}")
        item_to_queue = (priority, task_info)
        if task_type == "gpu": self.task_queue_gpu.put(item_to_queue)
        elif task_type == "io": self.task_queue_io.put(item_to_queue)
        else: self.task_queue_cpu.put(item_to_queue)

    def _dispatch_task(self, run_id: str, task_id: str, func_name:str, task_type:str, gpu_indices: Optional[List[int]]):
        try:
            task_info = self.status_mgr.get_task_info(run_id, task_id)
            
            # 1. 从 task_info 中获取所有必需的信息
            serialized_func = task_info.get("serialized_func")
            if not serialized_func:
                raise ValueError(f"Function for task {task_id} not found in task_info.")

            server_root_path = task_info.get("server_root_path")
            if not server_root_path:
                raise ValueError(f"`server_root_path` not found in task_info for task {task_id}.")

            task_inputs = task_info.get("inputs", {})
            output_parameters = task_info.get("output_parameters", {})

            node_id = self.status_mgr.get_selected_node(run_id, task_id)
            ctx = self.dag_ctx_mgr.get_context(run_id)
            
            # 2. 计算GPU资源 (此部分逻辑不变)
            gpus_to_request = 0
            if gpu_indices and task_type == 'gpu':
                if len(gpu_indices) > 1:
                    gpus_to_request = len(gpu_indices)
                else:
                    total_gpu_mem = self.resource_mgr.node2avai_resources[node_id]["gpu_info"][gpu_indices[0]]["gpu_mem_total"]
                    requested_mem = task_info.get("gpu_mem", 2048)
                    gpus_to_request = max(0.001, requested_mem / total_gpu_mem)
            
            # 3. 派发远程任务，并传递【所有】必需的参数
            dispatch_task_time = time.time()
            result_ref = remote_task_runner.options(
                num_gpus=gpus_to_request,
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(
                serialized_func,
                task_id,
                run_id, 
                self.master_addr, 
                task_inputs,         # <-- 已补全
                output_parameters,   # <-- 已补全
                task_type, 
                gpu_indices,
                ctx
            )
          
            with self.running_tasks_lock:
                # running_tasks 的记录逻辑保持不变
                task_to_run = {
                    "run_id": run_id, "dag_id": task_info.get('dag_id'), "task_id": task_id,
                    "func_name": func_name, "node_id": node_id, "ref": result_ref, "type": task_type,
                    "gpu_indices": gpu_indices,
                    "gpu_mem": task_info.get("gpu_mem", 0),
                    "cpu_num": task_info.get("cpu_num", 0), "mem": task_info.get("mem", 0),
                    "backend": task_info.get("backend"),
                    f"{func_name}_request_api_url": task_info.get(f"{func_name}_request_api_url"),
                    "dispatch_task_time": dispatch_task_time
                }
                self.running_tasks.append(task_to_run)
        except Exception as e:
            self.logger.error(f"[DISPATCH FAILED] Task {task_id} failed to dispatch: {e}")
            self.status_mgr.set_status(run_id, task_id, "failed", err_msg= str(e))