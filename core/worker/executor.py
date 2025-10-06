import os
import ray
import time
import queue
import torch
import random
import threading
import hashlib  # 新增
import zipfile  # 新增
import io       # 新增
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
from maze.utils.log_config import setup_logging # <--- 1. 导入我们的新函数
import logging # <--- 导入logging

def _calculate_local_sha256(file_path):
    """一个计算文件SHA256哈希值的辅助函数"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def write_log(dag_id,run_id,task_id,dag_func_file,func_name,status):
    log_dir = './log'
    log_file = os.path.join(log_dir, 'log.txt')
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as file:
        file.write(f"{dag_id},{run_id},{task_id},{dag_func_file},{func_name},{status},{time.time()}\n")

@ray.remote(num_cpus=0, max_calls=1)
def remote_task_runner(
    serialized_func: bytes,
    task_id: str,
    run_id: str,
    master_addr: str,
    task_inputs: Dict,
    output_parameters: Dict,
    task_type:str,
    gpu_indices: Optional[List[int]],
    ctx_actor: object
) -> Dict[str, Any]:
    """
    (V8 - 最终无Redis版)
    - 直接从参数接收函数字节码。
    - 直接返回元数据字典，不再写入Redis。
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    worker_start_exec_time = time.time()
    try:
        # --- 阶段1：文件同步拉取 (Pull) ---
        logger.debug(f"🔄 [Sync] Task {task_id} on worker starting file synchronization for run '{run_id}'...")
        original_dir = os.getcwd()
        data_dir = os.path.join(original_dir, "taskspace")
        os.makedirs(data_dir, exist_ok=True)

        master_hashes_url = f"http://{master_addr}/files/hashes/{run_id}"
        
        response = requests.get(master_hashes_url)
        response.raise_for_status()
        master_hashes = response.json().get('hashes', {})
        logger.debug(f"  - [Sync] Fetched {len(master_hashes)} official file hashes from master.")
        
        local_hashes = {}
        for root, _, files in os.walk(data_dir):
            for name in files:
                file_abs_path = os.path.join(root, name)
                file_rel_path = os.path.relpath(file_abs_path, data_dir)
                local_hashes[file_rel_path] = _calculate_local_sha256(file_abs_path)

        files_to_update = [
            rel_path for rel_path, master_hash in master_hashes.items()
            if rel_path not in local_hashes or local_hashes[rel_path] != master_hash
        ]
        
        if files_to_update:
            logger.debug(f"  - [Sync] Found {len(files_to_update)} files to update: {files_to_update}")
            master_download_url = f"http://{master_addr}/files/download/{run_id}"
            response = requests.post(master_download_url, json={"files": files_to_update})
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(data_dir)
            logger.debug(f"  - [Sync] Worker directory updated successfully.")
        else:
            logger.debug("  - [Sync] All local files are up-to-date.")

        # --- 阶段2：参数解析 ---
        logger.debug(f"🔎 [Resolve] Task {task_id} starting robust parameter resolution...")
        func = cloudpickle.loads(serialized_func)

        if task_type == "gpu" and gpu_indices:
            visible_devices = ",".join(map(str, gpu_indices))
            logger.debug(f"  -> [EnvSetup] Setting CUDA_VISIBLE_DEVICES='{visible_devices}' for task {task_id}.")
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        def _resolve_single_source_remote(source: str, context_actor: object) -> Any:
            index = None
            base_source = source
            index_match = re.search(r'\[(\d+)\]$', source)
            if index_match:
                index = int(index_match.group(1))
                base_source = source[:index_match.start()]
            
            parts = base_source.split('.')
            if len(parts) < 3:
                raise ValueError(f"无效的输入源格式: '{source}'。必须使用 'task_id.output.key' 的格式。")
                
            upstream_task_id, _, output_key = parts[0], parts[1], ".".join(parts[2:])
            
            # --- 核心修正：模拟单机版的两步查找 ---
            # 1. 先从 DAGContext 获取上游任务返回的【整个结果对象】
            upstream_result_obj = ray.get(context_actor.get.remote(f"{upstream_task_id}.output"))
            
            # 2. 然后再从这个对象中获取具体的字段
            if not isinstance(upstream_result_obj, dict):
                 raise TypeError(f"任务 '{upstream_task_id}' 的输出不是字典，无法访问。")
            if output_key not in upstream_result_obj:
                raise KeyError(f"在任务 '{upstream_task_id}' 的输出中找不到键: '{output_key}'。")
            
            resolved_object = upstream_result_obj[output_key]
            # --- 修正结束 ---

            if index is not None:
                if isinstance(resolved_object, (list, tuple)):
                    if index >= len(resolved_object):
                        raise IndexError(f"索引 {index} 超出范围。源: '{source}' 的列表长度为 {len(resolved_object)}。")
                    return resolved_object[index]
                else:
                    raise TypeError(f"尝试对非序列类型进行索引访问。源: '{source}'")
            return resolved_object

        resolved_kwargs = {}
        # 遍历函数签名中的每一个参数
        for param in inspect.signature(func).parameters.values():
            param_name = param.name
            # 优先级 1: 检查用户在 add_task 时提供的 'task_inputs' 字典
            if param_name in task_inputs:
                source = task_inputs[param_name]
                if isinstance(source, list):
                    resolved_kwargs[param_name] = [
                        _resolve_single_source_remote(item, ctx_actor) if isinstance(item, str) and ".output" in item else item
                        for item in source
                    ]
                elif isinstance(source, str) and ".output" in source:
                    resolved_kwargs[param_name] = _resolve_single_source_remote(source, ctx_actor)
                else:
                    resolved_kwargs[param_name] = source
            # 如果在 inputs 中没找到，则进入 优先级2 和 3
            else:
                found_in_config = False    
                # 优先级 2: 从 DAGContext 中按键名匹配 (Context中已包含paths和online_apis)
                try:
                    # 直接尝试从 context actor 获取同名参数
                    value = ray.get(ctx_actor.get.remote(param_name))
                    resolved_kwargs[param_name] = value                    
                    found_in_config = True
                except KeyError:
                    pass
                # 优先级 3: 如果配置中也没找到，则使用函数定义的默认值
                if not found_in_config and param.default is not inspect.Parameter.empty:
                    resolved_kwargs[param_name] = param.default
        logger.debug("  - ✅ [Resolve] All parameters resolved successfully.")
        
        # --- 阶段3：执行用户函数 ---
        try:
            # 关键：切换到数据目录执行用户函数
            os.chdir(data_dir)
            logger.debug(f"🚀 [Execute] Task {task_id} starting execution in isolated data directory '{data_dir}'...")
            user_function_result = func(**resolved_kwargs)
            logger.debug(f"  - ✅ [Execute] Task {task_id} finished execution.")
        finally:
            # 关键：无论成功或失败，都切回原始目录
            os.chdir(original_dir)

        # --- 核心修正：实现与 _run_local 完全一致的智能包装逻辑 ---
        final_output_obj = None
        if user_function_result is not None:
            output_dict = {}
            # 从传递过来的元数据中解析输出字段名
            keys = list(output_parameters.get('properties', {}).keys())
            
            # 严格遵守“必须且只能有一个输出”的规则
            if len(keys) != 1:
                raise ValueError(
                    f"Task '{func.__name__}'s @tool decorator in output_parameters "
                    f"must define exactly one output property, but {len(keys)} were found: {keys}"
                )
            
            output_key = keys[0]
            output_dict[output_key] = user_function_result
            final_output_obj = output_dict
        
        # 将【包装后的】结果对象（或 None）存入 DAGContext
        ray.get(ctx_actor.put.remote(f"{task_id}.output", final_output_obj))
        
        # --- 阶段4：文件同步推送 (Push) ---
        logger.debug(f"⤴️ [Push] Task {task_id} checking for file changes to push...")
        files_to_push = {}
        for root, dirs, files in os.walk(data_dir):
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            if '.git' in dirs:
                dirs.remove('.git')

            for name in files:
                if name.endswith('.pyc') or name.endswith('.py'):
                    continue
                
                file_abs_path = os.path.join(root, name)
                file_rel_path = os.path.relpath(file_abs_path, data_dir)
                
                current_hash = _calculate_local_sha256(file_abs_path)
                
                if file_rel_path not in master_hashes or master_hashes[file_rel_path] != current_hash:
                    with open(file_abs_path, 'rb') as f:
                        files_to_push[file_rel_path] = f.read()

        if files_to_push:
            logger.debug(f"  - [Push] Found {len(files_to_push)} new/modified DATA files to upload.")
            upload_url = f"http://{master_addr}/files/upload/{run_id}"
            multipart_files = [(path, (path, content)) for path, content in files_to_push.items()]
            response = requests.post(upload_url, files=multipart_files)
            response.raise_for_status()
            logger.debug("  - ✅ [Push] Files uploaded to master successfully.")
        else:
            logger.debug("  - [Push] No data file changes detected.")

        return {
            "status": "finished", 
            "worker_start_exec_time": worker_start_exec_time,
            "end_time": time.time()}
    except Exception as e:
        import traceback
        error_message = f"Error in remote_task_runner: {e}\n{traceback.format_exc()}"
        logger.debug(f"[FAILED] Task {task_id} failed with error: {error_message}")
        return {"status": "failed", "err_msg": error_message}