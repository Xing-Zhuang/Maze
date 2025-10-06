import uuid
from abc import ABC, abstractmethod
from maze.agent.graph import ExecuteDAG
import logging
from typing import Any
import requests
import logging
import zipfile
import json
import os
import io
import shutil
from maze.core.register.task_registry import task_registry

class Agent(ABC):
    """Agent的抽象基类"""
    def __init__(self, name: str, agent_config: dict = None):
        self.id = f"agent_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.config = agent_config or {}

class DAGAgent(Agent):
    def __init__(self, name: str, root_path: str = None, agent_config: dict = None, scheduler_addr: str = "127.0.0.1:6382"):
        # --- 核心修改：在最开始自动配置客户端日志 ---
        # --- 原有的初始化逻辑保持不变 ---
        super().__init__(name, agent_config)
        self.workflow = ExecuteDAG(scheduler_addr=scheduler_addr, root_path=root_path)
        self.workflow.name = name

    def get(self, task_id: str, run_id: str) -> dict:
        """
        (V3 - 最终版)
        获取任务的当前状态和结果，同时兼容本地和服务器模式。
        此方法是非阻塞的，并始终返回一个包含状态信息的字典。
        """
        if task_id not in self.workflow._tasks:
            raise ValueError(f"Task with id '{task_id}' not found in this workflow's definition.")

        # --- 针对本地模式的逻辑 ---
        if run_id.startswith("local_"):
            print(f"Getting local result for task '{task_id}' from run '{run_id}'...")
            
            # 从本地工作流的 results 字典中获取 TaskResult 对象
            task_result_obj = self.workflow.results.get(task_id)
            if not task_result_obj:
                 return {"status": "error", "task_status": "unknown", "msg": f"Task '{task_id}' not found in local run results."}

            # 根据 TaskResult 对象构建与服务器API格式一致的响应
            response = {"task_status": task_result_obj.status}
            if task_result_obj.status == "finished":
                response["data"] = task_result_obj.result
            elif task_result_obj.status == "failed":
                response["error"] = task_result_obj.error
            return response

        # --- 针对服务器模式的逻辑 ---
        else:
            print(f"Getting server result for task '{task_id}' from run '{run_id}'...")
            get_url = f"{self.workflow.scheduler_addr}/get/"
            payload = { "run_id": run_id, "task_id": task_id }
            
            try:
                response = requests.post(get_url, json=payload, timeout=60)
                # 优雅地处理404等错误，而不是直接崩溃
                if response.status_code != 200:
                    try:
                        # 尝试解析服务器返回的错误JSON
                        return response.json()
                    except json.JSONDecodeError:
                        # 如果返回的不是JSON，则自己构造一个错误信息
                        return {"status": "error", "msg": f"Server returned status {response.status_code}", "task_status": "unknown"}
                
                # 直接返回服务器的完整JSON响应
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to get result for task '{task_id}': {e}", exc_info=True)
                return {"status": "error", "msg": str(e)}

    def list_available_tools(self, display: bool = True) -> list:
        """
        从服务器获取所有可用的工具（模型）列表。

        :param display: 是否在控制台以表格形式打印结果。默认为 True。
        :return: 包含所有工具元数据字典的列表。
        """
        list_url = f"{self.workflow.scheduler_addr}/tools"
        print(f"Fetching available tools from {list_url}...")
        
        try:
            response = requests.get(list_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "success":
                print(f"Failed to list tools: {data.get('msg')}")
                return []

            tools = data.get("tools", [])
            
            if display:
                print("\n" + "="*20 + " Available Tools " + "="*20)
                if not tools:
                    print("No tools found on the server.")
                else:
                    # 打印表格
                    max_name = max([len(t.get('name', '')) for t in tools] + [10])
                    max_type = max([len(t.get('type', '')) for t in tools] + [8])
                    header = f"| {'Name':<{max_name}} | {'Type':<{max_type}} | Description"
                    print(header)
                    print(f"|{'-'*(max_name+2)}|{'-'*(max_type+2)}|{'-'*20}")
                    for tool in tools:
                        print(f"| {tool.get('name', ''):<{max_name}} | {tool.get('type', ''):<{max_type}} | {tool.get('description', '')}")
                print("="*57)

            return tools

        except requests.exceptions.RequestException as e:
            print("Failed to connect to the server to list tools.", exc_info=True)
            return []

    # --- 新增方法2: upload_tool ---
    def upload_tool(self, tool_path: str, description: str, tool_type: str, version: str = "1.0.0", author: str = "unknown"):
        """
        将本地的工具文件夹打包并上传到服务器的 model_cache 中。

        :param tool_path: 工具在本地的文件夹路径。文件夹名将作为工具的唯一名称。
        :param description: 工具的人类可读描述。
        :param tool_type: 工具类型 (e.g., 'vision', 'language')。
        :param version: (可选) 工具版本。
        :param author: (可选) 工具作者。
        :return: 成功返回 True，失败返回 False。
        """
        if not os.path.isdir(tool_path):
            print(f"Upload failed: Provided path '{tool_path}' is not a valid directory.")
            return False

        tool_name = os.path.basename(os.path.normpath(tool_path))
        upload_url = f"{self.workflow.scheduler_addr}/tools/upload"
        print(f"Preparing to upload tool '{tool_name}' from '{tool_path}' to {upload_url}...")

        # 在内存中创建zip压缩包
        memory_file = io.BytesIO()
        shutil.make_archive(base_name='temp_tool_archive', format='zip', root_dir=tool_path, base_dir='.')
        # make_archive 会在磁盘上创建文件，我们需要读取它到内存
        with open('temp_tool_archive.zip', 'rb') as f:
            memory_file.write(f.read())
        os.remove('temp_tool_archive.zip')
        memory_file.seek(0)
        
        # 准备multipart/form-data
        files = {'tool_archive': ('tool.zip', memory_file, 'application/zip')}
        data = {
            "tool_name": tool_name,
            "description": description,
            "tool_type": tool_type,
            "version": version,
            "author": author,
        }

        try:
            print(f"Uploading tool '{tool_name}'...")
            response = requests.post(upload_url, data=data, files=files, timeout=300)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("status") == "success":
                print(f"Successfully uploaded tool '{tool_name}'.")
                return True
            else:
                print(f"Failed to upload tool '{tool_name}': {response_data.get('msg')}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Upload failed for tool '{tool_name}'.", exc_info=True)
            return False

    def update_task(self, task_id: str, new_name: str = None, new_inputs: dict = None):
        """
        安全地更新一个已存在任务的属性，如名称或输入依赖。

        如果更新输入依赖 (`new_inputs`) 会导致工作流产生循环，
        操作将被自动回滚并抛出 ValueError 异常。

        :param task_id: 需要更新的任务ID。
        :param new_name: (可选) 新的任务别名。
        :param new_inputs: (可选) 新的输入字典，用于重新定义依赖关系。
        """
        print(f"Attempting to update task with id: {task_id}")
        try:
            self.workflow.update_task(task_id, new_name=new_name, new_inputs=new_inputs)
            print(f"Successfully updated task '{task_id}'.")
        except ValueError as e:
            print(f"Failed to update task '{task_id}': {e}")
            raise

    # 3. 将所有工作流相关操作，全部“委托”给内部的 self.workflow 对象
    def add_task(self, func, task_name: str = None, inputs: dict = None, file_paths: list = None, resources: dict = None):
        return self.workflow.add_task(func, task_name, inputs, file_paths, resources)

    def remove_task(self, task_id: str):
        """
        安全地从工作流中移除一个任务节点。
        如果该任务有下游依赖，操作将被中止并抛出异常。
        :param task_id: 需要被移除的任务的ID。
        """
        print(f"Attempting to remove task with id: {task_id}")
        try:
            # 调用 self.workflow (即 ExecuteDAG 实例) 的 remove_task 方法
            self.workflow.remove_task(task_id)
            print(f"Successfully removed task '{task_id}'.")
        except ValueError as e:
            # 捕获并记录错误，然后重新抛出，以便用户脚本能感知到失败
            print(f"Failed to remove task '{task_id}': {e}")
            raise

    def visualize(self, style_options: dict = None):
        return self.workflow.visualize(style_options)

    def show_structure(self):
        return self.workflow.show_structure()

    def submit(self, mode: str = 'local'):
        # 调用 workflow 对象的 submit 方法，这在逻辑上非常清晰
        return self.workflow.submit(mode=mode)

    def destroy(self, run_id: str) -> bool:
        """
        安全地销毁并清理指定run_id的所有相关资源。
        - 只有在工作流结束后才会执行清理。
        - 兼容本地和服务器模式。
        """
        if not run_id:
            print("Destroy failed: run_id cannot be empty.")
            return False

        # --- 核心修改：统一的安全检查逻辑 ---
        if run_id.startswith("local_"):
            # --- 本地模式的逻辑 ---
            print(f"Initiating cleanup for local run: {run_id}")
            
            # 1. 检查本地线程是否仍在运行
            thread = self.workflow._local_run_threads.get(run_id)
            if thread and thread.is_alive():
                print(f"Destroy aborted: Local workflow for run '{run_id}' is still running.")
                return False

            # 2. 如果线程已结束，则执行清理
            self.workflow._local_run_results.pop(run_id, None)
            self.workflow.results.clear()
            self.workflow._local_run_threads.pop(run_id, None) # 清理线程引用
            
            print("  -> In-memory state for local run has been cleared.")
            
            # 3. 打印提示信息给用户
            print(
                "\n" + "="*50 +
                "\n[ATTENTION] For local runs, Maze does not automatically delete files." +
                "\nPlease manually clean up any generated files." +
                "\n" + "="*50
            )
            return True
        
        else:
            # --- 服务器模式的逻辑 (保持不变) ---
            destroy_url = f"{self.workflow.scheduler_addr}/runs/destroy"
            payload = {"run_id": run_id}
            
            print(f"Sending request to {destroy_url} to clean up server run '{run_id}'...")
            
            try:
                response = requests.post(destroy_url, json=payload, timeout=30)
                response_data = response.json()

                if response.status_code == 200 and response_data.get("status") == "success":
                    print(f"Successfully cleaned up server run '{run_id}'.")
                    return True
                else:
                    print(f"Server failed to clean up run '{run_id}': {response_data.get('msg')}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Failed to send destroy request for run '{run_id}'.", exc_info=True)
                return False
            
    def get_summary(self, run_id: str) -> dict:
        """
        获取指定run_id的执行摘要数据。兼容本地和服务器模式。
        返回一个包含工作流和任务级别摘要的字典。
        """
        # --- 本地模式逻辑 ---
        if run_id.startswith("local_"):
            print(f"Getting local summary for run '{run_id}'...")
            task_summaries = []
            
            # 遍历本地结果，生成任务摘要
            for task_id, task_result in self.workflow.results.items():
                task_summaries.append({
                    "name": task_result.task_name,
                    "task_id": task_id,
                    "status": task_result.status,
                    "duration": f"{task_result.duration:.2f}" if task_result.duration is not None else "N/A"
                })
            
            # 计算工作流摘要
            run_summary = {}
            thread = self.workflow._local_run_threads.get(run_id)
            if thread and thread.is_alive():
                run_summary["status"] = "running"
                run_summary["total_duration"] = "N/A"
            else:
                run_summary["status"] = "completed"
                # 计算总时长：从第一个任务开始到最后一个任务结束
                start_times = [res.submit_time for res in self.workflow.results.values() if res.submit_time]
                end_times = [res.finish_time for res in self.workflow.results.values() if res.finish_time]
                if start_times and end_times:
                    total_duration = max(end_times) - min(start_times)
                    run_summary["total_duration"] = f"{total_duration:.2f}"
                else:
                    run_summary["total_duration"] = "N/A"
            
            return {
                "status": "success",
                "run_summary": run_summary,
                "task_summaries": task_summaries
            }

        # --- 服务器模式逻辑 ---
        else:
            print(f"Getting server summary for run '{run_id}'...")
            summary_url = f"{self.workflow.scheduler_addr}/runs/{run_id}/summary"
            try:
                response = requests.get(summary_url, timeout=60)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Failed to get summary for run '{run_id}': {e}", exc_info=True)
                return {"status": "error", "msg": str(e)}

    # --- 新增方法2: display_summary ---
    def display_summary(self, run_id: str):
        """
        获取并以美观的表格形式，在控制台打印指定run_id的执行摘要。
        """
        print(f"\nFetching execution summary for Run ID: {run_id}...")
        summary_data = self.get_summary(run_id)

        if not summary_data or summary_data.get("status") != "success":
            print(f"❌ Failed to retrieve summary: {summary_data.get('msg', 'Unknown error')}")
            return

        run_summary = summary_data.get("run_summary", {})
        task_summaries = summary_data.get("task_summaries", [])
        
        # --- 打印表格 ---
        # 动态计算列宽
        max_name_len = max([len(t.get('name', 'N/A')) for t in task_summaries] + [15])
        max_id_len = max([len(t.get('task_id', 'N/A')) for t in task_summaries] + [15])

        # 表头
        header = f"| {'Task Name':<{max_name_len}} | {'Task ID':<{max_id_len}} | {'Status':<10} | {'Duration (s)':<15} |"
        separator = f"+{'-' * (max_name_len + 2)}+{'-' * (max_id_len + 2)}+{'-' * 12}+{'-' * 17}+"
        
        print(separator)
        print(header)
        print(separator)

        # 表内容
        for task in task_summaries:
            row = f"| {task.get('name', 'N/A'):<{max_name_len}} | {task.get('task_id', 'N/A'):<{max_id_len}} | {task.get('status', 'N/A'):<10} | {task.get('duration', 'N/A'):<15} |"
            print(row)
        
        print(separator)
        
        # 表尾
        duration_str = run_summary.get('total_duration', 'N/A')
        footer = f"| Total Workflow Duration: {duration_str}s"
        print(footer)
        footer_sep_len = max(len(footer), len(separator) - 2)
        print("+" + "-" * (footer_sep_len) + "+")

    def download_results(self, run_id: str, download_to_path: str = '.', extract: bool = True) -> str:
        """
        从服务器下载指定 run_id 的所有结果文件。

        :param run_id: 需要下载的运行ID。
        :param download_to_path: 文件下载并解压到的本地目录，默认为当前目录。
        :param extract: 是否在下载后自动解压zip文件，并删除zip文件。默认为 True。
        :return: 成功时返回文件被解压到的最终路径，失败则返回 None。
        """
        if run_id.startswith("local_"):
            print("This was a local run. All files are already on your local filesystem.")
            return None
            
        download_url = f"{self.workflow.scheduler_addr}/runs/{run_id}/download"
        print(f"Downloading results for run '{run_id}' from {download_url}...")

        try:
            # 使用流式下载，适合大文件
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                
                # 确保下载目录存在
                os.makedirs(download_to_path, exist_ok=True)
                zip_path = os.path.join(download_to_path, f"{run_id}_results.zip")
                
                # 将文件流写入本地zip文件
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
                
                print(f"Successfully downloaded results archive to '{zip_path}'.")

                # 如果需要，自动解压
                if extract:
                    print(f"Extracting archive to '{download_to_path}'...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(path=download_to_path)
                    
                    # 删除临时的zip文件
                    os.remove(zip_path)
                    print("Extraction complete and archive removed.")
                
                return os.path.abspath(download_to_path)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download results for run '{run_id}': {e}", exc_info=True)
            return None
        except Exception as e:
            print(f"An error occurred during file handling for run '{run_id}': {e}", exc_info=True)
            return None
        
    def register_task(self, func: callable):
        """
        为一个 Agent 会话动态注册一个用户自定义的任务。
        :param func: 需要注册的、被 @tool 装饰的函数对象。
        """
        print(f"Registering custom task '{func._tool_meta.get('name')}' for this agent session.")
        task_registry.register_task(func)