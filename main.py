import argparse
import os
import ray
import threading
import queue
import json
import base64
import uuid
import shutil
import hashlib
import io
import zipfile
import traceback
from flask import Flask, request, jsonify, send_file
from pathlib import Path
import shutil
from maze.agent.config import config
from maze.core.path.task_dispatcher import TaskScheduler
from maze.core.path.dag_context import DAGContextManager
from maze.core.path.task_status_manager import TaskStatusManager
from maze.core.worker.resource_manager import ComputeNodeResourceManager
from maze.core.path.daps import dag_manager_daps
from maze.utils.log_config import setup_logging
import logging
from maze.core.register.task_registry import task_registry
app = Flask(__name__)

# ===================================================================
# 辅助函数
# ===================================================================

def _calculate_sha256(file_path):
    """一个计算文件SHA256哈希值的辅助函数"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

# ===================================================================
# API 路由定义 (整合自 scheduler.py 和 api_server.py)
# ===================================================================

@app.route('/runs/<run_id>/summary', methods=['GET'])
def get_run_summary(run_id):
    """
    获取指定 run_id 的工作流执行摘要，包括总耗时和各任务耗时。
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Received request for summary of run_id: {run_id}")
    
    status_mgr = app.config['STATUS_MGR']
    task_summaries = []
    
    # 1. 收集所有相关任务的摘要
    for (r_id, t_id), record in status_mgr.task_records.items():
        if r_id == run_id:
            task_info = record.task_info or {}
            task_summaries.append({
                "name": task_info.get("func_name", "N/A"),
                "task_id": t_id,
                "status": record.status,
                "duration": f"{record.duration:.2f}" if record.duration is not None else "N/A"
            })

    if not task_summaries:
        return jsonify({"status": "error", "msg": f"Run with id '{run_id}' not found."}), 404

    # 2. 获取整个工作流的摘要
    run_summary = status_mgr.run_records.get(run_id, {})
    # 如果工作流还在跑，run_records里可能还没有记录，我们给一个默认状态
    if not run_summary:
        run_summary = {"status": "running", "total_duration": "N/A"}

    # 3. 组合并返回最终结果
    return jsonify({
        "status": "success",
        "run_summary": run_summary,
        "task_summaries": task_summaries
    })

@app.route('/runs/destroy', methods=['POST'])
def destroy_run():
    """
    接收 run_id，并安全地执行销毁和清理操作。
    仅当工作流中所有任务都已结束后，才会执行清理。
    """
    logger = logging.getLogger(__name__)
    data = request.get_json()
    run_id = data.get("run_id")
    
    if not run_id:
        logger.warning("Destroy request received without a run_id.")
        return jsonify({"status": "error", "msg": "run_id is required"}), 400
    
    logger.info(f"🔥 Received request to destroy and clean up run_id: {run_id}")

    try:
        status_mgr = app.config['STATUS_MGR']

        # --- 核心修改：使用新的 task_records 数据结构 ---
        is_workflow_active = False
        # 获取所有任务的记录对象
        all_task_records = status_mgr.task_records
        
        # 遍历记录，检查是否有属于该 run_id 的任务仍在活动
        for (r_id, t_id), record in all_task_records.items():
            if r_id == run_id:
                # 从记录对象中获取状态
                if record.status not in ["finished", "failed"]:
                    is_workflow_active = True
                    logger.warning(f"Destroy aborted: Task '{t_id}' in run '{run_id}' is still active with status '{record.status}'.")
                    break # 发现一个活跃任务，即可中断检查

        if is_workflow_active:
            return jsonify({
                "status": "error", 
                "msg": "Workflow is still running and cannot be destroyed."
            }), 400
        # --- 修改结束 ---

        # 如果检查通过，则执行后续的清理逻辑
        logger.info(f"Workflow for run '{run_id}' is not active. Proceeding with cleanup.")
        
        # 1. 释放 DAGContext Actor
        ctx_mgr = app.config['DAG_CTX_MGR']
        ctx_released = ctx_mgr.release_context(run_id)
        if ctx_released:
            logger.info(f"  -> Step 1/3: DAGContext for run '{run_id}' released successfully.")
        else:
            logger.warning(f"  -> Step 1/3: DAGContext for run '{run_id}' not found or already released.")

        # 2. 清理 TaskStatusManager 中的记录
        status_mgr.cleanup_run(run_id)
        logger.info(f"  -> Step 2/3: Task status records for run '{run_id}' cleaned up.")

        # 3. 删除磁盘上的运行目录
        project_root = app.config['PROJECT_ROOT']
        run_path = os.path.join(project_root, "maze", "runtime_artifacts", "runs", run_id)
        
        if os.path.isdir(run_path):
            shutil.rmtree(run_path)
            logger.info(f"  -> Step 3/3: Run directory '{run_path}' deleted successfully.")
        else:
            logger.warning(f"  -> Step 3/3: Run directory '{run_path}' not found on disk.")

        return jsonify({"status": "success", "msg": f"All artifacts for completed run_id '{run_id}' have been cleaned up."})

    except Exception as e:
        logger.error(f"An error occurred during cleanup for run_id '{run_id}'.", exc_info=True)
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/submit_agent/', methods=['POST'])
def submit_agent():
    logger = logging.getLogger(__name__)
    logger.info("🚀 Received a new submission via /submit_agent/ endpoint.")
    try:
        if 'workflow_payload' not in request.files or 'project_archive' not in request.files:
            return jsonify({"status": "error", "msg": "Missing 'workflow_payload' or 'project_archive'."}), 400

        workflow_payload_file = request.files['workflow_payload']
        payload_json = workflow_payload_file.read().decode('utf-8')
        payload_dict = json.loads(payload_json)
        
        for task_id, task_data in payload_dict['tasks'].items():
            encoded_func = task_data['serialized_func']
            task_data['serialized_func'] = base64.b64decode(encoded_func)
        
        logger.info(f"  - Workflow payload for '{payload_dict['name']}' parsed successfully.")

        project_archive_file = request.files['project_archive']
        run_id = str(uuid.uuid4())
        server_root_path = os.path.join(config.get('paths', {}).get('project_root'), "maze", "runtime_artifacts", "runs", run_id)
        os.makedirs(server_root_path, exist_ok=True)
        
        archive_path = os.path.join(server_root_path, 'project.zip')
        project_archive_file.save(archive_path)
        shutil.unpack_archive(archive_path, server_root_path)
        os.remove(archive_path)
        logger.info(f"  - Project archive extracted to server path: '{server_root_path}'")

        submission_package = {
            "submission_type": "dynamic_agent",
            "run_id": run_id,
            "server_root_path": server_root_path,
            "workflow_payload": payload_dict
        }
        
        app.config['DAG_SUBMISSION_QUEUE'].put(submission_package)
        logger.info(f"  - Submission package for run_id '{run_id}' has been enqueued.")

        return jsonify({"status": "success", "msg": "Agent submitted successfully.", "run_id": run_id})

    except Exception as e:
        logger.error(f"❌ Error processing agent submission: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/get/', methods=['POST'])
def get():
    logger = logging.getLogger(__name__)
    data = request.get_json()
    run_id = data.get("run_id")
    task_id = data.get("task_id")
    
    if not run_id or not task_id:
        return jsonify({"status": "error", "msg": "run_id and task_id are required"}), 400

    try:
        # 1. 先从 TaskStatusManager 获取状态
        status_mgr = app.config['STATUS_MGR']
        task_status = status_mgr.get_status(run_id, task_id)
        logger.debug(f"Querying status for task '{task_id}' in run '{run_id}': {task_status}")

        # 2. 根据状态进行分支处理
        if task_status == "finished":
            # ... (处理 'finished' 状态的逻辑) ...
            dag_ctx_mgr = app.config['DAG_CTX_MGR']
            dag_ctx = dag_ctx_mgr.get_context(run_id)
            if dag_ctx is None:
                return jsonify({"status": "error", "msg": f"Context for run_id '{run_id}' not found, though task is marked 'finished'."}), 404
            
            try:
                task_result_obj = ray.get(dag_ctx.get.remote(f"{task_id}.output"))
                
                final_ret_data = None
                if isinstance(task_result_obj, dict):
                    serializable_dict = {}
                    for key, value in task_result_obj.items():
                        serializable_dict[key] = f"bytes_data(len:{len(value)})" if isinstance(value, bytes) else value
                    final_ret_data = serializable_dict
                elif task_result_obj is not None:
                    final_ret_data = str(task_result_obj)

                return jsonify({"status": "success", "task_status": "finished", "data": final_ret_data})

            except KeyError:
                logger.error(f"Inconsistency: Task '{task_id}' status is 'finished' but result not found in DAGContext for run '{run_id}'.")
                return jsonify({"status": "error", "task_status": "finished", "msg": "Result not found in context."}), 404

        elif task_status in ["received", "waiting", "running"]:
            return jsonify({"status": "success", "task_status": task_status})

        elif task_status == "failed":
            failed_info = status_mgr.get_failed_task_info(run_id, task_id)
            error_msg = failed_info.get("err_msg", "Unknown error.")
            return jsonify({"status": "success", "task_status": "failed", "error": error_msg})
            
        else: # "unknown"
            return jsonify({"status": "error", "msg": f"Task with id '{task_id}' not found in run '{run_id}'."}), 404

    except Exception as e:
        logger.error(f"An unexpected error occurred in /get/ endpoint for run '{run_id}'.", exc_info=True)
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/files/hashes/<run_id>', methods=['GET'])
def get_file_hashes(run_id):
    try:
        run_path = os.path.join(config.get('paths', {}).get('project_root'), "maze", "runtime_artifacts", "runs", run_id)
        if not os.path.isdir(run_path):
            return jsonify({"status": "error", "msg": f"Run with id '{run_id}' not found."}), 404

        hashes = {}
        for root, _, files in os.walk(run_path):
            for name in files:
                file_abs_path = os.path.join(root, name)
                file_rel_path = os.path.relpath(file_abs_path, run_path)
                hashes[file_rel_path] = _calculate_sha256(file_abs_path)
        
        return jsonify({"status": "success", "hashes": hashes})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/files/download/<run_id>', methods=['POST'])
def download_files(run_id):
    try:
        run_path = os.path.join(config.get('paths', {}).get('project_root'), "maze", "runtime_artifacts", "runs", run_id)
        if not os.path.isdir(run_path):
            return jsonify({"status": "error", "msg": f"Run with id '{run_id}' not found."}), 404
        
        files_to_download = request.get_json().get('files', [])
        if not files_to_download:
            return jsonify({"status": "error", "msg": "File list is empty."}), 400

        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_rel_path in files_to_download:
                file_abs_path = os.path.join(run_path, file_rel_path)
                if os.path.exists(file_abs_path):
                    zf.write(file_abs_path, arcname=file_rel_path)
        
        memory_file.seek(0)
        
        return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=f'{run_id}_update.zip')
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/files/upload/<run_id>', methods=['POST'])
def upload_files(run_id):
    logger = logging.getLogger(__name__)
    try:
        run_path = os.path.join(config.get('paths', {}).get('project_root'), "maze", "runtime_artifacts", "runs", run_id)
        if not os.path.isdir(run_path):
            os.makedirs(run_path, exist_ok=True)

        if not request.files:
            return jsonify({"status": "success", "msg": "No files to upload."})

        for file_key in request.files:
            file = request.files[file_key]
            file_abs_path = os.path.join(run_path, file_key)
            os.makedirs(os.path.dirname(file_abs_path), exist_ok=True)
            file.save(file_abs_path)
            
        logger.info(f"✅ [Push] Received and saved {len(request.files)} file(s) for run_id '{run_id}'.")
        return jsonify({"status": "success", "msg": f"Successfully uploaded {len(request.files)} files."})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/runs/<run_id>/download', methods=['GET'])
def download_run_results(run_id):
    """
    将指定 run_id 的整个运行目录打包成zip文件并提供下载。
    """
    logger = logging.getLogger(__name__)
    project_root = app.config['PROJECT_ROOT']
    run_path = os.path.join(project_root, "maze", "runtime_artifacts", "runs", run_id)
    
    logger.info(f"Received request to download results for run_id: {run_id}")
    
    if not os.path.isdir(run_path):
        logger.error(f"Download request failed: Run directory not found for run_id '{run_id}'.")
        return jsonify({"status": "error", "msg": f"Run with id '{run_id}' not found."}), 404

    try:
        # 在内存中创建zip文件，避免在服务器上产生临时文件
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 遍历run_path下的所有文件和文件夹
            for root, _, files in os.walk(run_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # arcname是文件在zip包里的路径，我们使用相对路径以保持结构
                    arcname = os.path.relpath(file_path, run_path)
                    zf.write(file_path, arcname=arcname)
        
        # 将文件指针移到开头
        memory_file.seek(0)
        
        logger.info(f"Successfully created zip archive in memory for run '{run_id}'. Sending file...")
        
        # 使用 send_file 将内存中的文件作为附件发送
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{run_id}_results.zip'
        )
    except Exception as e:
        logger.error(f"Failed to create zip file for run '{run_id}'.", exc_info=True)
        return jsonify({"status": "error", "msg": "Failed to create result archive."}), 500

@app.route('/tools', methods=['GET'])
def list_tools():
    logger = logging.getLogger(__name__)
    try:
        project_root = app.config['PROJECT_ROOT']
        paths_config = config.get('paths', {})
        model_cache_dir_name = paths_config.get('model_cache_dir', 'model_cache')
        model_cache_path = os.path.join(project_root, model_cache_dir_name)

        if not os.path.isdir(model_cache_path):
            return jsonify({"status": "success", "tools": []})

        tools_list = []
        for tool_name in os.listdir(model_cache_path):
            tool_dir = os.path.join(model_cache_path, tool_name)
            metadata_path = os.path.join(tool_dir, 'metadata.json')
            if os.path.isdir(tool_dir) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        tools_list.append(metadata)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not read or parse metadata for tool '{tool_name}': {e}")
        
        logger.info(f"Found and listed {len(tools_list)} available tools.")
        return jsonify({"status": "success", "tools": tools_list})

    except Exception as e:
        logger.error("Failed to list tools.", exc_info=True)
        return jsonify({"status": "error", "msg": "Failed to list tools."}), 500

@app.route('/tools/upload', methods=['POST'])
def upload_tool():
    logger = logging.getLogger(__name__)
    try:
        # 1. 检查文件和表单数据是否存在
        if 'tool_archive' not in request.files:
            return jsonify({"status": "error", "msg": "Missing 'tool_archive' file part."}), 400
        
        required_fields = ['tool_name', 'description', 'tool_type']
        if not all(field in request.form for field in required_fields):
            return jsonify({"status": "error", "msg": f"Missing required form data. Required fields are: {required_fields}"}), 400

        # 2. 获取元数据和文件
        metadata = {
            "name": request.form['tool_name'],
            "description": request.form['description'],
            "type": request.form['tool_type'],
            "version": request.form.get('version', '1.0.0'),
            "author": request.form.get('author', 'unknown'),
            "usage_notes": request.form.get('usage_notes', '')
        }
        tool_file = request.files['tool_archive']
        
        # 3. 准备目标路径
        project_root = app.config['PROJECT_ROOT']
        paths_config = config.get('paths', {})
        model_cache_dir_name = paths_config.get('model_cache_dir', 'model_cache')
        
        tool_dir = os.path.join(project_root, model_cache_dir_name, metadata['name'])
        
        if os.path.exists(tool_dir):
            return jsonify({"status": "error", "msg": f"Tool '{metadata['name']}' already exists."}), 409 # 409 Conflict

        os.makedirs(tool_dir, exist_ok=True)
        
        # 4. 解压上传的zip文件
        logger.info(f"Extracting tool '{metadata['name']}' to '{tool_dir}'...")
        with zipfile.ZipFile(tool_file.stream, 'r') as zf:
            zf.extractall(tool_dir)
        
        # 5. 自动生成 metadata.json 文件
        metadata_path = os.path.join(tool_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Successfully uploaded and registered new tool: '{metadata['name']}'.")
        return jsonify({"status": "success", "msg": f"Tool '{metadata['name']}' uploaded successfully."})

    except Exception as e:
        logger.error("Failed to upload tool.", exc_info=True)
        # 如果出错，尝试清理已创建的文件夹
        if 'tool_dir' in locals() and os.path.isdir(tool_dir):
            shutil.rmtree(tool_dir)
        return jsonify({"status": "error", "msg": "Failed to upload tool."}), 500

@app.route('/tools/<tool_name>', methods=['DELETE'])
def delete_tool(tool_name):
    logger = logging.getLogger(__name__)
    logger.info(f"Received request to delete tool: '{tool_name}'")

    try:
        project_root = app.config['PROJECT_ROOT']
        paths_config = config.get('paths', {})
        model_cache_dir_name = paths_config.get('model_cache_dir')
        model_cache_path = os.path.join(project_root, model_cache_dir_name)
        
        tool_path = os.path.join(model_cache_path, tool_name)

        # --- 关键：路径安全检查 ---
        # 解析真实路径，防止 ".." 等路径遍历攻击
        safe_base_path = os.path.abspath(model_cache_path)
        target_path = os.path.abspath(tool_path)

        if not target_path.startswith(safe_base_path):
            logger.error(f"Security risk detected! Attempt to delete file outside of model_cache: '{tool_name}'")
            return jsonify({"status": "error", "msg": "Invalid tool name."}), 400

        # 检查工具是否存在
        if not os.path.isdir(target_path):
            logger.warning(f"Delete failed: Tool directory '{tool_name}' not found.")
            return jsonify({"status": "error", "msg": f"Tool '{tool_name}' not found."}), 404

        # 执行删除
        shutil.rmtree(target_path)
        logger.info(f"Successfully deleted tool directory: '{target_path}'")
        return jsonify({"status": "success", "msg": f"Tool '{tool_name}' deleted successfully."})

    except Exception as e:
        logger.error(f"Failed to delete tool '{tool_name}'.", exc_info=True)
        return jsonify({"status": "error", "msg": "Failed to delete tool due to a server error."}), 500

# ===================================================================
# 主服务入口
# ===================================================================
def main():
    setup_logging()
    task_registry.discover_tasks()
    logger = logging.getLogger(__name__)
    server_config = config.get('server', {})
    paths_config = config.get('paths', {})
    project_root = paths_config.get('project_root')
    if not project_root:
        raise ValueError("'project_root' not defined in [paths] section of config.toml")

    try:
        ray.init(address='auto',
            runtime_env={
                "working_dir": project_root,
                "excludes": [".git", "model_cache", "runs", "AgentOS", "backup", "fastercnn", "maze-temp", "*.safetensors", "*.bin"]
            })
        logger.info("✅ Ray connected successfully.")

        logger.info("🚀 Initializing core services...")
        
        dag_submission_queue = queue.Queue()
        task_completion_queue = queue.Queue()

        status_mgr = TaskStatusManager()
        dag_ctx_mgr = DAGContextManager()
        resource_mgr = ComputeNodeResourceManager()

        scheduler = TaskScheduler(
            resource_mgr=resource_mgr, 
            status_mgr=status_mgr, 
            dag_ctx_mgr=dag_ctx_mgr,
            completion_queue=task_completion_queue,
            master_addr=f"{server_config.get('host')}:{server_config.get('port')}",
            proj_path=project_root, 
            model_folder=paths_config.get('model_folder'),
            models_config_path=os.path.join(project_root, 'maze', 'config', 'models.json')
        )
        logger.info("✅ All services initialized.")

        app.config['DAG_SUBMISSION_QUEUE'] = dag_submission_queue
        app.config['TASK_SCHEDULER'] = scheduler
        app.config['STATUS_MGR'] = status_mgr
        app.config['DAG_CTX_MGR'] = dag_ctx_mgr
        app.config['RESOURCE_MGR'] = resource_mgr
        app.config['PROJECT_ROOT'] = project_root
        
        logger.info("🚀 Starting background scheduler thread...")
        
        daps_args = argparse.Namespace(
            # daps.py 内部不再需要 redis_ip 和 redis_port
            time_pred_model_path=os.path.join(project_root, "maze", "utils", "timepred", "model"),
            min_sample4train=10,
            min_sample4incremental=10,
        )

        scheduler_thread = threading.Thread(
            target=dag_manager_daps,
            args=(
                daps_args,
                dag_submission_queue,
                task_completion_queue,
                scheduler,
                threading.Event()
            ),
            daemon=True
        )
        scheduler_thread.start()
        logger.info("✅ Background scheduler is running.")
        
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8000)
        logger.info(f"🚀 Starting Maze Unified Server on http://{host}:{port}")
        app.run(host=host, port=port, threaded=True)

    except Exception as e:
        logger.error(f"❌ An error occurred during server startup: {e}\n{traceback.format_exc()}")
    finally:
        if ray.is_initialized():
            logger.info("👋 Shutting down Ray...")
            ray.shutdown()
        logger.info("Exiting.")

if __name__ == '__main__':
    main()


