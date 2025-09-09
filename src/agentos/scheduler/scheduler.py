import uuid
import ray
import multiprocessing
from flask import Flask, request, jsonify
import cloudpickle
import requests
import argparse
import networkx as nx
import json
import threading
from datetime import datetime, timedelta
import queue
import os
import importlib.util
import redis
import time
import copy
from agentos.scheduler.mlq import dag_manager_mlq
from agentos.scheduler.heft import dag_manager_heft
from agentos.scheduler.cpop import dag_manager_cpop
from agentos.scheduler.fcfs import dag_manager_fcfs
from agentos.scheduler.ltf import dag_manager_ltf
from agentos.scheduler.peft import dag_manager_peft
from agentos.scheduler.ftc import dag_manager_ftc
import csv
app = Flask(__name__) 
dag_que = multiprocessing.Queue() 
dag_status_dict = multiprocessing.Manager().dict()

def parse_args():
    """
    parse command arguments.
    """
    #解析参数
    parser = argparse.ArgumentParser(description="Run script with parameters.")
    # parser.add_argument("--proj_path",  default="/app/workplace/AgentOS", 
    #                     help="Path to the directory of the AgentOS framework")
    parser.add_argument("--proj_path",  default="/root/workspace/d23oa7cp420c73acue30/AgentOS", 
                        help="Path to the directory of the AgentOS framework")
    parser.add_argument("--tokenizer_path",  default="model_cache/Qwen/Qwen3-32B", 
                        help="Path to the directory of the AgentOS framework")    
    parser.add_argument("--data_path", default= "data",
                        help= "Path to dataset")
    parser.add_argument("--dag_path", default= "src/agentos/workflows",
                        help= "The path to workflow of dag")
    parser.add_argument("--master_addr", default="127.0.0.1:5000",
                        help="Address (IP:port) of the master node Task Scheduler for distributed coordination")
    parser.add_argument("--redis_ip", default="172.17.0.2", 
                        help="IP address of the Redis server for task queue management")
    parser.add_argument("--redis_port", default="6379",
                        help="Port number of the Redis server (default: 6379)")
    parser.add_argument("--strategy", default="mlq", 
                        help="Task submission strategy (choices: mlq, fifo, priority; default: mlq)")
    parser.add_argument("--flask_port", default="5001", 
                        help="HTTP port for running the Flask server (default: 5001)")
    parser.add_argument("--time_pred_model_path",  default="src/agentos/utils/timepred/model", 
                        help="Path to the pre-trained model directory for execution time prediction")
    parser.add_argument("--min_sample4train", default= 10,
                        help="Minimum number of samples for time predict model to train")
    parser.add_argument("--min_sample4incremental", default= 10,
                        help="Minimum number of samples for incremental learning for time predict model")  
    parser.add_argument("--task_exec_time_csv", default= "task_exec_time.csv")
    parser.add_argument("--task_exec_result_jsonl", default= "task_exec_result.jsonl")
    args = parser.parse_args()
    args.tokenizer_path= os.path.join(args.proj_path, args.tokenizer_path)
    args.time_pred_model_path= os.path.join(args.proj_path, args.time_pred_model_path)
    args.task_exec_time_csv= os.path.join(args.proj_path, "src/agentos/results", args.task_exec_time_csv)
    args.task_exec_result_jsonl= os.path.join(args.proj_path, "src/agentos/results", args.task_exec_result_jsonl)    
    return args

args= parse_args()

@app.route('/dag/', methods=['POST'])
def dag():
    data = request.get_json()
    dag_ids = data.get('dag_ids')
    dag_sources = data.get('dag_sources')
    dag_types = data.get('dag_types')
    dag_supplementary_files = data.get('dag_supplementary_files')
    sub_time= data.get('sub_time')
    if not dag_ids or not isinstance(dag_ids, list):
        return {"status": "error", "msg": "dag_ids (list) is required"}, 400

    results = []
    for dag_id, dag_source, dag_type, supplementary_files in zip(dag_ids, dag_sources, dag_types, dag_supplementary_files):
        try:
            # 1. 为本次运行生成唯一的 run_id
            run_id = str(uuid.uuid4())

            # 2. 使用静态 dag_id 加载DAG定义
            dag_folder = os.path.join(args.proj_path, args.dag_path, dag_source, dag_type)
            dag_json_file = os.path.join(dag_folder, 'dag.json')
            if not os.path.exists(dag_json_file):
                raise FileNotFoundError(f"dag.json not found at {dag_json_file}")

            with open(dag_json_file, 'r') as file:
                dag_data = json.load(file)
            
            nodes = dag_data.get('nodes', [])
            # 3. 构造可读、唯一的 task_id
            task2id = {node["task"]: f"{run_id}-{node['task']}" for node in nodes}

            # 4. 状态字典以 run_id 为键
            dag_status_dict[run_id] = {node["task"]: {"status": "unfinished", "start_exec_time": "", "finish_exec_time": "", "arrival_time": "", "leave_time": ""} for node in nodes}
            
            # 5. 传递给后台时，需要同时包含 run_id 和 dag_id
            dag_que.put((run_id, dag_id, dag_source, dag_type, supplementary_files, task2id, sub_time))
            
            # 6. 准备返回给客户端的信息
            results.append({"dag_id": dag_id, "run_id": run_id, "task2id": task2id})
        except Exception as e:
            results.append({"dag_id": dag_id, "error": str(e)})

    return {"status": "success", "msg": "DAGs submission processed.", "data": results}

 
@app.route('/status/', methods=['POST'])
def status():
    data = request.get_json()
    run_id = data.get('run_id') # 客户端现在使用 run_id 查询
    if run_id not in dag_status_dict:
        return jsonify({"status": "error", "msg": "run_id not found, please check the ID", "data": None})
    res = {"status": "success", "msg": "query success", "data": {"dag_status": dict(dag_status_dict[run_id])}}
    return res

#获取dag中task的返回值
@app.route('/get/', methods=['POST'])
def get():
    data = request.get_json()
    dag_id = data["run_id"]
    task_id = data["task_id"]
    func_name = data["func_name"]

    data_status = dag_status_dict[dag_id]
    if data_status[func_name]["status"] == "unfinished":
        res = {"status": "error","msg":"task has not yet been completed","data":None}
        return res
    
    redis_client = redis.Redis(
        host= args.redis_ip,
        port= args.redis_port,
        decode_responses=True
    )
    raw_data_from_redis= redis_client.get(f"result:{task_id}")
    parsed_data= raw_data_from_redis
    if raw_data_from_redis:
        try:
            parsed_data= json.loads(raw_data_from_redis)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode Redis data for task {task_id}, keeping as raw string.")
            pass
    res = {"status": "success","msg":"query success","data":{"task_ret_data":parsed_data}}
    task_result= {"dag_id": dag_id, "task_id": task_id, "func_name": func_name, "status": data_status[func_name]["status"], "data": parsed_data}
    try:
        with open(args.task_exec_result_jsonl, 'a', encoding= 'utf-8') as f:
            json_str= json.dumps(task_result, ensure_ascii= False)
            f.write(json_str + '\n')
        print(f"💠 任务结果数据已成功追加到: {args.task_exec_result_jsonl}")
    except Exception as e:
        print(f"写入文件失败: {e}")
    return res


#释放dag资源
@app.route('/release/', methods=['POST'])
def release():
    data = request.get_json()
    run_id = data.get("run_id")
    dag_id= data.get("dag_id")
    task2id = data["task2id"]
    if not run_id:
        return jsonify({"status": "error", "msg": "run_id not found in request"}), 400

    if run_id not in dag_status_dict:
        return jsonify({"status": "error", "msg": f"run_id '{run_id}' not found, it may have been already released."})
    
    # 1. 正确地获取整个任务状态字典，它本身就是 dag_tasks_status
    dag_tasks_status = dag_status_dict[run_id]
    
    # 2. 然后再正确地遍历这个字典，来检查每个任务的状态
    for task_name, task_info in dag_tasks_status.items():
        if task_info["status"] == "unfinished":
            return {"status": "error", "msg": f"Cannot release. Task '{task_name}' in run '{run_id}' is still 'unfinished'."}

    try:
        # 'a'模式表示追加写入
        with open(args.task_exec_time_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for task_name, task_info in dag_tasks_status.items():
                task_id = task2id.get(task_name) # 通过任务名获取任务ID
                # 准备要写入的一行数据
                row = [
                    dag_id,
                    run_id,
                    task_name,
                    task_id,
                    task_info.get("sub_time", ""),
                    task_info.get("arrival_time", ""),
                    task_info.get("start_exec_time", ""),
                    task_info.get("finish_exec_time", ""),
                    task_info.get("finish_exec_time", "")- task_info.get("start_exec_time", ""),
                    task_info.get("leave_time", ""),
                    task_info.get("finish_exec_time", "")- task_info.get("arrival_time", ""),
                    task_info.get("leave_time", "")- task_info.get("sub_time", ""),
                    task_info.get("scheduler_cost_time", ""),
                    task_info.get("pred_exec_time", ""),
                    task_info.get("pred_cost_time", "")
                ]
                writer.writerow(row)
        print(f"DAG {dag_id} execution times have been successfully logged.")
    except Exception as e:
        print(f"Error logging DAG {dag_id} to CSV: {e}")

    #释放dag中各个task在redis中的返回值
    redis_client = redis.Redis(
        host= args.redis_ip,  
        port= args.redis_port,
        decode_responses=True
    )
    for func_name,task_id in task2id.items():
        redis_client.delete(f"func:{task_id}")
        redis_client.delete(f"result:{task_id}")
    #释放dag context
    payload = {
        "run_id": run_id 
    }
    try: # <--- 新增：为requests增加异常处理
        response = requests.post(f"http://{args.master_addr}/release_context", json=payload)
        response.raise_for_status() # 检查HTTP请求是否成功
        data = response.json()
        if data["status"] == "success":
            res = {"status": "success", "msg": "resources released successfully"}
        else:
            res = {"status": "error", "msg": "failed to release resources on master"}
    except requests.exceptions.RequestException as e:
        res = {"status": "error", "msg": f"Failed to connect to master for resource release: {e}"}
    del dag_status_dict[run_id]
    return res  
 

if __name__ == '__main__':

    csv_head = ['dag_id', 'run_id', 'task_name', 'task_id', 'sub_time', 'arrival_time', 'start_exec_time', 'finish_exec_time', 'exec_time', 'leave_time', 'completion_time', 'response_time', 'scheduler_cost_time', 'pred_exec_time', 'pred_cost_time']
    if not os.path.exists(args.task_exec_time_csv):
        with open(args.task_exec_time_csv, 'w', newline='') as f:
            writer= csv.writer(f)
            writer.writerow(csv_head)
        print(f"'{args.task_exec_time_csv}' created with headers.")
    if not os.path.exists(args.task_exec_result_jsonl):
        with open(args.task_exec_result_jsonl, 'w', newline='') as f:
            pass
        print(f"'{args.task_exec_result_jsonl}' created with headers.")
    
    if args.strategy == "heft":
        dag_manager_pro = multiprocessing.Process(target=dag_manager_heft,
                                                args=(args, dag_que, dag_status_dict))
    elif args.strategy == "cpop":
        dag_manager_pro = multiprocessing.Process(target=dag_manager_cpop,
                                                args=(args, dag_que, dag_status_dict))
    elif args.strategy == "mlq":
        dag_manager_pro = multiprocessing.Process(target=dag_manager_mlq, 
                                                args= (args, dag_que, dag_status_dict))
    elif args.strategy == "fcfs":
        dag_manager_pro = multiprocessing.Process(target=dag_manager_fcfs, 
                                                args= (args, dag_que, dag_status_dict))
    elif args.strategy == "ltf":
        dag_manager_pro = multiprocessing.Process(target=dag_manager_ltf,
                                                args= (args, dag_que, dag_status_dict))
    elif args.strategy == "peft":
        dag_manager_pro= multiprocessing.Process(target=dag_manager_peft,
                                                args= (args, dag_que, dag_status_dict))
    elif args.strategy == "ftc":
        dag_manager_pro= multiprocessing.Process(target=dag_manager_ftc,
                                                args= (args, dag_que, dag_status_dict))

    dag_manager_pro.start()
    print(f"😊 准备启动Flask应用，端口：{args.flask_port}")
    app.run(port=args.flask_port)