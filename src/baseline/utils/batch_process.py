#!/usr/bin/env python3
import json
import argparse
import sys
import os
import time

# 从dispatch_task.py导入需要的函数和类，不要导入args
from dispatch_task import submit_tasks, MasterApiClient

def load_data_from_jsonl(file_path):
    """从JSONL文件加载数据"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def batch_process(jsonl_file, dag_type=None, batch_size=5, wait_for_completion=False, poll_interval=5, timeout=36000, master_addr="172.17.0.1:5002"):
    """批量处理JSONL文件中的任务
    
    Args:
        jsonl_file: JSONL文件路径
        dag_type: 筛选特定类型的任务（可选）
        batch_size: 每批次处理的任务数量
        wait_for_completion: 是否等待任务完成
        poll_interval: 轮询间隔（秒）
        timeout: 超时时间（秒）
        master_addr: Master API地址
    """
    # 创建客户端
    client = MasterApiClient(master_addr)
    
    # 加载数据
    data = load_data_from_jsonl(jsonl_file)
    print(f"加载了 {len(data)} 条任务数据")
    
    # 筛选特定类型的任务（如果指定）
    if dag_type:
        data = [item for item in data if item["dag_type"] == dag_type]
        print(f"筛选出 {len(data)} 条 {dag_type} 类型的任务")
    
    # 所有已提交的DAG
    all_submitted_dags = {}
    start_time = time.time()
    
    # 批量处理
    total_batches = (len(data) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        # 提取该批次的dag_ids和其他信息
        dag_ids = [item["dag_id"] for item in batch_data]
        dag_sources = [item["dag_source"] for item in batch_data]
        dag_types = [item["dag_type"] for item in batch_data]
        dag_supplementary_files = [item.get("dag_supplementary_files", []) for item in batch_data]
        
        # 准备提交的数据
        submission_payload = {
            "dag_ids": dag_ids,
            "dag_sources": dag_sources,
            "dag_types": dag_types,
            "dag_supplementary_files": dag_supplementary_files
        }
        
        print(f"批次 {batch_idx+1}/{total_batches}, 处理 {len(dag_ids)} 条任务")
        print(f"DAG IDs: {dag_ids}")
        
        # 提交任务
        try:
            scheduled_dags = submit_tasks(submission_payload, master_addr)
            print(f"批次 {batch_idx+1} 提交成功，共 {len(scheduled_dags)} 个DAG")
            
            # 添加到全部已提交DAG
            for dag in scheduled_dags:
                all_submitted_dags[dag['uuid']] = dag['dag_id']
            
        except Exception as e:
            print(f"批次 {batch_idx+1} 提交失败: {e}")
    
    # 如果不需要等待任务完成，直接返回
    if not wait_for_completion or not all_submitted_dags:
        print(f"\n提交完成，共 {len(all_submitted_dags)} 个DAG。")
        return
    
    # 等待所有任务完成
    print(f"\n⏱️  开始轮询 {len(all_submitted_dags)} 个DAG的状态（每 {poll_interval} 秒一次）...")
    running_dags = all_submitted_dags.copy()
    
    while running_dags and (time.time() - start_time < timeout):
        finished_this_poll = set()
        print("\n--- 本轮状态查询 ---")

        for dag_uuid, dag_id in running_dags.items():
            status_data = client.check_dag_status(dag_uuid)
            
            if status_data:
                status = status_data.get("status")
                if status == "Finished":
                    print(f"✅ DAG '{dag_id}' (UUID: {dag_uuid}) 已完成！")
                    
                    # 立刻获取并打印结果
                    result_data = client.get_final_result(dag_uuid)
                    print(f"--- 最终输出 for DAG {dag_uuid} ---")
                    if result_data and "final_result" in result_data:
                        pretty_result = json.dumps(result_data["final_result"], indent=4, ensure_ascii=False)
                        print(pretty_result)
                    else:
                        print("未能获取到最终结果。")
                    print("--------------------------------" + "-" * len(dag_uuid))
                    
                    finished_this_poll.add(dag_uuid)
                else:
                    # 优化输出
                    completed = status_data.get("completed_tasks", 0)
                    total = status_data.get("total_tasks", "?")
                    print(f"  -> 仍在运行: DAG '{dag_id}' (进度: {completed}/{total})")
            else:
                print(f"  -> 等待服务器响应: DAG '{dag_uuid}'")

        if finished_this_poll:
            for uuid in finished_this_poll:
                running_dags.pop(uuid)
        
        if running_dags:
            time.sleep(poll_interval)

    if not running_dags:
        print("\n" + "="*50)
        print("✅✅✅ 所有DAG任务均已成功执行！正在打印最终调度报告...")
        print("="*50)
        report_content = client.trigger_report()
        if report_content:
            print(report_content)
    else:
        print(f"\n⏰ 等待超时！仍有 {len(running_dags)} 个任务未在 {timeout} 秒内完成。")

    print(f"\n🎉 批处理脚本执行完毕，总耗时: {time.time() - start_time:.2f} 秒。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量处理JSONL文件中的任务')
    parser.add_argument('--file', type=str, required=True, help='JSONL文件路径')
    parser.add_argument('--type', type=str, help='筛选特定类型的任务（可选）')
    parser.add_argument('--batch', type=int, default=5, help='每批次处理的任务数量（默认5）')
    parser.add_argument('--wait', action='store_true', help='等待任务完成并显示结果')
    parser.add_argument('--poll', type=int, default=5, help='轮询间隔（秒）（默认5秒）')
    parser.add_argument('--timeout', type=int, default=36000, help='超时时间（秒）（默认10小时）')
    parser.add_argument('--master_addr', default="172.17.0.1:5002", help='Master API地址')
    
    args_local = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args_local.file):
        print(f"错误: 文件 {args_local.file} 不存在")
        sys.exit(1)
    
    batch_process(
        args_local.file, 
        args_local.type, 
        args_local.batch,
        args_local.wait, 
        args_local.poll, 
        args_local.timeout,
        args_local.master_addr
    ) 