import time
import requests
import json
import argparse
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------

def clear_console():
    """清空终端屏幕，以实现动态刷新效果。"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_dags_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从 .jsonl 文件加载DAG定义。每行一个JSON对象。
    """
    print(f"\n📂 正在从文件加载DAG定义: {file_path}")
    dag_definitions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        dag_definitions.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"⚠️ 警告: 跳过无法解析的行: {line.strip()}")
        print(f"  -> 成功加载 {len(dag_definitions)} 个DAG定义。")
        return dag_definitions
    except FileNotFoundError:
        print(f"❌ 错误: 查询文件未找到! 路径: {file_path}")
        return []

# --------------------------------------------------------------------------
# 客户端API类 (这是你原有的、功能正确的API客户端)
# --------------------------------------------------------------------------
class MasterApiClient:
    """一个用于与 master_api.py (Flask API) 交互的客户端。"""
    def __init__(self, master_addr: str):
        self.base_url = f"http://{master_addr}"
        print(f"✅ 客户端已初始化，目标 Master API: {self.base_url}")

    def submit_dags(self, payload: Dict) -> List[Dict[str, Any]]:
        """向 Master 节点提交一个批次的 DAG 任务。"""
        url = f"{self.base_url}/submit_dag"
        print(f"\n🚀 正在向 {url} 提交 {len(payload['dag_ids'])} 个DAG任务...")
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            res = response.json()
            submitted_dags = res.get("submitted", [])
            for dag_info in submitted_dags:
                if "error" in dag_info:
                    print(f"  -> ❌ 提交失败: DAG ID '{dag_info.get('dag_id')}', 原因: {dag_info.get('error')}")
                else:
                    print(f"  -> ✅ 提交成功: DAG ID '{dag_info['dag_id']}', 实例 UUID '{dag_info['uuid']}'")
            return [info for info in submitted_dags if "error" not in info]
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP 请求失败: {e}")
            return []

    def check_dag_status(self, dag_uuid: str) -> Optional[Dict[str, Any]]:
        """查询单个 DAG 的当前状态。"""
        try:
            response = requests.get(f"{self.base_url}/dag_status/{dag_uuid}", timeout=5)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            return None

    def get_final_result(self, dag_uuid: str) -> Optional[Dict[str, Any]]:
        """获取单个已完成 DAG 的最终结果。"""
        try:
            response = requests.get(f"{self.base_url}/get_final_result/{dag_uuid}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  -> ⚠️ 获取结果失败 for {dag_uuid}: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取结果 {dag_uuid} 时网络连接失败: {e}")
            return None

# --------------------------------------------------------------------------
# 新增的监控与流程编排函数
# --------------------------------------------------------------------------
def monitor_and_process_dags(client: MasterApiClient, running_dags: Dict[str, str], poll_interval: int):
    """
    监控所有正在运行的DAG实例，并在完成后获取和打印结果。
    """
    start_time = time.time()
    total_dags = len(running_dags)
    # 用于存储每个任务的最新状态文本，以便在仪表盘上显示
    dag_status_text = {uuid: "Pending..." for uuid in running_dags.keys()}

    while running_dags:
        clear_console()
        print("=" * 20 + " 实时任务监控仪表盘 " + "=" * 20)
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 运行中/总实例: {len(running_dags)}/{total_dags}")
        print("-" * 65)

        finished_this_poll = set()

        # 遍历所有仍在运行的任务，更新并打印它们的状态
        for uuid, dag_id in running_dags.items():
            status_info = client.check_dag_status(uuid)
            
            if status_info:
                status = status_info.get("status", "Unknown")
                process= "0/1"
                if status== "Finished":
                    process= "1/1"
                    finished_this_poll.add(uuid)
                dag_status_text[uuid] = f"状态: {status}, 进度: {process}"
            else:
                dag_status_text[uuid] = "等待服务器响应..."

            # 打印格式化的状态行
            print(f"  DAG: {dag_id:<38} | UUID: {uuid:<38} | {dag_status_text[uuid]}")
        
        print("-" * 65)

        # 处理本轮已完成的任务
        if finished_this_poll:
            print("\n发现已完成的任务，正在处理结果...")
            for uuid in finished_this_poll:
                dag_id_to_process = running_dags.pop(uuid) # 从运行列表中移除
                
                print(f"\n--- 最终输出 for DAG '{dag_id_to_process}' (UUID: {uuid}) ---")
                result_data = client.get_final_result(uuid)
                if result_data:
                    pretty_result = json.dumps(result_data, indent=4, ensure_ascii=False)
                    print(pretty_result)
                else:
                    print("未能获取到最终结果。")
                print("-" * 70)

        # 如果还有任务在运行，则等待指定间隔后再次轮询
        if running_dags:
            print(f"\n下一次更新在 {poll_interval} 秒后...")
            time.sleep(poll_interval)

    total_time = time.time() - start_time
    print(f"\n✅✅✅ 所有DAG任务均已执行完毕！总耗时: {total_time:.2f} 秒。")

# --------------------------------------------------------------------------
# 主程序执行块
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="向 AgentOS-Host 提交任务的客户端。")
    parser.add_argument("--master_addr", default="localhost:5002", help="Master API 的地址 (IP:port)。")
    parser.add_argument("--proj_path", default="/root/workspace/d23oa7cp420c73acue30/AgentOS", help="项目根目录的绝对路径。")
    parser.add_argument("--query_file", default="data/tbench/tbench_query.jsonl", help="相对于项目根目录的任务查询文件路径。")
    parser.add_argument("--poll_interval", type=int, default=10, help="状态查询的间隔时间（秒）。")
    args = parser.parse_args()

    # 1. 从文件自动加载DAG定义
    query_file_full_path = os.path.join(args.proj_path, args.query_file)
    dag_definitions = load_dags_from_jsonl(query_file_full_path)

    if not dag_definitions:
        print("\n未能从文件加载任何DAG，程序退出。")
        sys.exit(1)

    # 2. 根据加载的数据，动态构建提交的payload
    submission_payload = {
        "dag_ids": [d["dag_id"] for d in dag_definitions],
        "dag_sources": [d["dag_source"] for d in dag_definitions],
        "dag_types": [d["dag_type"] for d in dag_definitions],
        "dag_supplementary_files": [d["dag_supplementary_files"] for d in dag_definitions],
        "sub_time": time.time()
    }

    # dag_ids = [
    #     "0c4f9fd8-01c4-4fbe-b933-3570a3cd771a",
    #     # "0d2ec70b-46f3-4c95-8172-c383c7539a94",
    #     # "6b4156de-bcb8-4146-a730-f699e220004e",
    #     # "0a33f7a3-5cfa-42c7-8cab-19260908720b"
    # ]
    # submission_payload = {
    #     "dag_ids": dag_ids,
    #     "dag_sources": ["openagi"] * len(dag_ids),
    #     "dag_types": [
    #         "document_qa",
    #         # "image_captioning_complex",
    #         # "multimodal_vqa_complex",
    #         # "text_processing_multilingual"
    #         ], # , "vision", "speech", "file", "reason"
    #     "dag_supplementary_files": [
    #         ["context.txt", "question.txt", "questions.txt"],
    #         # ["images/27.jpg", "images/28.jpg", "images/29.jpg", "images/3.jpg", "images/30.jpg", "images/31.jpg", "images/32.jpg", "images/33.jpg", "images/34.jpg", "images/35.jpg", "images/36.jpg", "images/37.jpg", "images/38.jpg", "images/39.jpg", "images/4.jpg", "images/40.jpg", "images/41.jpg", "images/42.jpg", "images/43.jpg", "images/44.jpg", "question.txt"],
    #         # ["images/81.jpg", "images/82.jpg", "images/83.jpg", "images/84.jpg", "images/85.jpg", "images/86.jpg", "images/87.jpg", "images/88.jpg", "images/89.jpg", "images/9.jpg", "images/90.jpg", "images/91.jpg", "images/92.jpg", "images/93.jpg", "images/94.jpg", "images/95.jpg", "images/96.jpg", "images/97.jpg", "images/98.jpg", "images/99.jpg", "question.txt"],
    #         # ["question.txt", "text.txt"]
    #     ],
    #     "sub_time": time.time()
    # }

    # dag_ids = [
    #     "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3",
    #     "df6561b2-7ee5-4540-baab-5095f742716a",
    #     "32102e3e-d12a-4209-9163-7b3a104efe5d",
    #     "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
    # ]
    # submission_payload = {
    #     "dag_ids": dag_ids,
    #     "dag_sources": ["gaia"] * len(dag_ids),
    #     "dag_types": ["speech", "vision", "file", "reason"], # , "vision", "speech", "file", "reason"
    #     "dag_supplementary_files": [
    #         ["99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"],
    #         ["df6561b2-7ee5-4540-baab-5095f742716a.png"],
    #         ["32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx"],
    #         [],
    #     ],
    #     "sub_time": time.time()
    # }

    dag_ids = [
        "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
        "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
        "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
        "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
        "df6561b2-7ee5-4540-baab-5095f742716a",
        "df6561b2-7ee5-4540-baab-5095f742716a",
        "df6561b2-7ee5-4540-baab-5095f742716a",
        "df6561b2-7ee5-4540-baab-5095f742716a",
    ]
    submission_payload = {
        "dag_ids": dag_ids,
        "dag_sources": ["gaia"] * len(dag_ids),
        "dag_types": ["reason", "reason", "reason", "reason", "vision", "vision", "vision", "vision"], # , "vision", "speech", "file", "reason"
        "dag_supplementary_files": [
            [],
            [],
            [],
            [],
            ["df6561b2-7ee5-4540-baab-5095f742716a.png"],
            ["df6561b2-7ee5-4540-baab-5095f742716a.png"],
            ["df6561b2-7ee5-4540-baab-5095f742716a.png"],
            ["df6561b2-7ee5-4540-baab-5095f742716a.png"],
        ],
        "sub_time": time.time()
    }

    # 3. 初始化客户端并提交任务
    client = MasterApiClient(args.master_addr)
    start_time = time.time()
    scheduled_dags = client.submit_dags(submission_payload)
    
    # 4. 如果有任务成功提交，则进入监控流程
    running_dags = {dag['uuid']: dag['dag_id'] for dag in scheduled_dags}
    if running_dags:
        monitor_and_process_dags(client, running_dags, poll_interval=args.poll_interval)
    else:
        print("\n没有成功提交的任务，程序退出。")

    print(f"\n🎉 客户端脚本执行完毕，总耗时: {time.time() - start_time:.2f} 秒。")