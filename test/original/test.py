import unittest
import requests
import websocket
import json
import time
import argparse
import os
from typing import Any, Dict, Optional

# -----------------------------
# 配置：获取 base_url
# -----------------------------

def get_base_url() -> str:
    """优先级：命令行参数 > 玄境变量 > 默认值"""
    # 注意：我们会在 main 中解析 args，所以这里用全局变量传递
    if hasattr(get_base_url, 'base_url_override'):
        return get_base_url.base_url_override
    return os.getenv("WORKFLOW_BASE_URL", "http://localhost:8000")

def get_ws_base_url() -> str:
    """从 HTTP URL 推导出 WebSocket URL（http -> ws, https -> wss）"""
    http_url = get_base_url()
    return http_url.replace("http://", "ws://").replace("https://", "wss://")

# -----------------------------
# 工具函数（使用动态 base_url）
# -----------------------------

def create_workflow() -> Optional[str]:
    url = f"{get_base_url()}/create_workflow"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return response.json().get("workflow_id")
        else:
            print(f"[ERROR] create_workflow failed: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    return None

def add_task(workflow_id: str, task_name: str) -> Optional[str]:
    url = f"{get_base_url()}/add_task"
    data = {
        'workflow_id': workflow_id,
        'task_type': 'code',
        'task_name': task_name,
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json().get("task_id")
        else:
            print(f"[ERROR] add_task failed: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    return None

def save_task(workflow_id: str, task_id: str, code_str: str,
              task_input: Dict[Any, Any], task_output: Dict[Any, Any], resources: Dict[str, float]):
    url = f"{get_base_url()}/save_task"
    data = {
        'workflow_id': workflow_id,
        'task_id': task_id,
        'code_str': code_str,
        'task_input': task_input,
        'task_output': task_output,
        'resources': resources,
    }
    try:
        response = requests.post(url, json=data)
        assert response.status_code == 200, f"save_task failed: {response.status_code}, {response.text}"
    except Exception as e:
        print(f"[ERROR] save_task failed: {e}")
        raise

def add_edge(workflow_id: str, source_task_id: str, target_task_id: str):
    url = f"{get_base_url()}/add_edge"
    data = {
        'workflow_id': workflow_id,
        'source_task_id': source_task_id,
        'target_task_id': target_task_id,
    }
    try:
        response = requests.post(url, json=data)
        assert response.status_code == 200, f"add_edge failed: {response.status_code}, {response.text}"
    except Exception as e:
        print(f"[ERROR] add_edge failed: {e}")
        raise

def run_workflow(workflow_id: str):
    url = f"{get_base_url()}/run_workflow"
    data = {'workflow_id': workflow_id}
    try:
        response = requests.post(url, json=data)
        assert response.status_code == 200, f"run_workflow failed: {response.status_code}, {response.text}"
    except Exception as e:
        print(f"[ERROR] run_workflow failed: {e}")
        raise

def get_workflow_result_sync(workflow_id: str, timeout: float = 15.0) -> Optional[Dict[Any, Any]]:
    """同步获取 WebSocket 结果"""
    result = {}

    def on_message(ws, message):
        try:
            msg_data = json.loads(message)
            result['data'] = msg_data
            ws.close()
        except Exception as e:
            print(f"Failed to parse message: {e}")

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_open(ws):
        pass

    ws_url = f"{get_ws_base_url()}/get_workflow_res/{workflow_id}"
    print(f"Connecting to WebSocket: {ws_url}")

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    import threading
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    start_time = time.time()
    while 'data' not in result and time.time() - start_time < timeout:
        time.sleep(0.1)

    ws.close()
    return result.get('data')


# -----------------------------
# 单元测试类
# -----------------------------

class TestWorkflowSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"Using base URL: {get_base_url()}")
        print("Creating workflow...")
        cls.workflow_id = create_workflow()
        assert cls.workflow_id is not None, "Failed to create workflow"
        print(f"Created workflow_id: {cls.workflow_id}")

    def setUp(self):
        self.task_id1 = None
        self.task_id2 = None

    def test_full_workflow_execution(self):
        workflow_id = self.__class__.workflow_id

        # 添加任务
        self.task_id1 = add_task(workflow_id, "task1")
        self.task_id2 = add_task(workflow_id, "task2")
        self.assertIsNotNone(self.task_id1, "Failed to add task1")
        self.assertIsNotNone(self.task_id2, "Failed to add task2")

        # Task1 代码
        code_str1 = '''
from datetime import datetime
import time

def task1(params):
    task_input = params.get("task1_input")
    time.sleep(1)
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    result = task_input + time_str
    return {"task1_output": result}
'''

        task_input1 = {
            "input_params": {
                "1": {
                    "key": "task1_input",
                    "input_schema": "from_user",
                    "data_type": "str",
                    "value": "这是task1的输入",
                }
            }
        }

        task_output1 = {
            "output_params": {
                "1": {
                    "key": "task1_output",
                    "data_type": "str"
                }
            }
        }

        resources1 = {"cpu": 1, "cpu_mem": 123, "gpu": 1, "gpu_mem": 123}
        save_task(workflow_id, self.task_id1, code_str1, task_input1, task_output1, resources1)

        # Task2 代码（依赖 task1）
        code_str2 = '''
def task2(params):
    task_input = params.get("task2_input")
    time.sleep(1)
    return {"task2_output": task_input + " processed at backend"}
'''

        task_input2 = {
            "input_params": {
                "1": {
                    "key": "task2_input",
                    "input_schema": "from_task",
                    "data_type": "str",
                    "value": f"{self.task_id1}.output.task1_output"
                }
            }
        }

        task_output2 = {
            "output_params": {
                "1": {
                    "key": "task2_output",
                    "data_type": "str"
                }
            }
        }

        resources2 = {"cpu": 1, "cpu_mem": 123, "gpu": 0, "gpu_mem": 0}
        save_task(workflow_id, self.task_id2, code_str2, task_input2, task_output2, resources2)

        # 建立连接
        add_edge(workflow_id, self.task_id1, self.task_id2)

        # 执行
        run_workflow(workflow_id)

        # 获取结果
        print("Waiting for workflow result...")
        result = get_workflow_result_sync(workflow_id, timeout=20.0)
        self.assertIsNotNone(result, "No result received from workflow")
        self.assertEqual(result.get("status"), "completed", "Workflow did not complete")

        outputs = result.get("outputs", {})
        self.assertIn(self.task_id2, outputs)
        final_output = outputs[self.task_id2].get("task2_output", "")
        self.assertIn("这是task1的输入", final_output)
        self.assertIn("processed at backend", final_output)
        print(f"✅ Final output: {final_output}")

    @classmethod
    def tearDownClass(cls):
        print(f"Test completed. Workflow ID: {cls.workflow_id}")


# -----------------------------
# 主程序入口：支持命令行参数
# -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run workflow integration tests.")
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Base URL of the workflow server (e.g., http://localhost:8000 or http://myserver:8000)'
    )
    args, unknown = parser.parse_known_args()

    # 动态修改 get_base_url 的行为
    if args.base_url:
        get_base_url.base_url_override = args.base_url.rstrip('/')
        print(f"📌 Overriding base URL: {get_base_url.base_url_override}")

    # 将未知参数传回 unittest
    unittest.main(argv=[__file__] + unknown, verbosity=2)