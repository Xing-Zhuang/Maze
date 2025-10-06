import threading
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# --- 新增：定义结构化的任务记录对象 ---
@dataclass
class TaskExecutionRecord:
    task_info: Dict[str, Any]
    status: str = "received"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None

class TaskStatusManager:
    """
    (V2 - 运行时数据中心)
    管理任务的状态、信息和执行时间记录。
    """
    def __init__(self)-> None:
        # --- 核心修改：使用新的数据结构 ---
        self.task_records: Dict[Tuple[str, str], TaskExecutionRecord] = {}
        self.run_records: Dict[str, Dict] = {}
        self.task2nodeId: Dict[Tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def add_task(self, task_info: dict)-> None:
        run_id = task_info["run_id"]        
        task_id = task_info["task_id"]
        key = (run_id, task_id)
        with self._lock:
            # 如果记录已存在，则不重复创建，只确保状态是 received
            if key in self.task_records:
                self.task_records[key].status = "received"
            else:
                self.task_records[key] = TaskExecutionRecord(task_info=task_info)

    def set_status(self, run_id: str, task_id: str, status: str, err_msg: str= "")-> None:
        key = (run_id, task_id)
        with self._lock:
            if key not in self.task_records:
                raise KeyError(f"Task {key} not registered, cannot set status")
            
            self.task_records[key].status = status
            if status == "failed":
                self.task_records[key].error_message = err_msg

    def get_task_info(self, run_id: str, task_id: str) -> Dict[str, object]:
        key = (run_id, task_id)
        with self._lock:
            record = self.task_records.get(key)
            return record.task_info if record else {}

    def get_status(self, run_id: str, task_id: str)-> str:
        key = (run_id, task_id)
        with self._lock:
            record = self.task_records.get(key)
            return record.status if record else "unknown"

    # --- 新增：记录任务完成时间的方法 ---
    def record_task_completion(self, run_id: str, task_id: str, timing_data: dict):
        key = (run_id, task_id)
        with self._lock:
            if key in self.task_records:
                record = self.task_records[key]
                record.start_time = timing_data.get("start_time")
                record.end_time = timing_data.get("end_time")
                if record.start_time and record.end_time:
                    record.duration = record.end_time - record.start_time

    # --- 新增：记录工作流完成时间的方法 ---
    def record_run_completion(self, run_id: str, total_duration: float):
        with self._lock:
            if run_id not in self.run_records:
                self.run_records[run_id] = {}
            self.run_records[run_id]['total_duration'] = total_duration
            self.run_records[run_id]['status'] = 'completed'

    def cleanup_run(self, run_id: str):
        with self._lock:
            keys_to_delete = [key for key in self.task_records if key[0] == run_id]
            for key in keys_to_delete:
                self.task_records.pop(key, None)
                self.task2nodeId.pop(key, None)
            self.run_records.pop(run_id, None)

    # ... (set_selected_node, get_selected_node 等其他方法保持不变) ...
    def set_selected_node(self, run_id: str, task_id: str, node_id: str)-> None:
        key= (run_id, task_id)
        with self._lock:
            self.task2nodeId[key]= node_id

    def get_selected_node(self, run_id: str, task_id: str)-> str:
        key= (run_id, task_id)
        with self._lock:
            return self.task2nodeId.get(key, None)