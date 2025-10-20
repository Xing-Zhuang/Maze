from sre_parse import SUCCESS
from typing import Any,List
from maze.core.workflow.task import CodeTask


class Workflow():
    def __init__(self, id: str):
        self.id: str = id
        self.tasks: dict[str, Any] = {}
        self.edges: dict[str, str] = {}
        self.in_degree: dict[str, int]={}
        self.successors_id: dict[str, List] =  {}
        
    def add_task(self,task_id,task:CodeTask):
        self.tasks[task_id] = task
        self.in_degree[task_id] = 0
        self.successors_id[task_id] = []

    def get_task(self,task_id:str):
        return self.tasks[task_id]

    def add_edge(self,source_task_id:str,target_task_id:str):
        self.edges[source_task_id] = target_task_id
        self.in_degree[target_task_id] = self.in_degree[target_task_id] + 1
        self.successors_id[source_task_id].append(target_task_id)

    def get_start_task(self):
        start_task = []
        for task_id in self.tasks:
            if self.in_degree[task_id] == 0:
                start_task.append(self.tasks[task_id])
        return start_task        
    
    def get_total_task_num(self):
        return len(self.tasks)

    def finish_task(self,task_id:str):
        new_ready_tasks = []

        for task_id in self.successors_id[task_id]:
            self.in_degree[task_id] = self.in_degree[task_id] - 1
            if self.in_degree[task_id] == 0:
                new_ready_tasks.append(self.tasks[task_id])

        return new_ready_tasks