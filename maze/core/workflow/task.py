 


from typing import Any


class CodeTask():
    def __init__(self,workflow_id:str,task_id:str):
        self.workflow_id = workflow_id
        self.task_id = task_id
        self.status = "pending" 
    
    def save_task(self,task_input, task_output, code_str, resources):
        self.task_input=task_input
        self.task_output=task_output
        self.code_str=code_str
        self.resources=resources
    
    def to_json(self) -> dict[str, Any]:
        return {
            "workflow_id":self.workflow_id,
            "task_id":self.task_id,
            "task_input":self.task_input,
            "task_output":self.task_output,
            "code_str":self.code_str,
            "resources":self.resources
        }

