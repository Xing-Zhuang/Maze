from typing import List
import threading

class TasksView():
    def __init__(self):
        self.lock = threading.Lock() #TasksView对象由submit线程和monitor线程共同操作，需要线程安全
        self.tasks = {}
        self.ref_to_taskid = {}
    
    def add_new_task(self, task:dict):
        with self.lock:
            self.tasks[task['task_id']] = {
                "status":"ready",  #任务状态

                "oeject_ref":None, #输出引用(ray)
                "result":None,     #任务结果
                
                "node_id":None,    #任务所在机器ID"
                "gpu_id":None,     #任务所在GPU ID"

                "detail":task,     #任务详情
            }
            """
            #任务详情
            {
                "workflow_id":self.workflow_id,
                "task_id":self.task_id,
                "task_input":self.task_input,
                "task_output":self.task_output,
                "code_str":self.code_str,
                "resources":self.resources
            }

            """
    
    def get_result(self,key):
        '''
        获取任务输出
        key = f"{task_id1}.output.task1_output"
        '''
        task_id = key.split(".")[0]
        output_key = key.split(".")[2]
        return self.tasks[task_id]['result'][output_key]
        
    def set_task_running(self, task_id:str, oeject_ref,choosed_node):
        '''
        设置任务为运行状态
        '''
        with self.lock:
            self.tasks[task_id]['status'] = "running"
            self.tasks[task_id]['oeject_ref'] = oeject_ref
            self.tasks[task_id]['node_id'] = choosed_node['node_id']
            if 'gpu_id' in choosed_node:
                self.tasks[task_id]['gpu_id'] = choosed_node['gpu_id']

            self.ref_to_taskid[oeject_ref] = task_id
          
    def get_running_tasks_ref(self):
        with self.lock:
            ref_list = []
            for task in self.tasks.values():
                if task['status'] == "running":
                    ref_list.append(task['oeject_ref'])
            return ref_list

    def set_task_finished(self, refs:List,results:List):
        with self.lock:
            for ref,result in zip(refs,results):
                task_id = self.ref_to_taskid[ref]
                self.tasks[task_id]['status'] = "finished"
                self.tasks[task_id]['result'] = result

    def get_task_by_ref(self, refs:List):
        with self.lock:
            task_list = []
            for ref in refs:
                task_id = self.ref_to_taskid[ref]
                task_list.append(self.tasks[task_id])
                
            return task_list