import ast
import inspect
import base64
import argparse
import json
import time
import traceback
import networkx as nx
from typing import Dict, Any, Optional

# 导入 AgentScope 的核心组件
from agentscope.agents import AgentBase
from agentscope.message import Msg

# --- 1. 基类定义 (通用 Actor + DAG 执行引擎) ---
class BaseActorAgent(AgentBase):
    """
    一个实现了 Actor 模型和通用 DAG 执行引擎的 AgentScope 基类。
    子类只需定义 self.tools 和 self.workflow 即可。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id)
        self.id= id
        self.workflow_type= workflow_type
        self.tools: Dict[str, callable] = {}
        self.workflow: Dict[str, Any] = {}
        print(f"✅ Actor基类 '{self.id}' 已初始化，内部队列和工作循环已启动。")

    def reply(self, x: Msg)-> Msg:
        """
        通用的、使用 networkx 实现的 DAG 执行逻辑。
        """
        start_exec_time = time.time()
        task_info = ast.literal_eval(x.content)
        uuid = task_info.get("uuid")
        execution_result = ""
        try:
            if not self.tools or not self.workflow:
                raise ValueError("子类必须定义 self.tools 和 self.workflow 属性。")

            exec_graph = nx.DiGraph([(e["source"], e["target"]) for e in self.workflow["edges"]])
            
            # 初始化上下文，这是工作流的“内存”
            arg_src = task_info.get("arg_src", {})
            workflow_context = {**arg_src} # 将arg_src解包放入
            if "args" in arg_src and arg_src.get("args"):
                workflow_context["args"] = argparse.Namespace(**json.loads(arg_src["args"]))
            if "supplementary_files" in arg_src and arg_src.get("supplementary_files"):
                 workflow_context["supplementary_files"] = {k: base64.b64decode(v) for k, v in arg_src["supplementary_files"].items()}
            # 串行执行 DAG
            while exec_graph.number_of_nodes() > 0:
                ready_tasks = [node for node, degree in exec_graph.in_degree() if degree == 0]
                if not ready_tasks: raise Exception("DAG 中存在环！")
                
                task_name_to_run = ready_tasks[0]
                task_func = self.tools[task_name_to_run]
                sig = inspect.signature(task_func)
                kwargs = {key: workflow_context[key] for key in sig.parameters if key in workflow_context}
                # 在独立线程中执行同步任务，避免阻塞
                result= task_func(**kwargs) # 直接调用同步函数
                if isinstance(result, dict): workflow_context.update(result)
                workflow_context[task_name_to_run] = result
                exec_graph.remove_node(task_name_to_run)

            # 获取最终结果
            final_graph = nx.DiGraph([(e["source"], e["target"]) for e in self.workflow["edges"]])
            sink_nodes = [node for node, degree in final_graph.out_degree() if degree == 0]
            final_node_name = sink_nodes[0]
            final_result = workflow_context.get(final_node_name)
            execution_result = json.dumps(final_result, indent=2, ensure_ascii=False) if isinstance(final_result, dict) else str(final_result)
            
        except Exception:
            execution_result = f"EXECUTION_ERROR: {traceback.format_exc()}"
            print(f"❌ [{self.name}] 工作流 {uuid} 执行失败:\n{execution_result}")
        end_exec_time = time.time()
        
        # 构造最终的响应内容
        response_content = {
            "dag_id": task_info.get("dag_id"), "uuid": uuid,
            "sub_time": task_info.get("sub_time"), "result": execution_result,
            "start_time": start_exec_time, "end_time": end_exec_time,
            "arrival_time": task_info.get("arrival_time")
        }
        return Msg(name=self.name, role="assistant", content= str(response_content))

# --- 2. 具体实现类 (无需改动) ---
class GAIA_File_Process_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.gaia.file.task import task1_file_process, \
            task2_llm_process_qwen, task3_llm_process_deepseek, task4_fuse_llm_answer
        self.workflow_type= workflow_type
        self.tools = {
            "task1_file_process": task1_file_process,
            "task2_llm_process_qwen": task2_llm_process_qwen,
            "task3_llm_process_deepseek": task3_llm_process_deepseek,
            "task4_fuse_llm_answer": task4_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_file_process", "target": "task2_llm_process_qwen"},
                {"source": "task1_file_process", "target": "task3_llm_process_deepseek"},
                {"source": "task2_llm_process_qwen", "target": "task4_fuse_llm_answer"},
                {"source": "task3_llm_process_deepseek", "target": "task4_fuse_llm_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


class GAIA_Reason_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.gaia.reason.task import task1_obtain_content, \
            task2_llm_process_qwen, task3_llm_process_deepseek, task4_fuse_llm_answer              
        self.workflow_type= workflow_type
        self.tools = {
            "task1_obtain_content": task1_obtain_content,
            "task2_llm_process_qwen": task2_llm_process_qwen,
            "task3_llm_process_deepseek": task3_llm_process_deepseek,
            "task4_fuse_llm_answer": task4_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_obtain_content", "target": "task2_llm_process_qwen"},
                {"source": "task1_obtain_content", "target": "task3_llm_process_deepseek"},
                {"source": "task2_llm_process_qwen", "target": "task4_fuse_llm_answer"},
                {"source": "task3_llm_process_deepseek", "target": "task4_fuse_llm_answer"}    
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class GAIA_Speech_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.gaia.speech.task import task1_speech_process, task2_speech_recognition, \
            task3_llm_process_qwen, task4_llm_process_deepseek, task5_fuse_llm_answer              
        self.workflow_type= workflow_type
        self.tools = {
            "task1_speech_process": task1_speech_process,
            "task2_speech_recognition": task2_speech_recognition,
            "task3_llm_process_qwen": task3_llm_process_qwen,
            "task4_llm_process_deepseek": task4_llm_process_deepseek,
            "task5_fuse_llm_answer": task5_fuse_llm_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_speech_process", "target": "task2_speech_recognition"},
                {"source": "task2_speech_recognition", "target": "task3_llm_process_qwen"},
                {"source": "task2_speech_recognition", "target": "task4_llm_process_deepseek"},        
                {"source": "task3_llm_process_qwen", "target": "task5_fuse_llm_answer"},
                {"source": "task4_llm_process_deepseek", "target": "task5_fuse_llm_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class GAIA_Vision_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的GAIA问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.gaia.vision.task import task1_obtain_content, task2_vlm_process, task3_output_final_answer           
        self.workflow_type= workflow_type
        self.tools = {
            "task1_obtain_content": task1_obtain_content,
            "task2_vlm_process": task2_vlm_process,
            "task3_output_final_answer": task3_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_obtain_content", "target": "task2_vlm_process"},
                {"source": "task2_vlm_process", "target": "task3_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class tbench_airline_book_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.airline_book.task import task0_init, task1_llm_process, task2a_search_direct_flight, task2b_search_onestop_flight, task2c_get_user_details, task3_llm_process_filter_and_decide, task4_book_reservation
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_search_direct_flight": task2a_search_direct_flight,
            "task2b_search_onestop_flight": task2b_search_onestop_flight,
            "task2c_get_user_details": task2c_get_user_details,
            "task3_llm_process_filter_and_decide": task3_llm_process_filter_and_decide,
            "task4_book_reservation": task4_book_reservation
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task0_init", "target": "task1_llm_process"},
                {"source": "task1_llm_process", "target": "task2a_search_direct_flight"},
                {"source": "task1_llm_process","target": "task2b_search_onestop_flight"},
                {"source": "task1_llm_process","target": "task2c_get_user_details"},
                {"source": "task2a_search_direct_flight","target": "task3_llm_process_filter_and_decide"},
                {"source": "task2b_search_onestop_flight","target": "task3_llm_process_filter_and_decide"},
                {"source": "task2c_get_user_details","target": "task3_llm_process_filter_and_decide"},
                {"source": "task3_llm_process_filter_and_decide","target": "task4_book_reservation"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class tbench_airline_cancel_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.airline_cancel.task import task0_init, task1_llm_process1, task2_get_user_and_reservation_details, task3_cancel_reservation, task4_search_new_flights, task5_llm_process2, task6_book_new_reservation
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process1": task1_llm_process1,
            "task2_get_user_and_reservation_details": task2_get_user_and_reservation_details,
            "task3_cancel_reservation": task3_cancel_reservation,
            "task4_search_new_flights": task4_search_new_flights,
            "task5_llm_process2": task5_llm_process2,
            "task6_book_new_reservation": task6_book_new_reservation
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task0_init", "target": "task1_llm_process1"},
                {"source": "task1_llm_process1", "target": "task2_get_user_and_reservation_details"},
                {"source": "task2_get_user_and_reservation_details", "target": "task3_cancel_reservation"},
                {"source": "task3_cancel_reservation", "target": "task4_search_new_flights"},
                {"source": "task4_search_new_flights", "target": "task5_llm_process2"},
                {"source": "task5_llm_process2", "target": "task6_book_new_reservation"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class tbench_retail_cancel_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.retail_cancel.task import task0_init, task1_llm_process, task2_execute_cancel, task3_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2_execute_cancel": task2_execute_cancel,
            "task3_output_result": task3_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                 {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2_execute_cancel"},
                {"source": "task2_execute_cancel","target": "task3_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")



class tbench_retail_return_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.retail_return.task import task0_init, task1_llm_process, task2_find_user, task3_get_order_details, task4_execute_return, task5_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2_find_user": task2_find_user,
            "task3_get_order_details": task3_get_order_details,
            "task4_execute_return": task4_execute_return,
            "task5_output_result": task5_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                 {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2_find_user"},
                {"source": "task2_find_user","target": "task3_get_order_details"},
                {"source": "task3_get_order_details","target": "task4_execute_return"},
                {"source": "task4_execute_return","target": "task5_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


class tbench_retail_modify_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.retail_modify.task import task0_init, task1_llm_process, task2a_find_user, task2b_get_order_details, task3_execute_modifications, task4_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_find_user": task2a_find_user,
            "task2b_get_order_details": task2b_get_order_details,
            "task3_execute_modifications": task3_execute_modifications,
            "task4_output_result": task4_output_result
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                 {"source": "task0_init","target": "task1_llm_process"},
                {"source": "task1_llm_process","target": "task2a_find_user"},
                {"source": "task1_llm_process","target": "task2b_get_order_details"},
                {"source": "task2a_find_user","target": "task3_execute_modifications"},
                {"source": "task2b_get_order_details","target": "task3_execute_modifications"},
                {"source": "task3_execute_modifications","target": "task4_output_result"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")


class tbench_retail_cancel_modify_Agent(BaseActorAgent):
    """
    一个具体的、基于文件的tbench问答工作流Agent。
    它只负责定义自己的工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.tbench.retail_cancel_modify.task import task0_init, task1_llm_process, task2a_find_user, task2b_get_order_details, task3_execute_operations,task4_output_result
        self.workflow_type= workflow_type
        self.tools = {
            "task0_init": task0_init,
            "task1_llm_process": task1_llm_process,
            "task2a_find_user": task2a_find_user,
            "task2b_get_order_details": task2b_get_order_details,
            "task3_execute_operations": task3_execute_operations,
            "task4_output_result": task4_output_result

        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [{
                    "source": "task0_init",
                    "target": "task1_llm_process"
                    },
                {
                    "source": "task1_llm_process",
                    "target": "task2a_find_user"
                },
                {
                    "source": "task1_llm_process",
                    "target": "task2b_get_order_details"
                },
                {
                    "source": "task1_llm_process",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task2a_find_user",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task2b_get_order_details",
                    "target": "task3_execute_operations"
                },
                {
                    "source": "task3_execute_operations",
                    "target": "task4_output_result"
                }
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")
        
class openagi_document_qa_Agent(BaseActorAgent):
    """
    OpenAGI 文档问答工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.openagi.document_qa.task import (
            task1_start_receive_task,
            task2_read_file,
            task3a_extract_text_content,
            task3b_llm_process_extract_structure_info,
            task3c_load_questions_batch,
            task4a_merge_document_analysis,
            task4b_prepare_qa_context,
            task5a_llm_process_batch_1,
            task5b_llm_process_batch_2,
            task5c_llm_process_batch_3,
            task7_merge_all_answers,
            task8_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file": task2_read_file,
            "task3a_extract_text_content": task3a_extract_text_content,
            "task3b_llm_process_extract_structure_info": task3b_llm_process_extract_structure_info,
            "task3c_load_questions_batch": task3c_load_questions_batch,
            "task4a_merge_document_analysis": task4a_merge_document_analysis,
            "task4b_prepare_qa_context": task4b_prepare_qa_context,
            "task5a_llm_process_batch_1": task5a_llm_process_batch_1,
            "task5b_llm_process_batch_2": task5b_llm_process_batch_2,
            "task5c_llm_process_batch_3": task5c_llm_process_batch_3,
            "task7_merge_all_answers": task7_merge_all_answers,
            "task8_output_final_answer": task8_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file"},
                {"source": "task2_read_file", "target": "task3a_extract_text_content"},
                {"source": "task2_read_file", "target": "task3b_llm_process_extract_structure_info"},
                {"source": "task2_read_file", "target": "task3c_load_questions_batch"},
                {"source": "task3a_extract_text_content", "target": "task4a_merge_document_analysis"},
                {"source": "task3b_llm_process_extract_structure_info", "target": "task4a_merge_document_analysis"},
                {"source": "task4a_merge_document_analysis", "target": "task4b_prepare_qa_context"},
                {"source": "task3c_load_questions_batch", "target": "task4b_prepare_qa_context"},
                {"source": "task4b_prepare_qa_context", "target": "task5a_llm_process_batch_1"},
                {"source": "task4b_prepare_qa_context", "target": "task5b_llm_process_batch_2"},
                {"source": "task4b_prepare_qa_context", "target": "task5c_llm_process_batch_3"},
                {"source": "task5a_llm_process_batch_1", "target": "task7_merge_all_answers"},
                {"source": "task5b_llm_process_batch_2", "target": "task7_merge_all_answers"},
                {"source": "task5c_llm_process_batch_3", "target": "task7_merge_all_answers"},
                {"source": "task7_merge_all_answers", "target": "task8_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class openagi_image_captioning_complex_Agent(BaseActorAgent):
    """
    OpenAGI 图像复杂描述工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.openagi.image_captioning_complex.task import (
            task1_start_receive_task,
            task2_read_and_enhance_images,
            task3a_extract_blip_captions,
            task3b_extract_ocr_text,
            task4_merge_image_features,
            task5a_vlm_process,
            task5b_vlm_process,
            task5c_vlm_process,
            task5d_vlm_process,
            task5_merge_results,
            task6_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_and_enhance_images": task2_read_and_enhance_images,
            "task3a_extract_blip_captions": task3a_extract_blip_captions,
            "task3b_extract_ocr_text": task3b_extract_ocr_text,
            "task4_merge_image_features": task4_merge_image_features,
            "task5a_vlm_process": task5a_vlm_process,
            "task5b_vlm_process": task5b_vlm_process,
            "task5c_vlm_process": task5c_vlm_process,
            "task5d_vlm_process": task5d_vlm_process,
            "task5_merge_results": task5_merge_results,
            "task6_output_final_answer": task6_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_and_enhance_images"},
                {"source": "task2_read_and_enhance_images", "target": "task3a_extract_blip_captions"},
                {"source": "task2_read_and_enhance_images", "target": "task3b_extract_ocr_text"},
                {"source": "task3a_extract_blip_captions", "target": "task4_merge_image_features"},
                {"source": "task3b_extract_ocr_text", "target": "task4_merge_image_features"},
                {"source": "task4_merge_image_features", "target": "task5a_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5b_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5c_vlm_process"},
                {"source": "task4_merge_image_features", "target": "task5d_vlm_process"},
                {"source": "task5a_vlm_process", "target": "task5_merge_results"},
                {"source": "task5b_vlm_process", "target": "task5_merge_results"},
                {"source": "task5c_vlm_process", "target": "task5_merge_results"},
                {"source": "task5d_vlm_process", "target": "task5_merge_results"},
                {"source": "task5_merge_results", "target": "task6_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")

class openagi_multimodal_vqa_complex_Agent(BaseActorAgent):
    """
    OpenAGI 多模态VQA复杂工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.openagi.multimodal_vqa_complex.task import (
            task1_start_receive_task,
            task2_read_file,
            task3_file_process,
            task4a_vision_llm_process,
            task4b_vision_llm_process,
            task4c_vision_llm_process,
            task4d_vision_llm_process,
            task4_merge_results,
            task5_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file": task2_read_file,
            "task3_file_process": task3_file_process,
            "task4a_vision_llm_process": task4a_vision_llm_process,
            "task4b_vision_llm_process": task4b_vision_llm_process,
            "task4c_vision_llm_process": task4c_vision_llm_process,
            "task4d_vision_llm_process": task4d_vision_llm_process,
            "task4_merge_results": task4_merge_results,
            "task5_output_final_answer": task5_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file"},
                {"source": "task2_read_file", "target": "task3_file_process"},
                {"source": "task3_file_process", "target": "task4a_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4b_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4c_vision_llm_process"},
                {"source": "task3_file_process", "target": "task4d_vision_llm_process"},
                {"source": "task4a_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4b_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4c_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4d_vision_llm_process", "target": "task4_merge_results"},
                {"source": "task4_merge_results", "target": "task5_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")
        
class openagi_text_processing_multilingual_Agent(BaseActorAgent):
    """
    OpenAGI 多语言文本处理工作流 Agent。
    只负责定义工具和工作流结构，执行逻辑由基类提供。
    """
    def __init__(self, id: str, workflow_type: str):
        super().__init__(id, workflow_type)
        from baseline.workflows.openagi.text_processing_multilingual.task import (
            task1_start_receive_task,
            task2_read_file_and_split_questions,
            task3_language_detect,
            task4_translate_text,
            task5a_text_analysis_summarize,
            task5b_text_analysis_sentiment,
            task6_prepare_llm_batches,
            task7a_llm_process_batch_1,
            task7b_llm_process_batch_2,
            task7c_llm_process_batch_3,
            task8_merge_answers,
            task9_output_final_answer
        )
        self.workflow_type = workflow_type
        self.tools = {
            "task1_start_receive_task": task1_start_receive_task,
            "task2_read_file_and_split_questions": task2_read_file_and_split_questions,
            "task3_language_detect": task3_language_detect,
            "task4_translate_text": task4_translate_text,
            "task5a_text_analysis_summarize": task5a_text_analysis_summarize,
            "task5b_text_analysis_sentiment": task5b_text_analysis_sentiment,
            "task6_prepare_llm_batches": task6_prepare_llm_batches,
            "task7a_llm_process_batch_1": task7a_llm_process_batch_1,
            "task7b_llm_process_batch_2": task7b_llm_process_batch_2,
            "task7c_llm_process_batch_3": task7c_llm_process_batch_3,
            "task8_merge_answers": task8_merge_answers,
            "task9_output_final_answer": task9_output_final_answer
        }
        self.workflow = {
            "nodes": [{"task": name} for name in self.tools.keys()],
            "edges": [
                {"source": "task1_start_receive_task", "target": "task2_read_file_and_split_questions"},
                {"source": "task2_read_file_and_split_questions", "target": "task3_language_detect"},
                {"source": "task3_language_detect", "target": "task4_translate_text"},
                {"source": "task4_translate_text", "target": "task5a_text_analysis_summarize"},
                {"source": "task4_translate_text", "target": "task5b_text_analysis_sentiment"},
                {"source": "task2_read_file_and_split_questions", "target": "task6_prepare_llm_batches"},
                {"source": "task5a_text_analysis_summarize", "target": "task6_prepare_llm_batches"},
                {"source": "task5b_text_analysis_sentiment", "target": "task6_prepare_llm_batches"},
                {"source": "task6_prepare_llm_batches", "target": "task7a_llm_process_batch_1"},
                {"source": "task6_prepare_llm_batches", "target": "task7b_llm_process_batch_2"},
                {"source": "task6_prepare_llm_batches", "target": "task7c_llm_process_batch_3"},
                {"source": "task7a_llm_process_batch_1", "target": "task8_merge_answers"},
                {"source": "task7b_llm_process_batch_2", "target": "task8_merge_answers"},
                {"source": "task7c_llm_process_batch_3", "target": "task8_merge_answers"},
                {"source": "task8_merge_answers", "target": "task9_output_final_answer"}
            ]
        }
        print(f"   -> 具体实现类 '{id}' 已配置，将使用基类的引擎处理 '{self.workflow_type}' 任务。")