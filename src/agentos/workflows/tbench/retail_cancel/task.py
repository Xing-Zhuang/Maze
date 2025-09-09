import ray
import oss2
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict
from agentos.scheduler import gpu, io, cpu
import os
import re
import gc
import sys
import dashscope

# --- 公共模块和工具导入 ---
from agentos.utils.tbench_tools.common import _extract_json_from_llm_output
from agentos.utils.tbench_tools.retail.cancel_pending_order import CancelPendingOrder

def estimate_tokens(text):
    """
    一个更精确的、用于估算中英混合文本token数的函数。
    它分别计算CJK字符和非CJK单词，并避免了重复计算。
    """
    # 1. 精确计算CJK字符数
    cjk_chars = sum(1 for char in text if '\u4E00' <= char <= '\u9FFF')
    # 2. 从文本中移除所有CJK字符和换行符，以便统计其他语言的单词
    # 我们用正则表达式将CJK字符替换为空格，以确保单词能被正确分割
    non_cjk_text = re.sub(r'[\u4E00-\u9FFF]', ' ', text)
    non_cjk_text = non_cjk_text.replace("\n", " ")
    # 3. 计算非CJK单词数，乘以经验系数
    # 使用split()可以有效地按空格分割出单词
    non_cjk_words_count = len(non_cjk_text.split())
    # 4. 将两部分加总
    # 这里的1.3是英文单词到token的经验系数，可以微调
    estimated_tokens = cjk_chars + int(non_cjk_words_count * 1.3)
    return estimated_tokens

def query_vllm_model(api_url: str, model_alias: str, messages: List, temperature: float= 0.6, max_token: int= 1024, top_p: float= 0.9, repetition_penalty: float= 1.1) -> tuple[dict, str]:
    """
    通过HTTP请求查询本地vLLM服务。
    
    :param api_url: vLLM服务的根URL (例如 http://127.0.0.1:8000)
    :param model_alias: 在vLLM中服务的模型别名 (例如 qwen3-32b)
    :param messages: OpenAI格式的消息列表
    :param temperature: 温度参数
    :param max_token: 最大生成token数
    :param top_p: Top-p采样参数
    :param repetition_penalty: 重复惩罚参数
    :return: 一个包含性能特征和模型响应文本的元组
    """
    import requests
    from rich.console import Console
    console = Console()
    # vLLM的聊天接口路径
    chat_url = f"{api_url.strip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    # 构建与OpenAI兼容的请求体
    payload = {
        "model": model_alias,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_token,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    # 为了计算特征，先将消息拼接起来
    conversation = ""
    for message in messages:
        conversation += f"{message['role']}: {message['content']}"
    try:
        response = requests.post(chat_url, json=payload, headers=headers, timeout=3600)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        response_data = response.json()
        content = response_data['choices'][0]['message']['content'].lstrip()
        
        # 返回与其它query函数格式一致的结果
        features = {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}
        return features, content

    except requests.exceptions.RequestException as e:
        error_msg = f"vLLM request failed: {str(e)}"
        console.print(f"[bold red]{error_msg}")
        features = {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}
        return features, f"[bold red]{error_msg}"

def query_llm(model:str, model_folder: str, messages:List, temperature= 0.6, max_token= 1024, top_p= 0.9, repetition_penalty= 1.1):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    # 加载预训练模型和分词器
    tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map= "auto")
    local_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_folder, model),
        torch_dtype="float16",
        low_cpu_mem_usage=True,
        device_map= "auto"
        )

    # 将messages转换成字符串格式
    conversation = ""
    for message in messages:
        conversation += f"{message['role']}: {message['content']}"

    # 编码输入
    input_ids = tokenizer.encode(conversation + tokenizer.eos_token, return_tensors='pt').to("cuda")

    # 生成回复
    output = local_model.generate(
        input_ids, 
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens= max_token,
        temperature= temperature, 
        top_p= top_p,
        repetition_penalty= repetition_penalty
    )
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    del local_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, response

def query_llm_online(api_url, api_key, payload: Dict[str, str], tokenizer_path: str)-> str:
    """Call SiliconFlow API to get LLM response"""
    import requests
    from rich.console import Console
    # 将messages转换成字符串格式
    conversation= ""
    for message in payload["messages"]:
        conversation+= f"{message['role']}: {message['content']}"
        
    console= Console()
    headers= {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    try:
        response= requests.post(
            api_url,
            json= payload,
            headers= headers,
            timeout= 600
        )
        response.raise_for_status()
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, response.json()['choices'][0]['message']['content'].lstrip()
    except Exception as e:
        console.print(f"[bold red]API request failed: {str(e)}")
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, f"[bold red]API request failed: {str(e)}"

def _extract_json_from_llm_output(llm_output: str) -> str:
    """
    从LLM输出中提取JSON字符串。
    支持多种格式：纯JSON、代码块中的JSON、带说明的JSON等。
    """
    import re
    
    # 尝试直接解析为JSON
    try:
        json.loads(llm_output.strip())
        return llm_output.strip()
    except:
        pass
    
    # 尝试从代码块中提取JSON
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, llm_output, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except:
            pass
    
    # 尝试提取JSON对象
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    for match in matches:
        try:
            json.loads(match)
            return match
        except:
            continue
    
    # 如果都失败了，返回原始输出
    return llm_output.strip()

# --- 工作流任务定义 ---
@io(mem= 1024)
def task0_init(context):
    """
    工作流的第0步：初始化环境和上下文。
    1.加载所有必需的后端数据（航班、用户、预订记录）。
    2.将用户最原始的指令和ID存入上下文。
    """
    start_time= time.time()
    print("--- 开始执行 Task 0: 初始化环境 ---")
    try:
        dag_id = ray.get(context.get.remote("dag_id"))
        instruction = ray.get(context.get.remote("question"))
        
        # 添加调试信息
        print(f"DAG ID: {dag_id}")
        print(f"Question field: {instruction}")
        print(f"Question field type: {type(instruction)}")
        print(f"Question field length: {len(instruction) if instruction else 0}")
        if not instruction:
            print("⚠️ 警告: question字段为空！")
            raise ValueError(f"任务 {dag_id} 的 question 字段为空")

        print(f"接收到指令: {instruction}")

        # 加载tau-bench的数据库json
        try:
            supplementary_files = ray.get(context.get.remote("supplementary_files"))
            products_data, users_data, orders_data= json.loads(supplementary_files['products.json']), json.loads(supplementary_files['users.json']), json.loads(supplementary_files['orders.json'])
            backend_data = {
                "products": products_data,
                "users": users_data,
                "orders": orders_data
            }
            context.put.remote("backend_data", backend_data)
            context.put.remote("instruction", instruction)
            print("--- Task 0 执行完毕: 环境初始化成功 ---")

            prompt = f"""
            You are a professional retail order processing assistant. Please carefully review the user's instructions and extract all orders that need to be **canceled** in strict JSON format.
            If the user mentions multiple orders to cancel, return a JSON list containing multiple objects.

            Each JSON object must include the following fields:
            - "order_id": (string) The order ID the user wants to cancel (starting with '#').
            - "reason": (string) [optional] The cancellation reason provided by the user.

            User instructions:
            "{instruction}"

            JSON output:
            """
            llm_process_feat= {"text_length": len(prompt), "token_count": estimate_tokens(prompt)}

            return json.dumps({
                "dag_id": dag_id,
                "succ_task_feat": {
                    "task1_llm_process": {"text_length": llm_process_feat["text_length"], "token_count": llm_process_feat["token_count"], "reason": 0}
                },
                "curr_task_feat": None,
                "start_time": start_time,
                "end_time": time.time()
            })
        except FileNotFoundError as e:
            print(f"错误:数据文件没找到{e}")
            raise
    except Exception as e:
        print(f"task0_init 发生错误: {str(e)}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),            
            "result": f"task0_init 发生错误: {str(e)}"
        })

@gpu(gpu_mem= 70000, model_name= "qwen3-32b", backend="huggingface")
def task1_llm_process(context):
    """
    工作流的第1步：使用LLM从用户指令中提取取消订单的意图和关键信息。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取意图 ---")
    try:
        backend= task1_llm_process._task_decorator["backend"]
        print(f"✅ LLM意图提取开始....")
        start_time = time.time()  # 记录开始时间 
        
        # 从上下文中获取必要信息
        dag_id = ray.get(context.get.remote("dag_id"))
        instruction = ray.get(context.get.remote("instruction"))
        api_url = ray.get(context.get.remote("api_url"))
        api_key = ray.get(context.get.remote("api_key"))

        use_online_model= ray.get(context.get.remote("use_online_model"))
        model_folder= ray.get(context.get.remote("model_folder"))
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        temperature, max_token, repetition_penalty, top_p= None, None, None, None
        if not use_online_model:
            temperature= ray.get(context.get.remote("temperature"))
            max_token= ray.get(context.get.remote("max_tokens"))
            top_p= ray.get(context.get.remote("top_p"))
            repetition_penalty= ray.get(context.get.remote("repetition_penalty"))

        if not instruction:
            raise ValueError(f"任务 {dag_id} 缺少 instruction")

        # 构建提示
        prompt = f"""
        You are a professional retail order processing assistant. Please carefully review the user's instructions and extract all orders that need to be **canceled** in strict JSON format.
        If the user mentions multiple orders to cancel, return a JSON list containing multiple objects.

        Each JSON object must include the following fields:
        - "order_id": (string) The order ID the user wants to cancel (starting with '#').
        - "reason": (string) [optional] The cancellation reason provided by the user.

        User instructions:
        "{instruction}"

        JSON output:
        """
        
        # 构建API请求payload
        payload = {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_token,
            "enable_thinking": False,
            "response_format": {"type": "text"}               
        }
        
        inference_features, llm_output= None, None
        if use_online_model:
            inference_features, llm_output= query_llm_online(
                api_url= api_url, 
                api_key= api_key, 
                payload= payload,
                tokenizer_path= tokenizer_path)
        elif backend == "vllm":
            inference_features, llm_output = query_vllm_model(
                api_url= ray.get(context.get.remote(f"task1_llm_process_request_api_url")),
                model_alias= "qwen3-32b",
                messages= payload["messages"],
                temperature= temperature,
                max_token= max_token,
                top_p= top_p,
                repetition_penalty= repetition_penalty
            )
        else:
            inference_features, llm_output= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)
        
        print(f"LLM原始输出: {llm_output}")

        # 提取JSON
        json_str = _extract_json_from_llm_output(llm_output)
        extracted_info= {}
        if not json_str:
            print("LLM未能从其输出中提取有效的JSON。")
        else:
            try:
                extracted_info = json.loads(json_str)
            except Exception as e:
                print(f"LLM未能从其输出中提取有效的JSON。")
        print(f"LLM提取的extracted_info: {extracted_info}")
        extracted_info = json.loads(json_str)
        # 将提取的信息存入上下文
        context.put.remote("extracted_info", extracted_info)
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        return json.dumps({
            "dag_id": dag_id,
            "curr_task_feat": inference_features,
            "status": "意图提取成功",
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"task1_llm_process 发生错误: {str(e)}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task1_llm_process 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        })

@cpu(cpu_num= 1, mem= 1024)
def task2_execute_cancel(context):
    """
    工作流的第2步：根据提取的信息执行取消订单操作。
    """
    # 从Tools目录中导入CancelPendingOrder工具

    print("--- 开始执行 Task 2: 执行取消订单 ---")
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote('dag_id'))
        # 从上下文中获取所需数据
        extracted_info = ray.get(context.get.remote("extracted_info"))
        backend_data = ray.get(context.get.remote("backend_data"))
        # 标准化处理，确保 cancellation_requests 是一个列表
        if isinstance(extracted_info, dict):
            cancellation_requests = [extracted_info]
        elif isinstance(extracted_info, list):
            cancellation_requests = extracted_info
        else:
            raise ValueError("LLM提取的信息格式不正确，既不是单个对象也不是列表。")

        all_results = []
        for request in cancellation_requests:
            order_id = request.get("order_id")
            if not order_id:
                print(f"跳过一个无效的请求，缺少'order_id': {request}")
                all_results.append({"error": "Missing order_id", "request": request})
                continue
            
            params = {
                "order_id": order_id,
                "reason": request.get("reason", "用户未提供原因")
            }

            print(f"准备执行取消订单 {order_id}，原因: '{params['reason']}'")
            # 调用工具执行取消操作
            result_str = CancelPendingOrder.invoke(backend_data, **params)
            
            if result_str.startswith("Error:"):
                print(f" > 工具执行失败 for order {order_id}: {result_str}")
                all_results.append({"order_id": order_id, "status": "error", "details": result_str})
            else:
                result_json = json.loads(result_str)
                print(f" > 成功取消订单 {order_id}")
                all_results.append({"order_id": order_id, "status": "success", "result": result_json})

        # 将最终结果存入上下文
        context.put.remote("final_result", all_results)
        print("--- Task 2 执行完毕 ---")
        return json.dumps({
            "dag_id": dag_id,
            "status": "取消订单操作完成",
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"task2_execute_cancel 发生错误: {str(e)}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task2_execute_cancel 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        })

@io(mem=1024)
def task3_output_result(context):
    """
    工作流的第3步：输出取消操作的最终结果。
    """
    print("--- 开始执行 Task 3: 输出最终结果 ---")
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote("dag_id"))
        final_result = ray.get(context.get.remote("final_result"))
        print("✅ 工作流成功执行完毕！")
        # 以格式化的JSON输出最终结果
        final_output = f"最终操作结果:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}"
        print(final_output)
        return json.dumps({
            "dag_id": dag_id,
            "status": "取消订单操作完成", 
            "result": final_output,
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"task3_output_result 发生错误: {str(e)}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task3_output_result 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        })
