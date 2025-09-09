import ray
import json
import os
import sys
import dashscope
from agentos.scheduler import io, gpu,cpu
import time
import re
import gc
from typing import Dict, Any, List
# --- 公共模块和工具导入 ---
from agentos.utils.tbench_tools.common import _extract_json_from_llm_output
from agentos.utils.tbench_tools.retail.find_user_id_by_email import FindUserIdByEmail
from agentos.utils.tbench_tools.retail.find_user_id_by_name_zip import FindUserIdByNameZip
from agentos.utils.tbench_tools.retail.get_user_details import GetUserDetails
from agentos.utils.tbench_tools.retail.get_order_details import GetOrderDetails
from agentos.utils.tbench_tools.retail.return_delivered_order_items import ReturnDeliveredOrderItems

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
            You are a professional retail order processing assistant. Please carefully review the user's instructions and extract all key information required for **returns** in strict JSON format.

            Required fields to extract:
            - "order_id": (string) The order ID the user wants to process (starting with '#').
            - "items": (list of strings) List of product the user wants to return, if all items should return, you should only return ['all'].
            - "reason": (string) [optional] Reason provided by the user.
            - "user_name": (string) [optional] User's name.
            - "zip_code": (string) [optional] User's postal code.
            - "email": (string) [optional] User's email address.
            - "payment_method_id": (string) [optional] Refund method ID explicitly specified by the user.

            User instructions: "{instruction}"

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
            "result": f"task0_init 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        })

@gpu(gpu_mem= 70000, model_name= "qwen3-32b", backend="huggingface")
def task1_llm_process(context):
    """
    工作流第1步：提取意图。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取意图 ---")
    try:
        backend= task1_llm_process._task_decorator["backend"]
        print(f"✅ LLM退货意图提取开始....")
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

        # 构建提示
        prompt = f"""
        You are a professional retail order processing assistant. Please carefully review the user's instructions and extract all key information required for **returns** in strict JSON format.

        Required fields to extract:
        - "order_id": (string) The order ID the user wants to process (starting with '#').
        - "items": (list of strings) List of product the user wants to return, if all items should return, you should only return ['all'].
        - "reason": (string) [optional] Reason provided by the user.
        - "user_name": (string) [optional] User's name.
        - "zip_code": (string) [optional] User's postal code.
        - "email": (string) [optional] User's email address.
        - "payment_method_id": (string) [optional] Refund method ID explicitly specified by the user.

        User instructions: "{instruction}"

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
            inference_features, llm_output= query_vllm_model(
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
        if not json_str:
            print("⭐ LLM未能从其输出中提取有效的JSON。")
            json_str= {}
        # print(f"⭐ instruction: {instruction}, json_str: {json_str}")
        extracted_info = json.loads(json_str)
        
        # 将提取的信息存入上下文
        context.put.remote("extracted_info", extracted_info)
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        
        return json.dumps({
            "dag_id": dag_id,
            "curr_task_feat": inference_features,
            "start_time": start_time,
            "end_time": time.time(),
            "status": "退货意图提取成功"
        })
    except Exception as e:
        print(f"task1_llm_process 发生错误: {str(e)}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),            
            "result": f"task1_llm_process 发生错误: {str(e)}"
        })

@cpu(cpu_num= 1, mem= 1024)
def task2_find_user(context):
    # --- 工具导入 ---

    print("--- 开始执行 Task 2: 查找用户 ---")
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote('dag_id'))
        extracted_info = ray.get(context.get.remote("extracted_info"))
        backend_data = ray.get(context.get.remote("backend_data"))
        user_id = None
        email = extracted_info.get("email")
        if email:
            print(f"尝试通过邮箱查找用户: {email}")
            user_id = FindUserIdByEmail.invoke(backend_data, email=email)
        
        user_name = extracted_info.get("user_name")
        zip_code = extracted_info.get("zip_code")
        if user_name and zip_code:
            name_parts = user_name.split()
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            print(f"尝试通过姓名和邮编查找用户, first_name: {first_name}, last_name: {last_name}, zip_code: {zip_code}")
            user_id = FindUserIdByNameZip.invoke(backend_data, first_name=first_name, last_name=last_name, zip=zip_code)
            
        if not user_id or user_id.startswith("Error:"): 
            print(f"无法根据提供的信息找到唯一的用户: {user_id}")
            return json.dumps({
                "dag_id": dag_id,
                "status": "查找用户失败",
                "start_time": start_time,
                "end_time": time.time()
            })
        user_details_str = GetUserDetails.invoke(backend_data, user_id=user_id)
        if "Error" in user_details_str: 
            print(f"找到用户ID后，无法获取用户详情: {user_details_str}")
            return json.dumps({
                "dag_id": dag_id,
                "status":  f"找到用户ID后，无法获取用户详情: {user_details_str}",
                "start_time": start_time,
                "end_time": time.time()
            })
        user_details = json.loads(user_details_str)
        context.put.remote("user_details", user_details)
        print(f"成功获取用户详情: {user_details}")
        print("--- Task 2 执行完毕 ---")
        return json.dumps({
            "dag_id": dag_id,
            "status": "查找用户成功",
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"Task 2 (查找用户) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task2_find_user 发生错误: {str(e)}"
        })

@cpu(cpu_num= 1, mem= 1024)
def task3_get_order_details(context):
    # --- 工具导入 --

    print("--- 开始执行 Task 3: 获取订单详情 ---")
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote('dag_id'))
        extracted_info = ray.get(context.get.remote("extracted_info"))
        order_id = extracted_info.get("order_id")
        if not order_id: 
            print("LLM未能从指令中提取'order_id'")
            return json.dumps({
                "dag_id": dag_id,
                "status": "获取订单详情失败",
                "start_time": start_time,
                "end_time": time.time()
            })
        backend_data = ray.get(context.get.remote("backend_data"))
        order_details_str = GetOrderDetails.invoke(backend_data, order_id)
        if "Error" in order_details_str: 
            print(f"无法获取订单详情: {order_details_str}")
            return json.dumps({
                "dag_id": dag_id,
                "status": f"无法获取订单详情: {order_details_str}",
                "start_time": start_time,
                "end_time": time.time()
            })
        order_details = json.loads(order_details_str)
        context.put.remote("order_details", order_details)
        print(f"成功获取订单 {order_id} 的详情。")
        print("--- Task 3 执行完毕 ---")
        return json.dumps({
            "dag_id": dag_id,
            "status": "获取订单详情成功",
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"Task 3 (获取订单详情) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task3_get_order_details 发生错误: {str(e)}"
        })

@cpu(cpu_num= 1, mem= 1024)
def task4_execute_return(context):
    # --- 工具导入 ---
    start_time = time.time()  # 记录开始时间 
    print("--- 开始执行 Task 4: 执行退货操作 ---")
    try:
        dag_id= ray.get(context.get.remote('dag_id'))
        extracted_info = ray.get(context.get.remote("extracted_info"))
        order_details = ray.get(context.get.remote("order_details"))
        user_details = ray.get(context.get.remote("user_details"))
        backend_data = ray.get(context.get.remote("backend_data"))
        payment_method_id = extracted_info.get("payment_method_id")
        if not payment_method_id:
            payment_methods_dict = user_details.get("payment_methods")
            if payment_methods_dict:
                payment_method_id = list(payment_methods_dict.values())[0]["id"]
                print(f"指令中未指定退款方式，使用用户的默认支付方式: {payment_method_id}")
            else: 
                print(f"用户没有任何可用的支付方式用于退款。")
                return json.dumps({
                    "dag_id": dag_id,
                    "final_result": "用户没有任何可用的支付方式用于退款。",
                    "start_time": start_time,
                    "end_time": time.time()
                })
        
        item_names_to_return = [name.lower() for name in extracted_info.get("items", [])]
        if not item_names_to_return: 
            print(f"退货操作需要指定商品。")
            return json.dumps({
                "dag_id": dag_id,
                "final_result": "退货操作需要指定商品。",
                "start_time": start_time,
                "end_time": time.time()
            })
        item_ids_to_return = [item["item_id"] for item in order_details.get("items", []) if item["name"].lower() in item_names_to_return]
        if not item_ids_to_return:
            print(f"在订单 {order_details.get('order_id')} 中未找到指定的退货商品: {item_names_to_return}")
            return json.dumps({
                "dag_id": dag_id,
                "final_result": f"在订单 {order_details.get('order_id')} 中未找到指定的退货商品: {item_names_to_return}",
                "start_time": start_time,
                "end_time": time.time()
            })
        params = {"order_id": order_details.get('order_id'), "item_ids": item_ids_to_return, "payment_method_id": payment_method_id}
        print(f"准备执行退货，参数: {params}")
        result_str = ReturnDeliveredOrderItems.invoke(backend_data, **params)
        if result_str.startswith("Error:"):
            print(f"工具执行失败: {result_str}")
            return json.dumps({
                "dag_id": dag_id,
                "final_result": f"工具执行失败: {result_str}",
                "start_time": start_time,
                "end_time": time.time()
            })
        final_result = {"action": "return", "result": json.loads(result_str)}
        context.put.remote("final_result", final_result)
        print("--- Task 4 执行完毕 ---")
        return json.dumps({
            "dag_id": dag_id,
            "final_result": final_result,
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"Task 4 (执行退货) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task4_execute_return 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        })

@io(mem= 1024)
def task5_output_result(context):
    print("--- 开始执行 Task 5: 输出最终结果 ---")
    start_time = time.time()  # 记录开始时间 
    try:
        dag_id= ray.get(context.get.remote('dag_id'))
        final_result = ray.get(context.get.remote("final_result"))
        print("✅ 工作流成功执行完毕！")
        final_output = f"最终操作结果:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}"
        print(final_output)
        return json.dumps({
            "dag_id": dag_id,
            "final_result": final_result,
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"task5_output_result 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task5_output_result 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        })