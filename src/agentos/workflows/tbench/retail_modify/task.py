import ray
import json
import os
import sys
import gc
import re
from agentos.scheduler import io, gpu, cpu
from typing import Dict, Any, List
import time
from agentos.utils.tbench_tools.retail.get_product_details import GetProductDetails
from agentos.utils.tbench_tools.retail.list_all_product_types import ListAllProductTypes
from agentos.utils.tbench_tools.retail.modify_pending_order_address import ModifyPendingOrderAddress
from agentos.utils.tbench_tools.retail.modify_pending_order_payment import ModifyPendingOrderPayment
from agentos.utils.tbench_tools.retail.modify_user_address import ModifyUserAddress
from agentos.utils.tbench_tools.retail.find_user_id_by_email import FindUserIdByEmail
from agentos.utils.tbench_tools.retail.find_user_id_by_name_zip import FindUserIdByNameZip
from agentos.utils.tbench_tools.retail.get_user_details import GetUserDetails
from agentos.utils.tbench_tools.retail.get_order_details import GetOrderDetails
from agentos.utils.tbench_tools.retail.modify_pending_order_items import ModifyPendingOrderItems

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

def _find_item_details_in_order(order_details: Dict[str, Any], item_spec: Dict[str, Any]) -> Dict[str, Any]:
    """在订单详情中查找与规格匹配的商品。"""
    if not item_spec or not item_spec.get("name"): return None
    item_name_to_find = item_spec["name"].lower()
    attributes_to_find = {k.lower(): str(v).lower() for k, v in item_spec.get("attributes", {}).items()}
    for item in order_details.get("items", []):
        if item["name"].lower() == item_name_to_find:
            item_options = {k.lower(): str(v).lower() for k, v in item.get("options", {}).items()}
            if all(item_options.get(k) == v for k, v in attributes_to_find.items()):
                return item
    return None

def _find_new_product_variant_id(backend_data: Dict[str, Any], product_id: str, original_item_options: Dict[str, Any], new_item_spec: Dict[str, Any]) -> str:
    """查找与新规格匹配的商品变体ID。"""
    product_details_str = GetProductDetails.invoke(backend_data, product_id)
    if "Error" in product_details_str: return None
    product_info = json.loads(product_details_str)
    target_options = original_item_options.copy()
    if new_item_spec and new_item_spec.get("attributes"):
        target_options.update(new_item_spec.get("attributes"))
    
    normalized_target = {k.lower(): str(v).lower() for k, v in target_options.items()}
    
    for variant in product_info.get("variants", {}).values():
        normalized_variant = {k.lower(): str(v).lower() for k, v in variant.get("options", {}).items()}
        if normalized_target == normalized_variant:
            return variant.get("item_id")
    return None

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
            You are a premier retail order processing assistant. Carefully review the user's instructions and return a single JSON object containing all modification details in strict JSON format.

            The JSON object may include one or more of the following optional fields: "user_info", "item_modification", "payment_modification", "order_address_modification", "user_address_modification".

            1. "user_info": (object) [if provided] Contains user information ("email", "user_name", "zip_code").

            2. "item_modification": (object) [if item modification requested]
            - "order_id": (string) Order ID to modify (starting with "#").
            - "items_to_modify": (object or list of objects) Items to modify, including "name" and "attributes".
            - "new_items_spec": (object or list of objects) New item specifications (only includes changed "attributes").
            - "payment_method_id": (string) [optional] Payment method ID.

            3. "payment_modification": (object) [if payment method modification requested]
            - "order_id": (string) Order ID.
            - "payment_method_id": (string) New payment method ID.

            4. "order_address_modification": (object) [if order address modification requested]
            - "order_id": (string) Order ID.
            - "address1": (string) The first line of the address, such as '123 Main St'.
            - "address2": (string) The second line of the address, such as 'Apt 1' or ''.
            - "city": (string) The city, such as 'San Francisco'.
            - "state": (string) The state, such as 'CA'.
            - "country": (string) The country, such as 'USA'.
            - "zip": (string) The zip code, such as '94101'.

            5. "user_address_modification": (object) [if user default address modification requested]
            - "user_id": (string) User ID.
            - "address1": (string) The first line of the address, such as '123 Main St'.
            - "address2": (string) The second line of the address, such as 'Apt 1' or ''.
            - "city": (string) The city, such as 'San Francisco'.
            - "state": (string) The state, such as 'CA'.
            - "country": (string) The country, such as 'USA'.
            - "zip": (string) The zip code, such as '94101'.

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
    工作流的第1步：使用LLM从用户指令中提取修改意图。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取意图 ---")
    try:
        backend= task1_llm_process._task_decorator["backend"]
        print(f"✅ LLM修改意图提取开始....")
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
        You are a premier retail order processing assistant. Carefully review the user's instructions and return a single JSON object containing all modification details in strict JSON format.

        The JSON object may include one or more of the following optional fields: "user_info", "item_modification", "payment_modification", "order_address_modification", "user_address_modification".

        1. "user_info": (object) [if provided] Contains user information ("email", "user_name", "zip_code").

        2. "item_modification": (object) [if item modification requested]
        - "order_id": (string) Order ID to modify (starting with "#").
        - "items_to_modify": (object or list of objects) Items to modify, including "name" and "attributes".
        - "new_items_spec": (object or list of objects) New item specifications (only includes changed "attributes").
        - "payment_method_id": (string) [optional] Payment method ID.

        3. "payment_modification": (object) [if payment method modification requested]
        - "order_id": (string) Order ID.
        - "payment_method_id": (string) New payment method ID.

        4. "order_address_modification": (object) [if order address modification requested]
        - "order_id": (string) Order ID.
        - "address1": (string) The first line of the address, such as '123 Main St'.
        - "address2": (string) The second line of the address, such as 'Apt 1' or ''.
        - "city": (string) The city, such as 'San Francisco'.
        - "state": (string) The state, such as 'CA'.
        - "country": (string) The country, such as 'USA'.
        - "zip": (string) The zip code, such as '94101'.

        5. "user_address_modification": (object) [if user default address modification requested]
        - "user_id": (string) User ID.
        - "address1": (string) The first line of the address, such as '123 Main St'.
        - "address2": (string) The second line of the address, such as 'Apt 1' or ''.
        - "city": (string) The city, such as 'San Francisco'.
        - "state": (string) The state, such as 'CA'.
        - "country": (string) The country, such as 'USA'.
        - "zip": (string) The zip code, such as '94101'.

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
            inference_features, llm_output= query_vllm_model(
                api_url= ray.get(context.get.remote(f"task1_llm_process_request_api_url")),
                model_alias= "qwen3-32b", 
                messages= payload["messages"], 
                temperature= temperature, 
                max_token= max_token, 
                top_p= top_p, 
                repetition_penalty= repetition_penalty)
        else:
            inference_features, llm_output= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)

        print(f"LLM原始输出: {llm_output}")
        # 提取JSON
        json_str = _extract_json_from_llm_output(llm_output)
        if not json_str:
            print("LLM未能从其输出中提取有效的JSON。")
            json_str= {}

        extracted_info = json.loads(json_str)
        
        # 将提取的信息存入上下文
        context.put.remote("extracted_info", extracted_info)
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")

        return json.dumps({
            "dag_id": dag_id,
            "curr_task_feat": inference_features,
            "status": "修改意图提取成功",
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"Task 1 (LLM提取意图) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task1_llm_process 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        })

@cpu(cpu_num= 1, mem= 1024)
def task2a_find_user(context):
    #工具

    print("--- 开始执行 Task 2a: 查找用户 ---")
    try:
        start_time= time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        extracted_info = ray.get(context.get.remote("extracted_info"))
        user_info = extracted_info.get("user_info")
        if not user_info:
            print("指令中未包含用户信息，跳过用户查找。")
            context.put.remote("user_details", None)
            return json.dumps({
                "dag_id": dag_id,
                "status": "指令中未包含用户信息，跳过用户查找。",
                "start_time": start_time,
                "end_time": time.time(),
            })
        
        backend_data = ray.get(context.get.remote("backend_data"))
        user_id = None
        if user_info.get("email"):
            user_id = FindUserIdByEmail.invoke(backend_data, email=user_info["email"])
        elif user_info.get("user_name") and user_info.get("zip_code"):
            name_parts = user_info["user_name"].split()
            user_id = FindUserIdByNameZip.invoke(backend_data, first_name=name_parts[0], last_name=" ".join(name_parts[1:]), zip=user_info["zip_code"])
        
        if not user_id or user_id.startswith("Error:"): raise ValueError(f"无法根据提供的信息找到用户: {user_info}")
        user_details_str = GetUserDetails.invoke(backend_data, user_id=user_id)
        if user_details_str.startswith("Error:"): raise ValueError(f"无法获取用户详情: {user_details_str}")
        
        user_details = json.loads(user_details_str)
        context.put.remote("user_details", user_details)
        print(f"成功找到用户: {user_id}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "task2a success",
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"Task 2a (查找用户) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task2a_find_user 发生错误: {str(e)}"
        })

@cpu(cpu_num= 1, mem= 1024)
def task2b_get_order_details(context):
    #工具
    print("--- 开始执行 Task 2b: 获取所有相关订单详情 ---")
    try:
        start_time= time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        extracted_info = ray.get(context.get.remote("extracted_info"))
        backend_data = ray.get(context.get.remote("backend_data"))
        order_ids_to_fetch = set()

        for key in ["item_modification", "payment_modification", "order_address_modification"]:
            if key in extracted_info and extracted_info[key] and extracted_info[key].get("order_id"):
                order_ids_to_fetch.add(extracted_info[key]["order_id"])

        if not order_ids_to_fetch:
            print("指令中未找到任何需要获取详情的order_id。")
            context.put.remote("order_details_map", {})
            return json.dumps({
                "dag_id": dag_id,
                "status": "指令中未找到任何需要获取详情的order_id。",
                "start_time": start_time,
                "end_time": time.time()
            })
        
        all_order_details = {}
        for order_id in order_ids_to_fetch:
            details_str = GetOrderDetails.invoke(backend_data, order_id)
            if not details_str.startswith("Error:"):
                all_order_details[order_id] = json.loads(details_str)
            else:
                print(f"警告: 无法获取订单 {order_id} 的详情: {details_str}")
        
        context.put.remote("order_details_map", all_order_details)
        print(f"成功获取 {len(all_order_details)} 个订单的详情。")
        return json.dumps({
            "dag_id": dag_id,
            "status": "task2b success",
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"Task 2b (获取订单详情) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task2a_find_user 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        })

@cpu(cpu_num= 1, mem= 1024)
def task3_execute_modifications(context):
    #工具
    print("--- 开始执行 Task 3: 执行所有修改操作 ---")
    try:
        dag_id = ray.get(context.get.remote("dag_id"))
        start_time= time.time()
        extracted_info = ray.get(context.get.remote("extracted_info"))
        backend_data = ray.get(context.get.remote("backend_data"))
        user_details = ray.get(context.get.remote("user_details"))
        order_details_map = ray.get(context.get.remote("order_details_map"))
        
        final_results = {}

        # 1. 修改商品
        if "item_modification" in extracted_info and extracted_info["item_modification"]:
            op_info = extracted_info["item_modification"]
            order_id = op_info["order_id"]
            order_details = order_details_map.get(order_id)
            if not order_details:
                result = {"status": "error", "details": f"未能获取订单 {order_id} 的详情。"}
            else:
                items_to_modify = op_info.get("items_to_modify", [])
                new_items_spec = op_info.get("new_items_spec", [])
                if isinstance(items_to_modify, dict): items_to_modify = [items_to_modify]
                if isinstance(new_items_spec, dict): new_items_spec = [new_items_spec]
                
                if len(items_to_modify) != len(new_items_spec):
                    result = {"status": "error", "details": "要修改的商品和新规格数量不匹配。"}
                else:
                    product_map = json.loads(ListAllProductTypes.invoke(backend_data))
                    item_ids_to_return, new_item_ids = [], []
                    error = None
                    for i, item_spec in enumerate(items_to_modify):
                        original_item = _find_item_details_in_order(order_details, item_spec)
                        if not original_item: error = f"未找到商品: {item_spec}"; break
                        product_id = product_map.get(original_item["name"])
                        if not product_id: error = f"未找到产品ID: {original_item['name']}"; break
                        new_id = _find_new_product_variant_id(backend_data, product_id, original_item.get("options", {}), new_items_spec[i])
                        if not new_id: error = f"找不到与新规格匹配的商品变体: {new_items_spec[i]}"; break
                        item_ids_to_return.append(original_item['item_id'])
                        new_item_ids.append(new_id)
                    
                    if error:
                        result = {"status": "error", "details": error}
                    else:
                        payment_id = op_info.get("payment_method_id") or list(user_details.get("payment_methods", {}).values())[0]["id"]
                        params = {"order_id": order_id, "item_ids": item_ids_to_return, "new_item_ids": new_item_ids, "payment_method_id": payment_id}
                        result_str = ModifyPendingOrderItems.invoke(backend_data, **params)
                        result = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}
            final_results["item_modification_result"] = result

        # 2. 修改支付方式
        if "payment_modification" in extracted_info and extracted_info["payment_modification"]:
            op_info = extracted_info["payment_modification"]
            result_str = ModifyPendingOrderPayment.invoke(backend_data, order_id=op_info["order_id"], payment_method_id=op_info["payment_method_id"])
            final_results["payment_modification_result"] = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}

        # 3. 修改订单地址
        if "order_address_modification" in extracted_info and extracted_info["order_address_modification"]:
            op_info = extracted_info["order_address_modification"]
            result_str = ModifyPendingOrderAddress.invoke(backend_data, order_id=op_info["order_id"], address1=op_info["address1"], address2=op_info["address2"], city=op_info["city"], state=op_info["state"], country=op_info["country"], zip=op_info["zip"])
            final_results["order_address_modification_result"] = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}
        
        # 4. 修改用户地址
        if "user_address_modification" in extracted_info and extracted_info["user_address_modification"]:
            op_info = extracted_info["user_address_modification"]
            if not user_details:
                result = {"status": "error", "details": "需要用户信息才能修改用户地址，但未能找到用户。"}
            else:
                result_str = ModifyUserAddress.invoke(backend_data, user_id=user_details["id"], address1=op_info["address1"], address2=op_info["address2"], city=op_info["city"], state=op_info["state"], country=op_info["country"], zip=op_info["zip"])
                result = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}
            final_results["user_address_modification_result"] = result

        context.put.remote("final_result", final_results)
        print("--- Task 3 执行完毕 ---")
        return json.dumps({
            "dag_id": dag_id,
            "status": "success in task2b",
            "start_time": start_time,
            "end_time": time.time(),
        })
    except Exception as e:
        print(f"Task 2b (获取订单详情) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task3_execute_modifications 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        })

@io(mem=2)
def task4_output_result(context):
    print("--- 开始执行 Task 4: 输出最终结果 ---")
    try:
        dag_id = ray.get(context.get.remote("dag_id"))
        start_time= time.time()
        final_result = ray.get(context.get.remote("final_result"))
        if not final_result:
            print("⚠️ 工作流执行完毕，但未执行任何操作。")
            return json.dumps({
                "dag_id": dag_id,
                "status": "⚠️ 工作流执行完毕，但未执行任何操作。", 
                "result": "",
                "start_time": start_time,
                "end_time": time.time()
                })
        
        print("✅ 工作流成功执行完毕！")
        final_output = f"最终操作结果:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}"
        print(final_output)
        return json.dumps({
            "dag_id": dag_id,
            "status": "success", 
            "result": final_output,
            "start_time": start_time,
            "end_time": time.time()
            })
    except Exception as e:
        print(f"Task 4 (输出结果) 发生错误: {e}")
        return json.dumps({
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task4_output_result 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        })