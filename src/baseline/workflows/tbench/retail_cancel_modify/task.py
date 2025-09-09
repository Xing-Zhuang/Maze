# --- 工作流任务定义 ---
def task0_init(args, dag_id, supplementary_files, question):
    """
    工作流的第0步：初始化环境和上下文。
    1.加载所有必需的后端数据（航班、用户、预订记录）。
    2.将用户最原始的指令和ID存入上下文。
    """
    import time
    import json
    start_time= time.time()
    instruction= question
    print("--- 开始执行 Task 0: 初始化环境 ---")
    try:
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
            products_data, users_data, orders_data= json.loads(supplementary_files['products.json']), json.loads(supplementary_files['users.json']), json.loads(supplementary_files['orders.json'])
            backend_data = {
                "products": products_data,
                "users": users_data,
                "orders": orders_data
            }
            print("--- Task 0 执行完毕: 环境初始化成功 ---")
            return {
                "dag_id": dag_id,
                "backend_data": backend_data,
                "instruction": instruction,
                "start_time": start_time,
                "end_time": time.time()
            }
        except FileNotFoundError as e:
            print(f"错误:数据文件没找到{e}")
            raise
    except Exception as e:
        print(f"task0_init 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "backend_data": None,
            "instruction": None,
            "start_time": start_time,
            "end_time": time.time()
        }

def task1_llm_process(args, dag_id, instruction, backend_data, vllm_manager= None, backend= "huggingface"):
    """
    工作流的第1步：使用LLM从用户指令中提取混合意图（取消和修改）。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取混合意图 ---")
    import json
    from typing import List, Dict
    def query_vllm_model(api_url: str, model_alias: str, messages: List, temperature: float= 0.6, max_token: int= 1024, top_p: float= 0.9, repetition_penalty: float= 1.1):
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
            return content

        except requests.exceptions.RequestException as e:
            error_msg = f"vLLM request failed: {str(e)}"
            return f"[bold red]{error_msg}"
    def query_llm(model:str, model_folder: str, messages, temperature= 0.6, max_token= 1024, top_p= 0.9, repetition_penalty= 1.1):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os
        import gc
        import torch
        # 加载预训练模型和分词器
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map= "cuda")
        local_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_folder, model),
            torch_dtype="float16",
            low_cpu_mem_usage=True,
            device_map= "cuda",
            offload_state_dict=False
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
        del input_ids
        del output
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return response

    def query_llm_online(api_url, api_key, payload, tokenizer_path: str)-> str:
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
                timeout= 3600
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].lstrip()
        except Exception as e:
            console.print(f"[bold red]API request failed: {str(e)}")
            return f"[bold red]API request failed: {str(e)}"
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
    try:
        print(f"✅ LLM混合意图提取开始....")
        import time
        import os
        start_time = time.time()  # 记录开始时间 
        
        # 从上下文中获取必要信息
        api_url = args.api_url
        api_key = args.api_key
        temperature= args.temperature
        max_token= args.max_token
        use_online_model= args.use_online_model
        model_folder= args.model_folder
        repetition_penalty= args.repetition_penalty
        top_p= args.top_p
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")

        if not instruction:
            raise ValueError(f"任务 {dag_id} 缺少 instruction")

        # 构建提示
        prompt = f"""
        You are a top-tier retail order processing assistant. Carefully review the user's instructions and return a single JSON object containing all operation details in strict JSON format.

        The JSON object should include the following optional fields: "user_info", "cancellation", "modification".

        1. "user_info": (object) [if provided] Contains user information.
        - "email": (string)
        - "user_name": (string)
        - "zip_code": (string)

        2. "cancellation": (object) [if cancellation requested]
        - "order_id": (string) Order ID to cancel (starts with "#").
        - "reason": (string) [optional] Cancellation reason.

        3. "modification": (object) [if modification requested]
        - "order_id": (string) Order ID to modify.
        - "item_to_modify": (object) Item to modify, including "name" and "attributes".
        - "new_item_spec": (object) New item specifications (only includes changed "attributes").
        - "payment_method_id": (string) [optional] Payment method ID.

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
        
        llm_output= None
        if use_online_model:
            llm_output= query_llm_online(
                api_url= api_url, 
                api_key= api_key, 
                payload= payload,
                tokenizer_path= tokenizer_path)
        elif backend== "vllm":
            llm_output= query_vllm_model(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages= payload["messages"],
                temperature= temperature,
                max_token= max_token,
                top_p= top_p,
                repetition_penalty= repetition_penalty
            )
        else:
            llm_output= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)

        
        print(f"LLM原始输出: {llm_output}")
        # 提取JSON
        json_str = _extract_json_from_llm_output(llm_output)
        if not json_str:
            print("LLM未能从其输出中提取有效的JSON。")
            json_str= {}
        extracted_info = json.loads(json_str)
        # 将提取的信息存入上下文
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        
        execution_time=time.time() - start_time
        return {
            "dag_id": dag_id,
            "extracted_info": extracted_info,
            "status": "混合意图提取成功",
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"task1_llm_extract_intent 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "extracted_info": None,
            "status": "failed",
            "result": f"task1_llm_extract_intent 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        }

def task2a_find_user(args, dag_id, extracted_info, backend_data):
    print("--- 开始执行 Task 2a: 查找用户 ---")
    from baseline.utils.tbench_tools.retail.find_user_id_by_email import FindUserIdByEmail
    from baseline.utils.tbench_tools.retail.find_user_id_by_name_zip import FindUserIdByNameZip
    from baseline.utils.tbench_tools.retail.get_user_details import GetUserDetails
    #工具导入
    try:
        import time
        import json
        start_time= time.time()
        user_info = extracted_info.get("user_info")
        if not user_info:
            print("指令中未包含用户信息，跳过用户查找。")
            return {
                "status": "success in task2a",
                "user_details": None,
                "dag_id": dag_id,
                "start_time": start_time,
                "end_time": time.time(),
                "dag_id": dag_id
            }
        
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
        print(f"成功找到用户: {user_id}")
        return {
            "dag_id": dag_id,
            "status": "success in task2a",
            "dag_id": dag_id,
            "user_details": user_details,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"Task 2a (查找用户) 发生错误: {e}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "user_details": None,
            "result": f"task2a_find_user 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        }

def task2b_get_order_details(args, dag_id, extracted_info, backend_data):
    #工具导入
    print("--- 开始执行 Task 2b: 获取所有相关订单详情 ---")
    import time
    try:
        start_time= time.time()
        order_ids_to_fetch = set()
        if "cancellation" in extracted_info and extracted_info["cancellation"] and extracted_info["cancellation"].get("order_id"):
            order_ids_to_fetch.add(extracted_info["cancellation"]["order_id"])
        if "modification" in extracted_info and extracted_info["modification"] and extracted_info["modification"].get("order_id"):
            order_ids_to_fetch.add(extracted_info["modification"]["order_id"])

        if not order_ids_to_fetch:
            print("指令中未找到任何需要获取详情的order_id。")
            return {
                "dag_id": dag_id,
                "status": "success in task2b",
                "order_details_map": {},
                "start_time": start_time,
                "end_time": time.time(),
            }
        from baseline.utils.tbench_tools.retail.get_order_details import GetOrderDetails
        import json
        all_order_details = {}
        for order_id in order_ids_to_fetch:
            details_str = GetOrderDetails.invoke(backend_data, order_id)
            if not details_str.startswith("Error:"):
                all_order_details[order_id] = json.loads(details_str)
            else:
                print(f"警告: 无法获取订单 {order_id} 的详情: {details_str}")
        
        print(f"成功获取 {len(all_order_details)} 个订单的详情。")
        return {
            "dag_id": dag_id,
            "status": "success in task2b",
            "order_details_map": all_order_details,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"Task 2b (获取订单详情) 发生错误: {e}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "order_details_map": None,
            "result": f"task2b_get_order_details 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }

def task3_execute_operations(args, dag_id, extracted_info, backend_data, user_details, order_details_map):
    #工具导入
    print("--- 开始执行 Task 3: 执行所有操作(取消/修改) ---")
    try:
        import time
        import json
        start_time= time.time()
        final_results = {}
        from baseline.utils.tbench_tools.retail.cancel_pending_order import CancelPendingOrder
        # 1. 执行取消操作
        if "cancellation" in extracted_info and extracted_info["cancellation"]:
            cancellation_ops = extracted_info["cancellation"]
            # 确保操作列表是列表格式，以便统一处理
            if isinstance(cancellation_ops, dict):
                cancellation_ops = [cancellation_ops]

            cancellation_results = []
            for op_info in cancellation_ops:
                order_id = op_info.get("order_id")
                if order_id:
                    reason = op_info.get("reason", "用户未提供原因")
                    result_str = CancelPendingOrder.invoke(backend_data, order_id=order_id, reason=reason)
                    result = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}
                else:
                    result = {"status": "skipped", "reason": "No order_id for cancellation."}
                cancellation_results.append(result)
            # 如果只有一个结果，则直接返回该结果，否则返回列表
            if cancellation_results:
                 final_results["cancellation_result"] = cancellation_results[0] if len(cancellation_results) == 1 else cancellation_results
        from baseline.utils.tbench_tools.retail.list_all_product_types import ListAllProductTypes
        from baseline.utils.tbench_tools.common import _find_item_details_in_order, _find_new_product_variant_id
        from baseline.utils.tbench_tools.retail.modify_pending_order_items import ModifyPendingOrderItems
        # 2. 执行修改操作
        if "modification" in extracted_info and extracted_info["modification"]:
            modification_ops = extracted_info["modification"]
            # 确保操作列表是列表格式，以便统一处理
            if isinstance(modification_ops, dict):
                modification_ops = [modification_ops]
            
            modification_results = []
            for op_info in modification_ops:
                order_id = op_info.get("order_id")
                if not order_id:
                    result = {"status": "skipped", "reason": "No order_id for modification."}
                else:
                    order_details = order_details_map.get(order_id)
                    if not order_details:
                        result = {"status": "error", "details": f"未能获取订单 {order_id} 的详情。"}
                    else:
                        item_to_modify = op_info.get("item_to_modify", {})
                        new_item_spec = op_info.get("new_item_spec", {})
                        
                        product_map_str = ListAllProductTypes.invoke(backend_data)
                        product_map = json.loads(product_map_str) if not product_map_str.startswith("Error:") else {}
                        original_item = _find_item_details_in_order(order_details, item_to_modify)
                        
                        if not original_item:
                            error = f"未找到商品: {item_to_modify}"
                            result = {"status": "error", "details": error}
                        else:
                            product_id = product_map.get(original_item["name"])
                            new_id = _find_new_product_variant_id(backend_data, product_id, original_item.get("options", {}), new_item_spec)
                            if not new_id:
                                result = {"status": "error", "details": f"找不到与新规格匹配的商品变体: {new_item_spec}"}
                            else:
                                payment_id = op_info.get("payment_method_id") or (list(user_details["payment_methods"].values())[0]["id"] if user_details and user_details.get("payment_methods") else None)
                                if not payment_id:
                                    result = {"status": "error", "details": "无法确定用于修改的支付方式。"}
                                else:
                                    params = {"order_id": order_id, "item_ids": [original_item['item_id']], "new_item_ids": [new_id], "payment_method_id": payment_id}
                                    result_str = ModifyPendingOrderItems.invoke(backend_data, **params)
                                    result = {"status": "success", "result": json.loads(result_str)} if not result_str.startswith("Error:") else {"status": "error", "details": result_str}
                modification_results.append(result)
            # 如果只有一个结果，则直接返回该结果，否则返回列表
            if modification_results:
                final_results["modification_result"] = modification_results[0] if len(modification_results) == 1 else modification_results
        print("--- Task 3 执行完毕 ---")
        return {
            "dag_id": dag_id,
            "status": "success in task3",
            "final_result": final_results,
            "dag_id": dag_id,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"Task 3 (执行操作) 发生错误: {e}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "final_result": None,
            "result": f"task3_execute_operations 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }

def task4_output_result(args, dag_id, final_result):
    print("--- 开始执行 Task 4: 输出最终结果 ---")
    try:
        import time
        import json
        start_time= time.time()
        if not final_result:
            print("⚠️ 工作流执行完毕，但未执行任何操作。")
            return {
                "status": "success", 
                "dag_id": dag_id,
                "result": None,
                "start_time": start_time,
                "end_time": time.time()
            }
        
        print("✅ 工作流成功执行完毕！")
        final_output = f"最终操作结果:\n{json.dumps(final_result, indent=2, ensure_ascii=False)}"
        print(final_output)
        return {
            "status": "success", 
            "dag_id": dag_id,
            "result": final_output,
            "start_time": start_time,
            "end_time": time.time()
            }
    except Exception as e:
        print(f"Task 4 (输出结果) 发生错误: {e}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task4_output_result 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }