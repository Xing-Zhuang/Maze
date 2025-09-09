# --- 工作流任务定义 ---
def task0_init(args, dag_id, question, supplementary_files):
    """
    工作流的第0步：初始化环境和上下文。
    1.加载所有必需的后端数据（航班、用户、预订记录）。
    2.将用户最原始的指令和ID存入上下文。
    """
    import time
    import json
    start_time= time.time()
    print("--- 开始执行 Task 0: 初始化环境 ---")
    try:
        instruction= question
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
            flights_data, users_data, reservations_data= json.loads(supplementary_files['flights.json']), json.loads(supplementary_files['users.json']), json.loads(supplementary_files['reservations.json'])
            backend_data = {
                "flights": flights_data,
                "users": users_data,
                "reservations": reservations_data
            }
            print("--- Task 0 执行完毕: 环境初始化成功 ---")
            return {
                "dag_id": dag_id,
                "backend_data": backend_data,
                "instruction": instruction,
                "start_time": start_time,
                "end_time": time.time(),
            }
        except FileNotFoundError as e:
            print(f"错误:数据文件没找到{e}")
            raise
    except Exception as e:
        print(f"task0_init 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "backend_data": None,
            "instruction": None,
            "result": f"task0_init 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),            
        }

def task1_llm_process1(args, dag_id, instruction, vllm_manager= None, backend= "huggingface"):
    """
    工作流的第1步：[LLM] 提取核心订票信息。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取核心订票信息 ---")
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
    def _extract_json_from_llm_output(llm_output: str) -> str:
        """
        从LLM输出中提取JSON字符串。
        支持多种格式：纯JSON、代码块中的JSON、带说明的JSON等。
        """
        import re
        import json
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
    import time
    try:
        print(f"✅ LLM机票预订信息提取开始....")
        start_time = time.time()  # 记录开始时间 
        import os
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

        # 构建提示
        prompt = f"""
        You are a professional flight booking assistant. Please carefully read the user's instructions below and extract the key information.
        You must return the extracted information in strict JSON format without any additional explanations or text.

        Required fields to extract:
        - "user_id": User ID (e.g., "mia_kim_4397")
        - "cancel_reservation_id": Reservation ID to be canceled
        - "origin": New booking origin airport code (3-letter, e.g., "JFK", "EWR")
        - "destination": New booking destination airport code (3-letter, e.g., "SEA", "LAX")
        - "departure_date": Departure date (format: "YYYY-MM-DD", default year is 2024)
        - "return_date": Return date (format: "YYYY-MM-DD")
        - "cabin": Cabin class (must be one of: "basic_economy", "economy", "business")
        - "baggages": Number of baggage items (integer)
        - "insurance": Whether insurance is needed ("yes" or "no")
        - "payment_preference": Payment preference (e.g., "use_smaller_gift_card_first")
        - "constraints": A list of strings containing all other constraints and preferences

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
        answer= None
        if use_online_model:
            answer= query_llm_online(
                api_url= api_url, 
                api_key= api_key, 
                payload= payload,
                tokenizer_path= tokenizer_path)
        elif backend== "vllm":
            answer= query_vllm_model(
                api_url= vllm_manager.get_next_endpoint("qwen3-32b"),
                model_alias= "qwen3-32b",
                messages= payload["messages"],
                temperature= temperature,
                max_token= max_token,
                top_p= top_p,
                repetition_penalty= repetition_penalty
            )
        else:
            answer= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)

        print(f"LLM原始输出: {answer}")
        import json
        # 提取JSON
        json_str = _extract_json_from_llm_output(answer)
        extracted_info= {}
        if not json_str:
            print("LLM未能从其输出中提取有效的JSON。")
        else:
            try:
                extracted_info = json.loads(json_str)
            except Exception as e:
                print(f"LLM未能从其输出中提取有效的JSON。")
        print(f"LLM提取的extracted_info: {extracted_info}")
        # 将提取的信息存入上下文
        user_id = extracted_info["user_id"]
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        
        return {
            "dag_id": dag_id,
            "extracted_info": extracted_info,
            "user_id": user_id,
            "start_time": start_time,
            "end_time": time.time(),
            "status": "机票预订信息提取成功"
        }
    except Exception as e:
        print(f"task1_llm_process 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "extracted_info": None,
            "user_id": None,
            "status": "failed",
            "result": f"task1_llm_process 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }

def task2_get_user_and_reservation_details(args, dag_id, backend_data, extracted_info, user_id):
    """
    工作流的第2步：[Tool] 获取用户和预订详情。
    1. 获取用户详细信息
    2. 获取需要取消的预订详情
    """
    from baseline.utils.tbench_tools.airline.get_user_details import GetUserDetails
    from baseline.utils.tbench_tools.airline.get_reservation_details import GetReservationDetails
    print("--- 开始执行 Task 2: 获取用户和预订详情 ---")
    try:
        import time
        import json
        print(f"正在获取用户ID: {user_id} 的详情...")
        start_time= time.time()
        # 获取用户详情
        user_details_str = GetUserDetails.invoke(backend_data, user_id)
        print(f"GetUserDetails返回: {user_details_str}")
        if not user_details_str or user_details_str.strip() == "":
            raise ValueError("GetUserDetails返回了空字符串")
        if "Error" in user_details_str:
            raise ValueError(f"获取用户详情失败: {user_details_str}")
        
        try:
            user_details = json.loads(user_details_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误，原始字符串: '{user_details_str}'")
            raise ValueError(f"无法解析用户详情JSON: {e}")
        
        print(f"成功获取用户详情: {user_details.get('name', 'Unknown')}")
        
        # 获取预订详情
        reservation_id = extracted_info.get("cancel_reservation_id")
        if reservation_id:
            print(f"正在获取预订ID: {reservation_id} 的详情...")
            reservation_details_str = GetReservationDetails.invoke(backend_data, reservation_id)
            print(f"GetReservationDetails返回: {reservation_details_str}")
            
            if not reservation_details_str or reservation_details_str.strip() == "":
                raise ValueError("GetReservationDetails返回了空字符串")
            
            if "Error" in reservation_details_str:
                raise ValueError(f"获取预订详情失败: {reservation_details_str}")
            
            try:
                reservation_details = json.loads(reservation_details_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误，原始字符串: '{reservation_details_str}'")
                raise ValueError(f"无法解析预订详情JSON: {e}")
            print(f"成功获取预订详情: {reservation_details.get('reservation_id', 'Unknown')}")
        else:
            print("没有需要获取的预订ID")
        
        print("--- Task 2 执行完毕 ---")
        return {
            "dag_id": dag_id,
            "status": "获取用户和预订详情成功",
            "user_details": user_details,
            "reservation_details": reservation_details,
            "cancel_reservation_id": reservation_id,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"task2_get_user_and_reservation_details 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "user_details": None,
            "reservation_details": None,
            "cancel_reservation_id": None,
            "status": "failed",
            "result": f"task2_get_user_and_reservation_details 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }

def task3_cancel_reservation(args, dag_id, backend_data, extracted_info, cancel_reservation_id):
    """
    工作流的第3步：[Tool] 取消预订。
    调用取消预订工具执行取消操作。
    """
    print("--- 开始执行 Task 3: 取消预订 ---")
    try:
        import time
        import json
        from baseline.utils.tbench_tools.airline.cancel_reservation import CancelReservation
        reservation_id = cancel_reservation_id
        start_time= time.time()
        if not reservation_id:
            print("没有需要取消的预订ID")
            return {"dag_id": dag_id, "status": "无需取消预订", "cancel_reservation_result": None, "start_time": start_time, "end_time": time.time()}
        
        # 调用工具
        cancel_reservation_result = CancelReservation.invoke(backend_data, reservation_id)
        print(f"取消预订结果: {cancel_reservation_result}")
        print("--- Task 3 执行完毕 ---")
        return {
            "status": "取消预订成功",
            "dag_id": dag_id,
            "cancel_reservation_result": cancel_reservation_result,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task3_cancel_reservation 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "cancel_reservation_result": None,
            "status": "failed",
            "result": f"task3_cancel_reservation 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }

def task4_search_new_flights(args, dag_id, backend_data, extracted_info, cancel_reservation_id):
    """
    工作流的第4步：[Tool] 搜索新航班。
    根据用户要求搜索新的航班。
    """
    from baseline.utils.tbench_tools.airline.search_direct_flight import SearchDirectFlight
    import time
    import json
    print("--- 开始执行 Task 4: 搜索新航班 ---")
    try:
        start_time= time.time()
        origin = extracted_info.get("origin")
        destination = extracted_info.get("destination")
        departure_date = extracted_info.get("departure_date")
        return_date = extracted_info.get("return_date")
        
        print(f"搜索条件: 从 {origin} 到 {destination}, 出发日期: {departure_date}, 返回日期: {return_date}")
        
        # 使用 SearchDirectFlight 工具搜索航班
        outbound_flights = []
        return_flights = []
        
        # 搜索去程航班
        outbound_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, departure_date)
        outbound_flights = json.loads(outbound_flights_str)
        if not outbound_flights:
            print(f"WARNING:未找到从 {origin}到{destination}的航班")
        else:
            print(f"找到 {len(outbound_flights)} 个去程航班")
            for flight in outbound_flights:
                print(f"航班号: {flight['flight_number']}, 价格: {flight['prices']}")
        
        # 搜索返程航班
        if return_date:
            return_flights_str = SearchDirectFlight.invoke(backend_data, destination, origin, return_date)
            return_flights = json.loads(return_flights_str)
            if not return_flights:
                print(f"WARNING: 未找到从 {destination} 到 {origin} 的航班")
            else:
                print(f"找到 {len(return_flights)} 个返程航班")
                for flight in return_flights:
                    print(f"航班号: {flight['flight_number']}, 价格: {flight['prices']}")
        
        print(f"找到 {len(outbound_flights)} 个去程航班和 {len(return_flights) if return_date else 0} 个返程航班")
        print("--- Task 4 执行完毕 ---")

        return {
            "dag_id": dag_id,
            "outbound_flights": outbound_flights,
            "return_flights": return_flights,
            "start_time": start_time,
            "end_time": time.time(),
            "status": "搜索新航班成功"
        }
    except Exception as e:
        print(f"task4_search_new_flights 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "outbound_flights": None,
            "return_flights": None,
            "status": "failed",
            "result": f"task4_search_new_flights 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        }

def task5_llm_process2(args, dag_id, instruction, backend_data, extracted_info, cancel_reservation_id, outbound_flights, return_flights, vllm_manager= None, backend= "huggingface"):
    """
    工作流的第5步：[LLM] 选择航班。
    使用LLM根据用户偏好从候选航班中选择最合适的航班。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 5: LLM选择航班 ---")
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
    def _extract_json_from_llm_output(llm_output: str) -> str:
        """
        从LLM输出中提取JSON字符串。
        支持多种格式：纯JSON、代码块中的JSON、带说明的JSON等。
        """
        import re
        import json
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
    import time
    import os
    try:
        print(f"✅ LLM航班选择开始....")
        start_time = time.time()  # 记录开始时间 
        api_url = args.api_url
        api_key = args.api_key
        temperature= args.temperature
        max_token= args.max_token
        use_online_model= args.use_online_model
        model_folder= args.model_folder
        repetition_penalty= args.repetition_penalty
        top_p= args.top_p
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        # 从上下文中获取必要信息
        if not instruction:
            raise ValueError(f"任务 {dag_id} 缺少 instruction")
        # 安全地获取return_flights，如果不存在则使用空列表
        import json
        if not outbound_flights:
            print("没有找到符合条件的去程航班")
        
        # 构建提示
        prompt = f"""
        You are a professional flight booking decision assistant.
        Your task is to select the most suitable flights from the candidate options based on user preferences.

        # User Preferences
        {json.dumps(extracted_info, indent=2)}

        # Candidate Flights
        Outbound Flights:
        {json.dumps(outbound_flights, indent=2)}

        Return Flights:
        {json.dumps(return_flights, indent=2)}

        # Your Task
        1. Carefully read and understand each of the user's requirements
        2. Select the most appropriate flight combination from the candidates
        3. Return the chosen flight numbers in strict JSON format as follows:
        {{
            "outbound_flight_number": "xxx",
            "return_flight_number": "xxx"
        }}

        Notes:
        - If no suitable flight is found, the corresponding value can be null.
        - Ensure the returned JSON format is correct.
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
        selected_flights = None
        json_str = _extract_json_from_llm_output(llm_output)
        if json_str:
            try:
                selected_flights = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"❌ JSON解析失败！无法解析的字符串是: '{json_str}'")
        # 在这里你可以决定是抛出异常，还是给一个默认值继续运行
        print(f"LLM选择的航班: {json.dumps(selected_flights, indent=2, ensure_ascii=False)}")
        
        return {
            "dag_id": dag_id,
            "selected_flights": selected_flights,
            "start_time": start_time,
            "end_time": time.time(),
            "status": "航班选择成功"
        }
    except Exception as e:
        print(f"task5_llm_process2 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "selected_flights": None,
            "result": f"task5_llm_process2 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        }

def task6_book_new_reservation(args, dag_id, user_id, instruction, backend_data, extracted_info, user_details, outbound_flights, return_flights= [], selected_flights= []):
    """
    工作流的第6步：[Tool] 执行新预订。
    使用选择好的航班信息执行新的预订。
    """

    print("--- 开始执行 Task 6: 执行新预订 ---")
    try:
        import time
        import json
        from baseline.utils.tbench_tools.airline.book_reservation import BookReservation
        start_time= time.time()
        # 安全地获取return_flights，如果不存在则使用空列表
        outbound_flight_num = selected_flights.get("outbound_flight_number")
        return_flight_num = selected_flights.get("return_flight_number")
        
        if not isinstance(selected_flights, dict):
            return {"dag_id": dag_id, "status": "新预订完成", "result": "选择的航班信息格式不正确", "start_time": start_time, "end_time": time.time()}            
            # raise ValueError("选择的航班信息格式不正确")

        if not outbound_flight_num:
            return {"dag_id": dag_id, "status": "新预订完成", "result": "LLM未能选择有效的去程航班", "start_time": start_time, "end_time": time.time()}            
            # raise ValueError("LLM未能选择有效的去程航班")

        # 根据航班号从候选航班中找到完整的航班信息以获取价格
        selected_outbound_flight = next((f for f in outbound_flights if f.get("flight_number") == outbound_flight_num), None)
        
        selected_return_flight = None
        if return_flight_num:
            selected_return_flight = next((f for f in return_flights if f.get("flight_number") == return_flight_num), None)

        if not selected_outbound_flight:
            return {"dag_id": dag_id, "status": "新预订完成", "result": f"无法在候选航班中找到去程航班 {outbound_flight_num}", "start_time": start_time, "end_time": time.time()}
            # raise ValueError(f"无法在候选航班中找到去程航班 {outbound_flight_num}")
        if return_flight_num and not selected_return_flight:
            return {"dag_id": dag_id, "status": "新预订完成", "result": f"无法在候选航班中找到返程航班 {return_flight_num}", "start_time": start_time, "end_time": time.time()}
            # raise ValueError(f"无法在候选航班中找到返程航班 {return_flight_num}")
        
        # 准备乘客信息 - 支持多乘客
        passengers = []
        # 从用户指令中提取乘客数量，如果没有指定则默认为1
        num_passengers = extracted_info.get("num_passengers", 1)
        
        if not user_details:
            print(f"用户信息缺失 user_details: {user_details}")
            return {
                "dag_id": dag_id,
                "status": "failed",
                "result": f"用户信息缺失 user_details: {user_details}",
                "start_time": start_time,
                "end_time": time.time(),
            }

        # 如果用户提供了其他乘客信息，则使用提供的信息
        if "passengers" in extracted_info:
            passengers = extracted_info["passengers"]
        else:
            # 否则使用用户信息作为默认乘客
            for _ in range(num_passengers):
                passengers.append({
                    "first_name": user_details["name"]["first_name"],
                    "last_name": user_details["name"]["last_name"],
                    "dob": user_details["dob"]
                })

        # 计算总价
        cabin = extracted_info.get("cabin", "basic_economy")
        total_price = 0
        if selected_outbound_flight:
            total_price += selected_outbound_flight.get("prices", {}).get(cabin, 0)
        if selected_return_flight:
            total_price += selected_return_flight.get("prices", {}).get(cabin, 0)
        
        total_price *= len(passengers)
        total_baggages = extracted_info.get("baggages", 0)
        # 行李费计算
        nonfree_baggages = total_baggages - len(passengers) if total_baggages > len(passengers) else 0
        total_price += 50 * nonfree_baggages  # 每件非免费行李$50

        if extracted_info.get("insurance") == "yes":
            total_price += 30 * len(passengers)

        # 支付逻辑
        payment_methods_for_booking = []
        remaining_balance = total_price

        user_payment_methods = user_details.get("payment_methods", {})
        # 优先使用证书
        certificates = sorted(
            [pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "certificate"],
            key=lambda x: x.get("amount", 0)
        )
        for pm in certificates:
            if remaining_balance > 0:
                amount_to_use = min(remaining_balance, pm.get("amount", 0))
                payment_methods_for_booking.append({"payment_id": pm.get("id"), "amount": amount_to_use})
                remaining_balance -= amount_to_use
        
        # 然后使用礼品卡
        gift_cards = sorted(
            [pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "gift_card"],
            key=lambda x: x.get("amount", 0)
        )
        if extracted_info.get("payment_preference") == "use_larger_gift_card_first":
            gift_cards.reverse()
        
        for pm in gift_cards:
            if remaining_balance > 0:
                amount_to_use = min(remaining_balance, pm.get("amount", 0))
                payment_methods_for_booking.append({"payment_id": pm.get("id"), "amount": amount_to_use})
                remaining_balance -= amount_to_use

        # 最后使用信用卡支付剩余部分
        if remaining_balance > 0.01:
            credit_card = next((pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "credit_card"), None)
            if credit_card:
                payment_methods_for_booking.append({"payment_id": credit_card.get("id"), "amount": round(remaining_balance, 2)})
                remaining_balance = 0
        
        if remaining_balance > 0.01:
            return {"dag_id": dag_id, "status": "支付失败", "result": f"支付失败：用户没有足够的支付方式或余额来完成支付。剩余金额: {remaining_balance}", "start_time": start_time, "end_time": time.time()}

        # 准备用于预订的航班信息（包含正确的日期）
        flights_for_booking_simple = []
        if outbound_flight_num:
            flights_for_booking_simple.append({
                "flight_number": outbound_flight_num,
                "date": extracted_info.get("departure_date")
            })
        if return_flight_num:
            flights_for_booking_simple.append({
                "flight_number": return_flight_num,
                "date": extracted_info.get("return_date")
            })

        # 组装预订参数
        booking_args = {
            "user_id": user_id,
            "origin": extracted_info.get("origin"),
            "destination": extracted_info.get("destination"),
            "flight_type": "round_trip" if return_flight_num else "one_way",
            "cabin": cabin,
            "flights": flights_for_booking_simple,
            "passengers": passengers,
            "payment_methods": payment_methods_for_booking,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": extracted_info.get("insurance", "no")
        }
        
        # 执行预订
        result = BookReservation.invoke(backend_data, **booking_args)
        print("--- Task 6 执行完毕 ---")
        return {
            "status": "新预订完成",
            "dag_id": dag_id,
            "booking_result": result,
            "result": result,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"task6_book_new_reservation 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "result": f"task6_book_new_reservation 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time(),
        }