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
                "start_time": start_time,
                "backend_data": backend_data,
                "instruction": instruction,
                "end_time": time.time()
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
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task0_init 发生错误: {str(e)}"
        }

def task1_llm_process(args, dag_id, instruction, backend_data, vllm_manager= None, backend= "huggingface"):
    """
    工作流的第1步：[LLM] 提取核心订票信息。
    采用与file/task.py相同的大模型推理模式。
    """
    print("--- 开始执行 Task 1: LLM提取核心订票信息 ---")
    import time
    import json
    import os
    import gc
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
        import time
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
                timeout= 3600
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].lstrip()
        except Exception as e:
            console.print(f"[bold red]API request failed: {str(e)}")
            return f"[bold red]API request failed: {str(e)}"
    try:
        print(f"✅ LLM机票预订信息提取开始....")
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
        # 构建提示
        prompt = f"""
        You are a professional flight booking assistant. Please carefully read the user's flight booking instructions below and extract the key information.  
        You need to return the extracted information in a strict JSON format, without any additional explanations or text.  

        The fields to be extracted are as follows:  
        - "user_id": User ID (e.g., "mia_jackson_2156").  
        - "origin": Origin airport code (3 letters, e.g., "JFK", "SFO").  
        - "destination": Destination airport code (3 letters, e.g., "SEA", "LAX").  
        - "date": Departure date (format: "YYYY-MM-DD", default year is 2024).  
        - "cabin": Cabin class (must be one of "basic_economy", "economy", "business").  
        - "baggages": Number of baggage items (integer).  
        - "insurance": Whether insurance is needed ("yes" or "no").  
        - "constraints": A list of strings containing all other constraints and preferences.  

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
            answer= query_vllm_model(api_url= vllm_manager.get_next_endpoint("qwen3-32b"), model_alias= "qwen3-32b", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)
        else:
            answer= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)

        print(f"LLM原始输出: {answer}")

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
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task1_llm_process 发生错误: {str(e)}"
        }

def task2a_search_direct_flight(dag_id, backend_data, extracted_info):
    """
    工作流的第2.1步：[Tool] 搜索直飞航班。
    """
    import re
    import time
    from baseline.utils.tbench_tools.airline.search_direct_flight import SearchDirectFlight
    print("--- 开始执行 Task 2a: 搜索直飞航班 ---")
    try:
        start_time= time.time()
        origin = extracted_info.get("origin")
        destination = extracted_info.get("destination")
        date = extracted_info.get("date")
        # 调用搜索工具
        direct_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, date)
        import json
        direct_flights = json.loads(direct_flights_str)
        print(f"搜索到 {len(direct_flights)} 个直飞航班。")
        print("--- Task 2a 执行完毕 ---")

        return {
            "dag_id": dag_id,
            "direct_flights": direct_flights,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task2a_search_direct_flight 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "start_time": start_time,
            "end_time": time.time(),
            "direct_flights": None,
            "result": f"task2a_search_direct_flight 发生错误: {str(e)}"
        }

def task2b_search_onestop_flight(dag_id, backend_data, extracted_info):
    """
    工作流的第2.2步：[Tool] 搜索中转航班。
    """
    print("--- 开始执行 Task 2b: 搜索中转航班 ---")
    from baseline.utils.tbench_tools.airline.search_onestop_flight import SearchOnestopFlight
    try:
        import time
        import json
        start_time= time.time()
        origin = extracted_info.get("origin")
        destination = extracted_info.get("destination")
        date = extracted_info.get("date")
        onestop_flights_str = SearchOnestopFlight.invoke(backend_data, origin, destination, date)
        onestop_flights = json.loads(onestop_flights_str)

        print(f"搜索到 {len(onestop_flights)} 个中转行程。")
        print("--- Task 2b 执行完毕 ---")
        return {
            "dag_id": dag_id,
            "onestop_flights": onestop_flights,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task2b_search_onestop_flight 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "onestop_flights": None,
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task2b_search_onestop_flight 发生错误: {str(e)}"
        }

def task2c_get_user_details(dag_id, backend_data, user_id):
    """
    工作流的第2.3步：[Tool] 获取用户详细信息。
    """
    print("--- 开始执行 Task 2c: 获取用户详情 ---")
    try:
        import time
        import json
        from baseline.utils.tbench_tools.airline.get_user_details import GetUserDetails
        start_time= time.time()
        user_details_str = GetUserDetails.invoke(backend_data, user_id)
        user_details= {}
        try:
            user_details = json.loads(user_details_str)
        except Exception as e:
            print(f"错误: {user_details_str}")
        print("获取用户详情成功。")
        print("--- Task 2c 执行完毕 ---")
        return {
            "dag_id": dag_id,
            "user_details": user_details,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task2c_get_user_details 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "user_details": None,
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task2c_get_user_details 发生错误: {str(e)}"
        }

def task3_llm_process_filter_and_decide(args, dag_id, instruction, direct_flights, onestop_flights, vllm_manager= None, backend= "huggingface"):
    """
    工作流的第3步：[LLM] 筛选与决策。
    使用LLM根据用户完整指令和所有约束，从候选航班中选择最终行程。
    采用与file/task.py相同的大模型推理模式。
    """
    import os
    import gc
    import json
    from typing import List, Dict
    import time
    print("--- 开始执行 Task 3: LLM 筛选与决策 ---")
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
        import time
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
        start_time= time.time()
        print(f"✅ LLM航班筛选决策开始....")
        # 从上下文中获取必要信息
        api_url = args.api_url
        api_key = args.api_key
        
        use_online_model= args.use_online_model
        model_folder= args.model_folder
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        temperature, max_token, repetition_penalty, top_p= None, None, None, None
        if not use_online_model:
            temperature= args.temperature
            max_token= args.max_token
            top_p= args.top_p
            repetition_penalty= args.repetition_penalty
        
        # 从上下文中组合直飞和中转航班，形成统一的候选列表 all_candidate_journeys
        all_candidates = []
        # 将直飞航班包装成单步行程，以统一格式
        for flight in direct_flights:
            all_candidates.append([flight])
        for journey in onestop_flights:
            all_candidates.append(journey)
        if not all_candidates:
            print("错误: 没有候选航班可供筛选。")
            return {
                "dag_id": dag_id,
                "status": "failed",
                "information": "错误: 没有候选航班可供筛选。",
                "selected_journey": None,
                "start_time": start_time,
                "end_time": time.time()
            }
        print(f"共有 {len(all_candidates)} 个候选行程（包含直飞和中转）供LLM决策。")

        # 构建提示
        prompt = f"""
        You are a professional and meticulous flight booking decision assistant.
        Your task is to select the single most suitable itinerary from the list of candidate itineraries provided below, based on the user's original request and all constraints.

        # User's original request
        "{instruction}"

        # List of candidate itineraries (JSON format)
        Each itinerary is a list containing one or more flights.
        {json.dumps(all_candidates, indent=2)}

        # Your task
        1. Carefully read and understand each of the user's requirements, including but not limited to: time preferences, price preferences (e.g., "cheapest"), airline preferences, number of layovers, etc.
        2. Strictly filter the candidate itineraries according to these requirements.
        3. From the itineraries that meet all conditions, select the single best option.
        4. Return your chosen itinerary in strict JSON format, without any additional explanations, comments, or text. The returned JSON object should be one element from the candidate itinerary list (a list containing one or more flight dictionaries).
        5. If no itinerary can satisfy the user's core requirements, return an empty JSON list `[]`.

        # JSON output
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
        
        # 调用在线LLM API
        if use_online_model:
            llm_output= query_llm_online(
                api_url= api_url, 
                api_key= api_key, 
                payload= payload,
                tokenizer_path= tokenizer_path)
        elif backend== "vllm":
            llm_output= query_vllm_model(api_url= vllm_manager.get_next_endpoint("qwen3-32b"), model_alias= "qwen3-32b", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)
        else:
            llm_output= query_llm(model_folder= model_folder, model= "Qwen/Qwen3-32B", messages= [{"role": "user", "content": prompt}], temperature= temperature, max_token= max_token, top_p= top_p, repetition_penalty= repetition_penalty)

        json_str = _extract_json_from_llm_output(llm_output)
        selected_journey = [] # 默认值改为空列表
        try:
            # 解析提取出的JSON字符串
            selected_journey = json.loads(json_str)
        except Exception as e:
            print(f"解析提取的JSON时出错: {e}")     
            
        print(f"LLM原始输出: {llm_output}")
        if isinstance(selected_journey, dict):
            selected_journey = [selected_journey]
        # 如果它不是列表（比如是None或字符串），则置为空列表以防后续任务出错
        elif not isinstance(selected_journey, list):
            selected_journey = []
        # 现在 selected_journey 变量一定是列表格式
        print(f"LLM最终选择的行程(格式化后): {selected_journey}")

        try:
            flight_numbers = [str(f.get('flight_number', 'UNKNOWN')) for f in selected_journey]
            print(f"LLM最终选择的行程: {flight_numbers}")
        except (TypeError, AttributeError) as e:
            print(f"无法解析航班号: {e}")
        return {
            "dag_id": dag_id,
            "selected_journey": selected_journey,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task3_filter_and_decide 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed",
            "selected_journey": None,
            "start_time": start_time,
            "end_time": time.time(),
            "result": f"task3_filter_and_decide 发生错误: {str(e)}"
        }

def task4_book_reservation(args, dag_id, user_id, backend_data, extracted_info, selected_journey, user_details):
    """
    工作流的第4步：[Tool] 执行订票。
    组装所有信息，调用最终的订票工具。
    """
    print("--- 开始执行 Task 4: 执行订票 ---")
    import time
    import json
    from baseline.utils.tbench_tools.airline.book_reservation import BookReservation
    try:
        start_time= time.time()
        # 从上下文中收集所有需要的信息

        # 1. 准备乘客信息 - 支持多乘客
        passengers = []
        # 从用户指令中提取乘客数量，如果没有指定则默认为1
        num_passengers = extracted_info.get("num_passengers", 1)
        
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

        # 2. 准备航班信息
        flights_for_booking = []
        for flight in selected_journey:
            flights_for_booking.append({
                "flight_number": flight["flight_number"],
                "date": extracted_info["date"]  # 使用从用户指令中提取的日期
            })

        # 3. 准备和计算支付信息
        cabin = extracted_info.get("cabin")
        total_price = sum(flight['prices'][cabin] for flight in selected_journey) * len(passengers)
        total_baggages = extracted_info.get("baggages", 0) * len(passengers)

        # 行李费计算
        nonfree_baggages = total_baggages - len(passengers) if total_baggages > len(passengers) else 0
        total_price += 50 * nonfree_baggages  # 每件非免费行李$50

        if extracted_info.get("insurance") == "yes":
            total_price += 30 * len(passengers)

        # 支付逻辑：优先用礼品券/证书，然后用信用卡
        payment_methods_for_booking = []
        remaining_balance = total_price

        user_payment_methods = user_details.get("payment_methods", {})
        # 优先使用证书
        for pm_id, pm_details in user_payment_methods.items():
            if pm_details["source"] == "certificate" and remaining_balance > 0:
                amount_to_use = min(remaining_balance, pm_details["amount"])
                payment_methods_for_booking.append({"payment_id": pm_id, "amount": amount_to_use})
                remaining_balance -= amount_to_use

        # 然后使用礼品卡
        for pm_id, pm_details in user_payment_methods.items():
            if pm_details["source"] == "gift_card" and remaining_balance > 0:
                amount_to_use = min(remaining_balance, pm_details["amount"])
                payment_methods_for_booking.append({"payment_id": pm_id, "amount": amount_to_use})
                remaining_balance -= amount_to_use

        # 最后使用信用卡支付剩余部分
        if remaining_balance > 0:
            credit_card_id = None
            for pm_id, pm_details in user_payment_methods.items():
                if pm_details["source"] == "credit_card":
                    credit_card_id = pm_id
                    break
            if credit_card_id:
                payment_methods_for_booking.append({"payment_id": credit_card_id, "amount": remaining_balance})
                remaining_balance = 0

        if remaining_balance > 0:
            return {
                "dag_id": dag_id,
                "status": "订票失败", 
                "result": "支付失败：用户没有足够的支付方式或余额来完成支付。",
                "start_time": start_time,
                "end_time": time.time()
                }

        # 组装最终参数
        booking_args = {
            "user_id": user_id,
            "origin": extracted_info.get("origin"),
            "destination": extracted_info.get("destination"),
            "flight_type": extracted_info.get("flight_type", "one_way"),  # 支持往返
            "cabin": cabin,
            "flights": flights_for_booking,
            "passengers": passengers,
            "payment_methods": payment_methods_for_booking,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": extracted_info.get("insurance", "no")
        }

        # 调用订票工具
        result = BookReservation.invoke(backend_data, **booking_args)
        print(f"订票工具调用结果: {result}")
        print("--- Task 4 执行完毕 ---")
        return {
            "dag_id": dag_id,
            "status": "订票完成",
            "final_answer": result,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task4_book_reservation 发生错误: {str(e)}")
        return {
            "dag_id": dag_id,
            "status": "failed", 
            "final_answer": f"task4_book_reservation 发生错误: {str(e)}",
            "start_time": start_time,
            "end_time": time.time()
        }