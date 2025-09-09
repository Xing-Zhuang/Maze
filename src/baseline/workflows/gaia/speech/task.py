def task1_speech_process(args, dag_id, question, supplementary_files):
    """
    Reads read.jsonl, finds the task based on task_id, reads the corresponding audio file,
    and stores task info and file content in context.
    """
    from pydub import AudioSegment
    from io import BytesIO
    import numpy as np
    import librosa
    import time
    import librosa
    from pydub import AudioSegment # 导入pydub 
    def decode_audio_to_array(audio_bytes: bytes, target_sr: int = 16000):
        """
        健壮的音频解码函数：将任意格式的音频字节解码为Numpy数组和采样率。
        """
        print("正在使用pydub解码音频...")
        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
        
        wav_buffer = BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        print("解码为WAV格式成功。正在使用librosa加载...")
        
        # librosa现在可以从WAV格式的缓冲区中安全地加载数据
        y, sr = librosa.load(wav_buffer, sr=target_sr)
        return y, sr
    import time
    try:
        start_time= time.time()
        file_name = ""
        content = None
        
        if supplementary_files:
            file_name = next(iter(supplementary_files.keys()), "")
            content = next(iter(supplementary_files.values()), None)
            print(f"  -> Found single supplementary file to process: '{file_name}'")
        else:
            print("  -> No supplementary files found in context.")
        audio_array, _= decode_audio_to_array(content, target_sr= 16000)
        
        # 存储任务信息和文件内容
        return {
            "dag_id": dag_id,
            "question": question, 
            "file_content": audio_array,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"task1 发生错误: {str(e)}")
        raise e

def task2_speech_recognition(args, dag_id, question, file_content):
    """
    Processes the audio file using Whisper for speech recognition.
    """
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration
    )   
    import os
    import time
    import gc, torch
    try:
        start_time= time.time()
        model_path= os.path.join(args.model_folder, "whisper_models/whisper-medium")
        print("加载 Whisper 模型中...")
        processor= WhisperProcessor.from_pretrained(model_path)
        model= WhisperForConditionalGeneration.from_pretrained(model_path)
        model.generation_config.forced_decoder_ids= None
        print("已从上下文中获取解码后的音频数组。")

        # 使用获取到的数组和采样率进行处理
        input_features = processor(
            file_content, # 使用本地加载的音频数组
            sampling_rate=16000, # 使用本地加载的采样率
            return_tensors="pt"
        ).input_features

        predicted_ids = model.generate(input_features)
        processed_content = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print("🔊 Whisper 识别结果：", processed_content)
        del model
        del processor
        del input_features
        del predicted_ids
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return {
            "dag_id": dag_id,
            "question": question,
            "processed_content": processed_content,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task2_speech_recognition 发生错误: {str(e)}")
        import traceback
        print(f"task2_speech_recognition 错误堆栈: {traceback.format_exc()}")
        raise e

def task3_llm_process_qwen(args, dag_id, question, processed_content, vllm_manager= None, backend= "huggingface"):
    """
    Processes the file content using LLM based on the question.
    """
    import gc
    import torch
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
    def query_llm(model, model_folder, messages, temperature= 0.6, max_token= 1024, top_p= 0.9, repetition_penalty= 1.1):
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载预训练模型和分词器
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map= "cuda")
        local_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_folder, model),
            torch_dtype="float16",
            device_map="cuda",
            low_cpu_mem_usage=True,
            offload_state_dict= False
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
    
    def query_llm_online(api_url, api_key, payload)-> str:
        """Call SiliconFlow API to get LLM response"""
        import requests
        from rich.console import Console

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
        start_time= time.time()
        print(f"✅ qwen处理开始....")
        if not question:
             raise ValueError(f"任务 {dag_id} 缺少 Question")

        # 构建提示
        # 生成推理模型的特征
        prompt= (
            "#Background#\n"
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
            "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
            "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
            f"#Question#\n{question}\n"
            "#Extracted text from speech#\n"
            f"{processed_content[:8000]}\n"
        )
        payload= {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_token,
            "enable_thinking": False,
            "response_format": {"type": "text"}               
        }
        answer= None
        if args.use_online_model:
            answer= query_llm_online(
                api_url= args.api_url,
                api_key= args.api_key, 
                payload= payload)
        elif backend== "vllm":
            vllm_api_url= vllm_manager.get_next_endpoint("qwen3-32b")
            answer= query_vllm_model(api_url= vllm_api_url, model_alias= "qwen3-32b", messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        else:
            answer= query_llm(model= "Qwen/Qwen3-32B", model_folder= args.model_folder, messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        return {
            "dag_id": dag_id,
            "question": question,
            "answer_qwen": answer,
            "start_time": start_time,
            "end_time": time.time()            
        }
    except Exception as e:
        print(f"task3_llm_process_qwen 发生错误: {str(e)}")
        raise e

def task4_llm_process_deepseek(args, dag_id, question, processed_content, vllm_manager= None, backend= "huggingface"):
    """
    Processes the file content using LLM based on the question.
    """
    import time
    import gc
    import torch
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
    def query_llm(model, model_folder, messages, temperature= 0.6, max_token= 1024, top_p= 0.9, repetition_penalty= 1.1):
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载预训练模型和分词器
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map= "cuda")
        local_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_folder, model),
            torch_dtype="float16",
            device_map="cuda",
            low_cpu_mem_usage=True,
            offload_state_dict= False
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
    
    def query_llm_online(api_url, api_key, payload)-> str:
        """Call SiliconFlow API to get LLM response"""
        import requests
        from rich.console import Console

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
        start_time= time.time()
        print(f"✅ deepseek处理开始....")
        if not question:
             raise ValueError(f"任务 {dag_id} 缺少 Question")

        # 构建提示
        # 生成推理模型的特征
        prompt= (
            "#Background#\n"
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
            "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
            "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
            f"#Question#\n{question}\n"
            "#Extracted text from speech#\n"
            f"{processed_content[:8000]}\n"
        )
        payload= {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_token,
            "response_format": {"type": "text"}               
        }
        answer= None
        if args.use_online_model:
            answer= query_llm_online(
                api_url= args.api_url,
                api_key= args.api_key, 
                payload= payload)
        elif backend== "vllm":
            vllm_api_url= vllm_manager.get_next_endpoint("deepseek-r1-32b")
            answer= query_vllm_model(api_url= vllm_api_url, model_alias= "deepseek-r1-32b", messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        else:
            answer= query_llm(model= "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", model_folder= args.model_folder, messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        return {
            "dag_id": dag_id,
            "question": question,
            "answer_deepseek": answer,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task4_llm_process_deepseek 发生错误: {str(e)}")
        raise e


def task5_fuse_llm_answer(args, dag_id, question, answer_qwen, answer_deepseek, vllm_manager= None, backend= "huggingface"):
    """
    Fusion Task: Takes answers from multiple experts, analyzes them, and generates a final, synthesized answer.
    """
    import time
    import gc
    import torch
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
    def query_llm(model, model_folder, messages, temperature= 0.6, max_token= 1024, top_p= 0.9, repetition_penalty= 1.1):
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载预训练模型和分词器
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map= "cuda")
        local_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_folder, model),
            torch_dtype="float16",
            low_cpu_mem_usage=True,
            offload_state_dict= False,
            device_map="cuda"
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
      
    def query_llm_online(api_url, api_key, payload)-> str:
        """Call SiliconFlow API to get LLM response"""
        import requests
        from rich.console import Console

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
        start_time= time.time()
        print("✅ Task 4 (Fusion): Starting answer synthesis...")
        # 2. 构建一个用于融合的 prompt
        prompt = (
            "You are a senior editor and a world-class reasoning expert. Your job is to synthesize the answers from two different AI assistants to produce one final, superior answer for the given question.\n\n"
            f"--- Original Question ---\n{question}\n\n"
            f"--- Answer from Assistant 1 (Qwen3) ---\n{answer_qwen}\n\n"
            f"--- Answer from Assistant 2 (DeepSeek) ---\n{answer_deepseek}\n\n"
            "--- Your Task ---\n"
            "Analyze both answers. Identify the strengths and weaknesses of each. Then, combine their best elements, correct any errors, and provide a single, comprehensive, and accurate final answer. Adhere to the final answer format requested in the original prompt.\n\n"
            "Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]."
        )
        
        # 3. 使用一个强大的模型（如Qwen3）来执行融合
        payload= {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_token,
            "response_format": {"type": "text"},            
            "enable_thinking": False
        }
        final_answer= None
        if args.use_online_model:
            final_answer= query_llm_online(
                api_url= args.api_url,
                api_key= args.api_key, 
                payload= payload)
        elif backend== "vllm":
            vllm_api_url= vllm_manager.get_next_endpoint("qwen3-32b")
            final_answer= query_vllm_model(api_url= vllm_api_url, model_alias= "qwen3-32b", messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        else:
            final_answer= query_llm(model= "Qwen/Qwen3-32B", model_folder= args.model_folder, messages= [{"role": "user", "content": prompt}], temperature= args.temperature, max_token= args.max_token, top_p= args.top_p, repetition_penalty= args.repetition_penalty)
        # 4. 准备并返回最终的输出
        print(f"  -> Fusion complete. Final answer generated for DAG {dag_id}.")
        return {
            "dag_id": dag_id,
            "final_answer": final_answer,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task5_fuse_final_answer failed: {str(e)}")
        raise e