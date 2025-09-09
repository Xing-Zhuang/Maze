def task1_obtain_content(args, dag_id, question, supplementary_files):
    """
    Reads read.jsonl, finds the task based on task_id, reads the corresponding image file,
    and stores task info and file content in context.
    """
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
        print("文件读取完成")
        # 存储任务信息和文件内容
        return {
            "dag_id": dag_id,
            "question": question,
            "file_content": content,
            "start_time": start_time,
            "end_time": time.time(),
        }
    except Exception as e:
        print(f"task1_obtain_content 发生错误: {str(e)}")
        raise e

def task2_vlm_process(args, dag_id, question, file_content, vllm_manager= None, backend= "huggingface"):
    """
    Processes the file content using LLM based on the question.
    """
    import base64
    from PIL import Image
    from io import BytesIO
    import gc
    import torch
    def query_vlm_vllm(api_url: str, model_alias: str, prompt: str, img_bytes: bytes, temperature:float= 0.6, max_token:int= 1024, top_p:float= 0.9, repetition_penalty:float= 1.1):
        import requests
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        payload = {
            "model": model_alias,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "temperature": temperature, "max_tokens": max_token, "top_p": top_p, "repetition_penalty": repetition_penalty,
        }
        try:
            print(f"-> [Request] 向 VLM 服务 {chat_url} 发送请求...")
            response = requests.post(chat_url, json=payload, headers=headers, timeout=3600)
            response.raise_for_status()
            response_data = response.json()
            content = response_data['choices'][0]['message']['content'].lstrip()
            return content
        except requests.exceptions.RequestException as e:
            error_msg = f"vLLM VLM request failed: {str(e)}"
            print(f"[bold red]{error_msg}")
            return f"[bold red]{error_msg}"
    def query_vlm(args: dict, model_name: str, prompt: str, img_bytes: bytes)-> str:
        """Call SiliconFlow API to get LLM response"""
        from transformers import AutoProcessor
        import os
        import torch
        from rich.console import Console
        from qwen_vl_utils import process_vision_info
        from transformers import Qwen2_5_VLForConditionalGeneration  # 确保路径正确
        import tempfile
        # --- IMPORTANT: Save image bytes to a temporary file ---
        # Create a temporary file to store the image bytes
        # suffix='.jpg' or '.png' is good practice for image processing libraries
        with tempfile.NamedTemporaryFile(delete= False, suffix= '.jpg') as temp_file:
            temp_file.write(img_bytes)
            temp_image_file= temp_file.name # Get the path of the temporary file
            print(f"Temporary image saved to: {temp_image_file}")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": temp_image_file,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        console= Console()

        try:
            model_path= os.path.join(args.model_folder, model_name)
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cuda",
                offload_state_dict= False
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to('cuda')
            output_ids= model.generate(
                **inputs, 
                max_new_tokens= args.max_token,
                temperature= args.temperature,
                top_p= args.top_p,
                repetition_penalty= args.repetition_penalty)
            response= processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            del model
            del processor
            del inputs
            del output_ids
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            if os.path.exists(temp_image_file):
                os.remove(temp_image_file)
                print(f"Temporary image file deleted: {temp_image_file}")

            return response.lstrip()
        except Exception as e:
            console.print(f"[bold red]API request failed: {str(e)}")
            return f"[bold red]API request failed: {str(e)}"

    def query_vlm_online(api_url: str, api_key: str, model_name: str, prompt: str, img_bytes: bytes)-> str:
        """Call SiliconFlow API to get LLM response"""
        import requests
        from rich.console import Console
        def encode_image_bytes_base64(image_bytes: bytes)-> str:
            return base64.b64encode(image_bytes).decode("utf-8")    
        # 加载预训练模型和分词器
        content= [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_bytes_base64(img_bytes)}"
                }
            }
        ]
        messages= [
            {
                "role": "user",
                "content": content
            }
        ]
        console= Console()
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
        payload= {
            "model": model_name,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": args.max_token,
            "response_format": {"type": "text"}
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

    def downsample_image_if_needed(image_bytes: bytes, max_dimension: int = 1024, max_size_mb: float = 1.5) -> bytes:
        """
        如果图片尺寸或大小超过阈值，则对其进行下采样。

        Args:
            image_bytes (bytes): 原始图片字节流。
            max_dimension (int): 图片允许的最大边长。
            max_size_mb (float): 图片允许的最大文件大小 (MB)。

        Returns:
            bytes: 处理后的图片字节流（可能被下采样）。
        """
        image_size_mb = len(image_bytes) / (1024 * 1024)
        try:
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            # 检查是否需要下采样
            if image_size_mb <= max_size_mb and width <= max_dimension and height <= max_dimension:
                print(f"图片尺寸({width}x{height})和大小({image_size_mb:.2f}MB)均在限制内，无需下采样。")
                return image_bytes
            print(f"图片尺寸({width}x{height})或大小({image_size_mb:.2f}MB)超限，开始下采样...")
            # 使用 thumbnail 方法保持纵横比进行缩放
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            
            # 将缩放后的图片保存到新的字节流中
            output_buffer = BytesIO()
            # 如果图片是PNG等带透明通道的格式，转换为RGB以保存为JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            img.save(output_buffer, format="JPEG", quality=85) # 使用85的质量保存为JPEG
            downsampled_bytes = output_buffer.getvalue()
            new_size_kb = len(downsampled_bytes) / 1024
            print(f"✅ 下采样完成。新尺寸: {img.size}, 新大小: {new_size_kb:.1f} KB")
            return downsampled_bytes

        except Exception as e:
            print(f"⚠️ 图片下采样过程中发生错误: {e}。将返回原始图片字节流。")
            return image_bytes    
    import time
    try:
        start_time= time.time()
        image_bytes= file_content
        # === 在此处集成下采样逻辑 ===
        print(f"原始图片大小: {len(image_bytes) / 1024:.1f} KB")
        # 调用下采样函数，它会根据需要返回原始或缩小后的图片字节
        image_bytes= downsample_image_if_needed(image_bytes, max_dimension= 1024, max_size_mb= 1.5)        
        if not question:
             raise ValueError(f"任务 {dag_id} 缺少 Question")
        # 构建提示
        # 生成推理模型的特征
        print(f"✅ vlm are prepared...")
        prompt= (
            "#Background#\n"
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
            "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
            "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
            f"#Question#\n{question}\n"
        )
        answer= None
        if args.use_online_model:
            answer= query_vlm_online(
                api_url= args.api_url,
                api_key= args.api_key,
                model_name= "Qwen/Qwen2.5-VL-32B-Instruct", 
                prompt= prompt, 
                img_bytes= image_bytes)
        elif backend== "vllm":
            vllm_api_url= vllm_manager.get_next_endpoint("qwen2.5-vl-32b")
            answer= query_vlm_vllm(api_url= vllm_api_url, model_alias= "qwen2.5-vl-32b", prompt= prompt, img_bytes= image_bytes, temperature= 0.6, max_token= args.max_token, top_p= 0.9, repetition_penalty= 1.1)
        else:
            answer= query_vlm(args= args, model_name= "Qwen/Qwen2.5-VL-32B-Instruct", prompt= prompt, img_bytes= image_bytes)
        print(f"✅ vlm processed finished!")
        return {
            "dag_id": dag_id,
            "question": question,
            "vlm_answer": answer,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task2_vlm_process 发生错误: {str(e)}")
        raise e
    
def task3_output_final_answer(args, dag_id, question, vlm_answer):
    """
    Outputs the final answer from the LLM.
    """
    import time
    try:
        start_time= time.time()
        # 使用通用输出格式
        return {
            "dag_id": dag_id,
            "question": question,
            "final_answer": vlm_answer,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"task3_output_final_answer 发生错误: {str(e)}")
        raise e