# ==============================================================================
#  重构后的 Task 函数 (纯函数式数据流, 适配 Simplified Vision DAG)
# ==============================================================================

# 调度器装饰器（如 @io, @gpu）保持不变。
# from agentos.scheduler import cpu, gpu, io

# @io(mem=3)
def task1_start_receive_task(args, dag_id, question, supplementary_files):
    """
    Task 1: 接收任务，确认核心参数。
    """
    import time
    try:
        print("✅ Task 1: Starting... Verifying initial parameters.")
        start_time = time.time()
        
        print(f"  -> Received dag_id: {dag_id}")
        print(f"  -> Received question: '{question[:100]}...'")
        print(f"  -> Received {len(supplementary_files)} supplementary files.")

        return {
            "dag_id": dag_id, "question": question, "supplementary_files": supplementary_files,
            "args": args, "start_time": start_time, "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task1_start_receive_task failed: {str(e)}")
        return {
            "dag_id": dag_id, "question": question, "supplementary_files": None,
            "args": args, "start_time": time.time(), "end_time": time.time()
        }

# @io(mem=5)
def task2_read_file(args, dag_id, question, supplementary_files):
    """
    Task 2: 读取所有图像文件。
    """
    import time
    try:
        print("✅ Task 2: Reading image files...")
        start_time = time.time()
        if not supplementary_files:
            raise ValueError("supplementary_files is None or empty.")
            
        image_files = {k: v for k, v in supplementary_files.items() if k.lower().endswith(('.png', '.jpg', '.jpeg'))}
        all_images = [{"file_name": name, "content": content} for name, content in image_files.items()]

        print(f"✅ Task 2: Finished. Read {len(all_images)} images.")
        return {
            "dag_id": dag_id, "question": question, "all_images": all_images,
            "args": args, "start_time": start_time, "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task2_read_file failed: {str(e)}")
        return {
            "dag_id": dag_id, "question": question, "all_images": None,
            "args": args, "start_time": time.time(), "end_time": time.time()
        }

# @io(mem=8)
def task3_file_process(args, dag_id, question, all_images):
    """
    Task 3: 增强图像质量并标准化尺寸。
    """
    import time
    import numpy as np
    import cv2
    from PIL import Image
    from io import BytesIO
    try:
        print("✅ Task 3: Enhancing and standardizing images...")
        start_time = time.time()
        if not all_images:
            raise ValueError("all_images is None or empty.")

        processed_images = []
        MAX_DIMENSION = 1024

        for image_info in all_images:
            img_array = np.frombuffer(image_info['content'], np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: continue

            h, w, _ = img.shape
            if h > MAX_DIMENSION or w > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                img = cv2.filter2D(img, -1, kernel)
            
            enhanced_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(enhanced_img_rgb)
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG')
            
            processed_images.append({"file_name": image_info['file_name'], "content": buffer.getvalue()})

        print(f"✅ Task 3: Standardization and enhancement complete. Processed {len(processed_images)} images.")
        return {
            "dag_id": dag_id, "question": question, "processed_images": processed_images,
            "args": args, "start_time": start_time, "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task3_file_process failed: {str(e)}")
        return {
            "dag_id": dag_id, "question": question, "processed_images": None,
            "args": args, "start_time": time.time(), "end_time": time.time()
        }

def task4a_vision_llm_process(args, dag_id, question, processed_images, vllm_manager= None, backend= "huggingface"):
    """
    Task 4a: 使用 VLM 处理图像数据，生成答案。
    """
    import time
    import asyncio
    import aiohttp
    import base64
    from typing import List, Dict, Any, Optional, Tuple
    start_time = time.time()
    async def _query_vlm_batch_async(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有 VLM 请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        async def _query_single(session: aiohttp.ClientSession, request: Dict[str, Any]) -> str:
            """异步发送单个VLM请求的coroutine。"""
            # 1. 将图像字节流编码为Base64
            base64_image = base64.b64encode(request['content']).decode('utf-8')
            
            # 2. 构建符合OpenAI多模态规范的payload
            payload = {
                "model": model_alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": request['prompt']
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"VLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, req) for req in image_requests]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vlm_batch_via_service(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> List[str]:
        """
        [新增] 使用vLLM服务批量处理图像描述任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的本地 query_vlm_batch 兼容。
        """
        if not image_requests:
            return []

        print(f"  -> Starting VLM batch processing via service: {len(image_requests)} images concurrently.")
        
        # 运行异步主函数
        descriptions = asyncio.run(_query_vlm_batch_async(
            api_url, model_alias, image_requests, temperature, max_token, top_p, repetition_penalty
        ))
        
        print("  -> VLM batch processing via service finished.")
        return descriptions
    
    def _vlm_processing_worker(task_id, args, question, all_processed_images, slice_start, slice_end):
        """
        一个通用的VLM工作函数，处理数据的一个切片。它包含了所有依赖项以确保独立性。
        """
        import os
        import gc
        import math
        import torch
        from PIL import Image
        from io import BytesIO
        from typing import List, Dict, Any

        def query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int) -> List[str]:
            from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            model_path = os.path.join(model_folder, model_name)
            tokenizer, processor, model = None, None, None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True)
                
                all_responses = []
                num_reqs = len(image_requests)
                for i in range(0, num_reqs, batch_size):
                    batch_reqs = image_requests[i:i + batch_size]
                    texts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req['prompt']}]}], tokenize=False, add_generation_prompt=True) for req in batch_reqs]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in batch_reqs]
                    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
                    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in decoded])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"✅ Task {task_id}: Starting VLM processing for slice [{slice_start}:{slice_end}]...")
        if not all_processed_images: return []

        # 安全地处理切片
        batch_to_process = all_processed_images[slice_start:slice_end] if slice_end is not None else all_processed_images[slice_start:]
        if not batch_to_process: return []

        image_requests = [{"prompt": f"Carefully observe this image and answer the following question: {question}", "content": img["content"], "file_name": img["file_name"]} for img in batch_to_process]
        if backend == "vllm":
            results = query_vlm_batch_via_service(
                vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), "qwen2.5-vl-32b", image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                batch_size=getattr(args, "vlm_batch_size", 8)
            )
        else:
            results = query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct", image_requests=image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_answers = [{"file_name": image_requests[i]["file_name"], "answer": answer} for i, answer in enumerate(results)]
        print(f"✅ Task {task_id}: Processing complete. Generated {len(final_answers)} answers.")
        return final_answers
    try:
        total = len(processed_images) if processed_images else 0
        idx1 = int(total * 0.2)
        results = _vlm_processing_worker("4a", args, question, processed_images, 0, idx1)
        return {
            "dag_id": dag_id,
            "final_answers_a": results,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task4a_vision_llm_process failed: {e}")
        return {
            "dag_id": dag_id,
            "final_answers_a": None,
            "start_time": start_time,
            "end_time": time.time()
        }
    
def task4b_vision_llm_process(args, dag_id, question, processed_images, vllm_manager= None, backend= "huggingface"):
    import time
    import asyncio
    import aiohttp
    import base64
    from typing import List, Dict, Any, Optional, Tuple
    start_time = time.time()
    async def _query_vlm_batch_async(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有 VLM 请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        async def _query_single(session: aiohttp.ClientSession, request: Dict[str, Any]) -> str:
            """异步发送单个VLM请求的coroutine。"""
            # 1. 将图像字节流编码为Base64
            base64_image = base64.b64encode(request['content']).decode('utf-8')
            
            # 2. 构建符合OpenAI多模态规范的payload
            payload = {
                "model": model_alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": request['prompt']
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"VLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, req) for req in image_requests]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vlm_batch_via_service(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> List[str]:
        """
        [新增] 使用vLLM服务批量处理图像描述任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的本地 query_vlm_batch 兼容。
        """
        if not image_requests:
            return []

        print(f"  -> Starting VLM batch processing via service: {len(image_requests)} images concurrently.")
        
        # 运行异步主函数
        descriptions = asyncio.run(_query_vlm_batch_async(
            api_url, model_alias, image_requests, temperature, max_token, top_p, repetition_penalty
        ))
        
        print("  -> VLM batch processing via service finished.")
        return descriptions
    
    def _vlm_processing_worker(task_id, args, question, all_processed_images, slice_start, slice_end):
        """
        一个通用的VLM工作函数，处理数据的一个切片。它包含了所有依赖项以确保独立性。
        """
        import os
        import gc
        import math
        import torch
        from PIL import Image
        from io import BytesIO
        from typing import List, Dict, Any

        def query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int) -> List[str]:
            from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            model_path = os.path.join(model_folder, model_name)
            tokenizer, processor, model = None, None, None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False,trust_remote_code=True)
                
                all_responses = []
                num_reqs = len(image_requests)
                for i in range(0, num_reqs, batch_size):
                    batch_reqs = image_requests[i:i + batch_size]
                    texts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req['prompt']}]}], tokenize=False, add_generation_prompt=True) for req in batch_reqs]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in batch_reqs]
                    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
                    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in decoded])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"✅ Task {task_id}: Starting VLM processing for slice [{slice_start}:{slice_end}]...")
        if not all_processed_images: return []

        # 安全地处理切片
        batch_to_process = all_processed_images[slice_start:slice_end] if slice_end is not None else all_processed_images[slice_start:]
        if not batch_to_process: return []

        image_requests = [{"prompt": f"Carefully observe this image and answer the following question: {question}", "content": img["content"], "file_name": img["file_name"]} for img in batch_to_process]
        if backend == "vllm":
            results = query_vlm_batch_via_service(
                vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), "qwen2.5-vl-32b", image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                batch_size=getattr(args, "vlm_batch_size", 8)
            )
        else:
            results = query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct", image_requests=image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_answers = [{"file_name": image_requests[i]["file_name"], "answer": answer} for i, answer in enumerate(results)]
        print(f"✅ Task {task_id}: Processing complete. Generated {len(final_answers)} answers.")
        return final_answers
    try:
        total = len(processed_images) if processed_images else 0
        idx1 = int(total * 0.2)
        idx2 = int(total * 0.4)
        results = _vlm_processing_worker("4b", args, question, processed_images, idx1, idx2)
        return {
            "dag_id": dag_id,
            "final_answers_b": results,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task4b_vision_llm_process failed: {e}")
        return {
            "dag_id": dag_id,
            "final_answers_b": None,
            "start_time": start_time,
            "end_time": time.time()
        }

# @gpu(gpu_mem=80000)
def task4c_vision_llm_process(args, dag_id, question, processed_images, vllm_manager= None, backend= "huggingface"):
    import time
    import asyncio
    import aiohttp
    import base64
    from typing import List, Dict, Any, Optional, Tuple
    start_time = time.time()
    async def _query_vlm_batch_async(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有 VLM 请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        async def _query_single(session: aiohttp.ClientSession, request: Dict[str, Any]) -> str:
            """异步发送单个VLM请求的coroutine。"""
            # 1. 将图像字节流编码为Base64
            base64_image = base64.b64encode(request['content']).decode('utf-8')
            
            # 2. 构建符合OpenAI多模态规范的payload
            payload = {
                "model": model_alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": request['prompt']
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"VLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, req) for req in image_requests]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vlm_batch_via_service(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> List[str]:
        """
        [新增] 使用vLLM服务批量处理图像描述任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的本地 query_vlm_batch 兼容。
        """
        if not image_requests:
            return []

        print(f"  -> Starting VLM batch processing via service: {len(image_requests)} images concurrently.")
        
        # 运行异步主函数
        descriptions = asyncio.run(_query_vlm_batch_async(
            api_url, model_alias, image_requests, temperature, max_token, top_p, repetition_penalty
        ))
        
        print("  -> VLM batch processing via service finished.")
        return descriptions
    
    def _vlm_processing_worker(task_id, args, question, all_processed_images, slice_start, slice_end):
        """
        一个通用的VLM工作函数，处理数据的一个切片。它包含了所有依赖项以确保独立性。
        """
        import os
        import gc
        import math
        import torch
        from PIL import Image
        from io import BytesIO
        from typing import List, Dict, Any

        def query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int) -> List[str]:
            from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            model_path = os.path.join(model_folder, model_name)
            tokenizer, processor, model = None, None, None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True)
                
                all_responses = []
                num_reqs = len(image_requests)
                for i in range(0, num_reqs, batch_size):
                    batch_reqs = image_requests[i:i + batch_size]
                    texts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req['prompt']}]}], tokenize=False, add_generation_prompt=True) for req in batch_reqs]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in batch_reqs]
                    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
                    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in decoded])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"✅ Task {task_id}: Starting VLM processing for slice [{slice_start}:{slice_end}]...")
        if not all_processed_images: return []

        # 安全地处理切片
        batch_to_process = all_processed_images[slice_start:slice_end] if slice_end is not None else all_processed_images[slice_start:]
        if not batch_to_process: return []

        image_requests = [{"prompt": f"Carefully observe this image and answer the following question: {question}", "content": img["content"], "file_name": img["file_name"]} for img in batch_to_process]
        if backend == "vllm":
            results = query_vlm_batch_via_service(
                vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), "qwen2.5-vl-32b", image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                batch_size=getattr(args, "vlm_batch_size", 8)
            )
        else:
            results = query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct", image_requests=image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024), top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_answers = [{"file_name": image_requests[i]["file_name"], "answer": answer} for i, answer in enumerate(results)]
        print(f"✅ Task {task_id}: Processing complete. Generated {len(final_answers)} answers.")
        return final_answers
    try:
        total = len(processed_images) if processed_images else 0
        idx2 = int(total * 0.4)
        idx3 = int(total * 0.6)
        results = _vlm_processing_worker("4c", args, question, processed_images, idx2, idx3)
        return {
            "dag_id": dag_id,
            "final_answers_c": results,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task4c_vision_llm_process failed: {e}")
        return {
            "dag_id": dag_id,
            "final_answers_c": None,
            "start_time": start_time,
            "end_time": time.time()
        }

# @gpu(gpu_mem=80000)
def task4d_vision_llm_process(args, dag_id, question, processed_images, vllm_manager= None, backend= "huggingface"):
    import time
    import asyncio
    import aiohttp
    import base64
    from typing import List, Dict, Any, Optional, Tuple
    start_time = time.time()
    async def _query_vlm_batch_async(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float,
        max_token: int,
        top_p: float,
        repetition_penalty: float
    ) -> List[str]:
        """[新增] 使用 aiohttp 并发执行所有 VLM 请求的异步核心。"""
        chat_url = f"{api_url.strip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        async def _query_single(session: aiohttp.ClientSession, request: Dict[str, Any]) -> str:
            """异步发送单个VLM请求的coroutine。"""
            # 1. 将图像字节流编码为Base64
            base64_image = base64.b64encode(request['content']).decode('utf-8')
            
            # 2. 构建符合OpenAI多模态规范的payload
            payload = {
                "model": model_alias,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": request['prompt']
                            }
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_token,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            try:
                async with session.post(chat_url, json=payload, timeout=3600) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return response_data['choices'][0]['message']['content'].strip()
            except Exception as e:
                error_msg = f"VLM async request failed: {str(e)}"
                print(f"[bold red]{error_msg}")
                return error_msg

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [_query_single(session, req) for req in image_requests]
            all_responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(resp) for resp in all_responses]

    def query_vlm_batch_via_service(
        api_url: str,
        model_alias: str,
        image_requests: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_token: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs # 添加**kwargs以忽略不用的参数如model_folder, batch_size
    ) -> List[str]:
        """
        [新增] 使用vLLM服务批量处理图像描述任务。
        - 通过并发API请求实现批处理。
        - 接口与现有的本地 query_vlm_batch 兼容。
        """
        if not image_requests:
            return []

        print(f"  -> Starting VLM batch processing via service: {len(image_requests)} images concurrently.")
        
        # 运行异步主函数
        descriptions = asyncio.run(_query_vlm_batch_async(
            api_url, model_alias, image_requests, temperature, max_token, top_p, repetition_penalty
        ))
        
        print("  -> VLM batch processing via service finished.")
        return descriptions
    
    def _vlm_processing_worker(task_id, args, question, all_processed_images, slice_start, slice_end):
        """
        通用的 VLM 工作函数，处理一段图片切片，批量推理并返回结果。
        """
        import os
        import gc
        import torch
        from PIL import Image
        from io import BytesIO
        from typing import List, Dict, Any

        def query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int) -> List[str]:
            from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            model_path = os.path.join(model_folder, model_name)
            tokenizer, processor, model = None, None, None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda", offload_state_dict= False, trust_remote_code=True
                )
                all_responses = []
                num_reqs = len(image_requests)
                for i in range(0, num_reqs, batch_size):
                    batch_reqs = image_requests[i:i + batch_size]
                    texts = [
                        processor.apply_chat_template(
                            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req['prompt']}]}],
                            tokenize=False, add_generation_prompt=True
                        ) for req in batch_reqs
                    ]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in batch_reqs]
                    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_token,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in decoded])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()

        print(f"✅ Task {task_id}: Starting VLM processing for slice [{slice_start}:{slice_end}]...")
        if not all_processed_images:
            print(f"  -> Task {task_id}: No images to process.")
            return []

        batch_to_process = all_processed_images[slice_start:slice_end] if slice_end is not None else all_processed_images[slice_start:]
        if not batch_to_process:
            print(f"  -> Task {task_id}: Batch is empty after slicing.")
            return []

        image_requests = [
            {
                "prompt": f"Carefully observe this image and answer the following question: {question}",
                "content": img["content"],
                "file_name": img["file_name"]
            }
            for img in batch_to_process
        ]
        if backend == "vllm":
            results = query_vlm_batch_via_service(
                vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), "qwen2.5-vl-32b", image_requests,
                temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                batch_size=getattr(args, "vlm_batch_size", 8)
            )
        else:
            results = query_vlm_batch(
                model_folder=args.model_folder,
                model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests,
                temperature=getattr(args, "temperature", 1.0),
                max_token=getattr(args, "max_token", 1024),
                top_p=getattr(args, "top_p", 0.95),
                repetition_penalty=getattr(args, "repetition_penalty", 1.1),
                batch_size=getattr(args, "vlm_batch_size", 8)
            )

        final_answers = [
            {"file_name": image_requests[i]["file_name"], "answer": answer}
            for i, answer in enumerate(results)
        ]
        print(f"✅ Task {task_id}: Processing complete. Generated {len(final_answers)} answers.")
        return final_answers
    try:
        total = len(processed_images) if processed_images else 0
        idx3 = int(total * 0.6)
        results = _vlm_processing_worker("4d", args, question, processed_images, idx3, None) # 切片到末尾
        return {
            "dag_id": dag_id,
            "final_answers_d": results,
            "start_time": start_time,
            "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task4d_vision_llm_process failed: {e}")
        return {
            "dag_id": dag_id,
            "final_answers_d": None,
            "start_time": start_time,
            "end_time": time.time()
        }


# @io(mem=5)
def task4_merge_results(args, dag_id, final_answers_a, final_answers_b, final_answers_c, final_answers_d):
    """
    Task 5: 合并来自所有并行VLM工作者的结果。
    """
    import time
    try:
        print("✅ Task 5: Merging parallel VLM results...")
        start_time = time.time()
        
        # 安全地合并列表，处理可能为None的情况
        all_final_answers = (final_answers_a or []) + (final_answers_b or []) + \
                            (final_answers_c or []) + (final_answers_d or [])
        
        print(f"✅ Task 5: Merging complete. Total items: {len(all_final_answers)}.")
        return {
            "dag_id": dag_id, "all_final_answers": all_final_answers,
            "args": args, "start_time": start_time, "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task5_merge_results failed: {e}")
        return {
            "dag_id": dag_id, "all_final_answers": None,
            "args": args, "start_time": time.time(), "end_time": time.time()
        }

# @io(mem=3)
def task5_output_final_answer(args, dag_id, all_final_answers):
    """
    Task 6: 格式化并输出最终答案。
    """
    import time
    try:
        print("✅ Task 6: Formatting final output.")
        start_time = time.time()
        
        if not all_final_answers:
            final_answer_text = "Workflow finished, but no answers were generated."
        else:
            final_answer_text = "\n\n".join([f"Answer for {d['file_name']}:\n{d['answer']}" for d in all_final_answers])
        
        print(f"🏁 Final Answer for DAG {dag_id} generated.")
        return {
            "dag_id": dag_id, "final_answer": final_answer_text,
            "start_time": start_time, "end_time": time.time()
        }
    except Exception as e:
        print(f"❌ task6_output_final_answer failed: {str(e)}")
        return {
            "dag_id": dag_id, "final_answer": f"Error during final output formatting: {e}",
            "start_time": time.time(), "end_time": time.time()
        }
