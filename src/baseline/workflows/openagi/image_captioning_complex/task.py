def task1_start_receive_task(args, dag_id, question, supplementary_files):
    """Task 1: 接收任务，确认核心参数。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 1: Starting... Verifying initial parameters.")
        
        target_language = "English"
        if "german" in question.lower() or "deutsch" in question.lower():
            target_language = "German"
        elif "chinese" in question.lower() or "中文" in question.lower():
            target_language = "Chinese"
            
        print(f"  -> Received dag_id: {dag_id}, question: '{question[:50]}...', lang: {target_language}")
        end_time = time.time()
        return {
            "dag_id": dag_id,
            "question": question,
            "supplementary_files": supplementary_files,
            "target_language": target_language,
            "args": args,
            "start_time": start_time,
            "end_time": end_time
        }
    except Exception as e:
        end_time = time.time()
        print(f"❌ task1_start_receive_task failed: {str(e)}")
        return {
            "dag_id": dag_id, "question": None, "supplementary_files": None,
            "target_language": "English", "args": args,
            "start_time": start_time,
            "end_time": end_time
        }

def task2_read_and_enhance_images(args, dag_id, supplementary_files):
    """Task 2: 读取图像并增强。"""
    import time
    start_time = time.time()
    import numpy as np
    import cv2
    from PIL import Image
    from io import BytesIO
    try:
        print("✅ Task 2: Reading and enhancing images...")
        if not supplementary_files:
            raise ValueError("supplementary_files is empty or None.")

        image_files = {k: v for k, v in supplementary_files.items() if k.lower().endswith(('.png', '.jpg', '.jpeg'))}
        enhanced_images = []

        for file_name, image_content in image_files.items():
            img_array = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            enhanced_img = img
            if blur_score < 100: # Sharpen if blurry
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                enhanced_img = cv2.filter2D(img, -1, kernel)
            
            pil_img = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
            with BytesIO() as img_byte_arr:
                pil_img.save(img_byte_arr, format='JPEG')
                enhanced_images.append({"file_name": file_name, "content": img_byte_arr.getvalue()})

        print(f"✅ Task 2: Finished. Enhanced {len(enhanced_images)} images.")
        end_time = time.time()
        return {"dag_id": dag_id, "enhanced_images": enhanced_images, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task2_read_and_enhance_images failed: {str(e)}")
        return {"dag_id": dag_id, "enhanced_images": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task3a_extract_blip_captions(args, dag_id, enhanced_images):
    """Task 3a (并行): 使用BLIP生成描述。"""
    import time
    start_time = time.time()
    import os, gc
    from io import BytesIO
    from PIL import Image
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    try:
        print("✅ Task 3a: Generating BLIP captions...")
        if not enhanced_images: 
            end_time = time.time()
            return {"dag_id": dag_id, "blip_captions": [], "args": args,
                    "start_time": start_time, "end_time": end_time}
        
        processor, model = None, None
        try:
            model_path = os.path.join(args.model_folder, "blip-image-captioning-large")
            processor = BlipProcessor.from_pretrained(model_path)
            model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda", offload_state_dict= False)
            
            blip_captions = []
            for image_info in enhanced_images:
                raw_image = Image.open(BytesIO(image_info["content"])).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
                out = model.generate(**inputs, max_new_tokens=75)
                caption = processor.decode(out[0], skip_special_tokens=True)
                blip_captions.append({"file_name": image_info["file_name"], "caption": caption})
                del inputs, out
        finally:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

        print("✅ Task 3a: BLIP captioning complete.")
        end_time = time.time()
        return {"dag_id": dag_id, "blip_captions": blip_captions, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task3a_extract_blip_captions failed: {str(e)}")
        return {"dag_id": dag_id, "blip_captions": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task3b_extract_ocr_text(args, dag_id, enhanced_images, target_language):
    """Task 3b (并行): 提取OCR文本。"""
    import time
    start_time = time.time()
    import gc
    import torch
    import easyocr
    try:
        print("✅ Task 3b: Extracting OCR text...")
        if not enhanced_images: 
            end_time = time.time()
            return {"dag_id": dag_id, "ocr_texts": [], "args": args,
                    "start_time": start_time, "end_time": end_time}

        lang_map = {"chinese": "ch_sim", "german": "de"}
        reader_langs = list(set([lang_map.get(target_language.lower(), 'en'), 'en']))
        reader = None
        try:
            reader = easyocr.Reader(reader_langs, gpu=True)
            ocr_texts = []
            for image_info in enhanced_images:
                results = reader.readtext(image_info["content"])
                text = " ".join([res[1] for res in results if res[2] > 0.4])
                ocr_texts.append({"file_name": image_info["file_name"], "text": text})
        finally:
             if reader: del reader; gc.collect(); torch.cuda.empty_cache()

        print("✅ Task 3b: OCR extraction complete.")
        end_time = time.time()
        return {"dag_id": dag_id, "ocr_texts": ocr_texts, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task3b_extract_ocr_text failed: {str(e)}")
        return {"dag_id": dag_id, "ocr_texts": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task4_merge_image_features(args, dag_id, blip_captions, ocr_texts, enhanced_images):
    """Task 4 (合并点): 合并所有图像特征，为VLM处理做准备。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 4: Merging image features...")
        if not all([blip_captions is not None, ocr_texts is not None, enhanced_images is not None]):
            raise ValueError("Upstream data is missing.")

        captions_map = {item['file_name']: item['caption'] for item in blip_captions}
        ocr_map = {item['file_name']: item['text'] for item in ocr_texts}
        
        merged_features_list = []
        for image_info in enhanced_images:
            file_name = image_info['file_name']
            merged_features_list.append({
                "file_name": file_name,
                "content": image_info["content"],
                "caption": captions_map.get(file_name, ""),
                "ocr_text": ocr_map.get(file_name, "")
            })
            
        print(f"✅ Task 4: Feature merging complete for {len(merged_features_list)} images.")
        end_time = time.time()
        return {"dag_id": dag_id, "merged_image_features": merged_features_list, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task4_merge_image_features failed: {str(e)}")
        return {"dag_id": dag_id, "merged_image_features": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task5a_vlm_process(args, dag_id, merged_image_features, target_language, vllm_manager= None, backend= "huggingface"):
    """[Parallel Worker A] 使用VLM为前20%图片生成描述。"""
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
    
    def _vlm_processing_worker(task_name, args, batch_to_process, target_language):
        """一个通用的VLM工作函数，处理一批图像。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math
        from io import BytesIO
        from PIL import Image
        import torch
        from typing import List, Dict, Any, Tuple, Optional

        def _query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            processor, model = None, None
            try:
                print(f"  -> [{task_name}] Loading VLM model...")
                model_path = os.path.join(model_folder, model_name)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    device_map="cuda", offload_state_dict= False, trust_remote_code=True
                )
                
                all_responses = []
                for i in range(0, len(image_requests), batch_size):
                    mini_batch = image_requests[i:i + batch_size]
                    texts = [req['prompt'] for req in mini_batch]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in mini_batch]
                    
                    prompts = []
                    for text in texts:
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

                    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True)
                    
                    responses = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in responses])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  -> [{task_name}] VLM resources released.")

        print(f"⚙️  {task_name}: Starting VLM description generation...")
        if not batch_to_process:
            print(f"  -> {task_name}: Input batch is empty. Skipping.")
            return []

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            # 使用vllm_manager进行批量处理
            descriptions= query_vlm_batch_via_service(
                api_url= vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), model_alias= "qwen2.5-vl-32b",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8))
        else:
            descriptions = _query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_results = [{"file_name": batch_to_process[i]["file_name"], "description": desc} for i, desc in enumerate(descriptions)]
        print(f"✅ {task_name}: Generated {len(final_results)} descriptions.")
        return final_results
    
    try:
        total = len(merged_image_features) if merged_image_features else 0
        idx1 = int(total * 0.2)
        print(f"[Task5a] total={total}, idx1={idx1}, batch=[:{idx1}], batch_size={idx1}")
        batch = merged_image_features[:idx1] if merged_image_features else []
        results = _vlm_processing_worker("Task 5a", args, batch, target_language)
        end_time = time.time()
        return {"dag_id": dag_id, "final_descriptions_a": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task5a_vlm_process failed: {str(e)}")
        return {"dag_id": dag_id, "final_descriptions_a": None,
                "start_time": start_time, "end_time": end_time}

def task5b_vlm_process(args, dag_id, merged_image_features, target_language, vllm_manager= None, backend= "huggingface"):
    """[Parallel Worker B] 使用VLM为20%-40%图片生成描述。"""
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
    
    def _vlm_processing_worker(task_name, args, batch_to_process, target_language):
        """一个通用的VLM工作函数，处理一批图像。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math
        from io import BytesIO
        from PIL import Image
        import torch
        from typing import List, Dict, Any, Tuple, Optional

        def _query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            processor, model = None, None
            try:
                print(f"  -> [{task_name}] Loading VLM model...")
                model_path = os.path.join(model_folder, model_name)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    device_map="cuda", offload_state_dict= False, trust_remote_code=True
                )
                
                all_responses = []
                for i in range(0, len(image_requests), batch_size):
                    mini_batch = image_requests[i:i + batch_size]
                    texts = [req['prompt'] for req in mini_batch]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in mini_batch]
                    
                    prompts = []
                    for text in texts:
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

                    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True)
                    
                    responses = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in responses])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  -> [{task_name}] VLM resources released.")

        print(f"⚙️  {task_name}: Starting VLM description generation...")
        if not batch_to_process:
            print(f"  -> {task_name}: Input batch is empty. Skipping.")
            return []

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            # 使用vllm_manager进行批量处理
            descriptions= query_vlm_batch_via_service(
                api_url= vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), model_alias= "qwen2.5-vl-32b",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8))
        else:
            descriptions = _query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_results = [{"file_name": batch_to_process[i]["file_name"], "description": desc} for i, desc in enumerate(descriptions)]
        print(f"✅ {task_name}: Generated {len(final_results)} descriptions.")
        return final_results
    try:
        total = len(merged_image_features) if merged_image_features else 0
        idx1 = int(total * 0.2)
        idx2 = int(total * 0.4)
        print(f"[Task5b] total={total}, idx1={idx1}, idx2={idx2}, batch=[{idx1}:{idx2}], batch_size={idx2-idx1}")
        batch = merged_image_features[idx1:idx2] if merged_image_features else []
        results = _vlm_processing_worker("Task 5b", args, batch, target_language)
        end_time = time.time()
        return {"dag_id": dag_id, "final_descriptions_b": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task5b_vlm_process failed: {str(e)}")
        return {"dag_id": dag_id, "final_descriptions_b": None,
                "start_time": start_time, "end_time": end_time}

def task5c_vlm_process(args, dag_id, merged_image_features, target_language, vllm_manager= None, backend= "huggingface"):
    """[Parallel Worker C] 使用VLM为40%-60%图片生成描述。"""
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
    
    def _vlm_processing_worker(task_name, args, batch_to_process, target_language):
        """一个通用的VLM工作函数，处理一批图像。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math
        from io import BytesIO
        from PIL import Image
        import torch
        from typing import List, Dict, Any, Tuple, Optional

        def _query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            processor, model = None, None
            try:
                print(f"  -> [{task_name}] Loading VLM model...")
                model_path = os.path.join(model_folder, model_name)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    device_map="cuda", offload_state_dict= False, trust_remote_code=True
                )
                
                all_responses = []
                for i in range(0, len(image_requests), batch_size):
                    mini_batch = image_requests[i:i + batch_size]
                    texts = [req['prompt'] for req in mini_batch]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in mini_batch]
                    
                    prompts = []
                    for text in texts:
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

                    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True)
                    
                    responses = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in responses])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  -> [{task_name}] VLM resources released.")

        print(f"⚙️  {task_name}: Starting VLM description generation...")
        if not batch_to_process:
            print(f"  -> {task_name}: Input batch is empty. Skipping.")
            return []

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            # 使用vllm_manager进行批量处理
            descriptions= query_vlm_batch_via_service(
                api_url= vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), model_alias= "qwen2.5-vl-32b",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8))
        else:
            descriptions = _query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_results = [{"file_name": batch_to_process[i]["file_name"], "description": desc} for i, desc in enumerate(descriptions)]
        print(f"✅ {task_name}: Generated {len(final_results)} descriptions.")
        return final_results
    try:
        total = len(merged_image_features) if merged_image_features else 0
        idx2 = int(total * 0.4)
        idx3 = int(total * 0.6)
        print(f"[Task5c] total={total}, idx2={idx2}, idx3={idx3}, batch=[{idx2}:{idx3}], batch_size={idx3-idx2}")
        batch = merged_image_features[idx2:idx3] if merged_image_features else []
        results = _vlm_processing_worker("Task 5c", args, batch, target_language)
        end_time = time.time()
        return {"dag_id": dag_id, "final_descriptions_c": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task5c_vlm_process failed: {str(e)}")
        return {"dag_id": dag_id, "final_descriptions_c": None,
                "start_time": start_time, "end_time": end_time}

def task5d_vlm_process(args, dag_id, merged_image_features, target_language, vllm_manager= None, backend= "huggingface"):
    """[Parallel Worker D] 使用VLM为60%-100%图片生成描述。"""
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
    def _vlm_processing_worker(task_name, args, batch_to_process, target_language):
        """一个通用的VLM工作函数，处理一批图像。它包含了所有依赖项以确保独立性。"""
        import os, gc, time, math
        from io import BytesIO
        from PIL import Image
        import torch
        from typing import List, Dict, Any, Tuple, Optional

        def _query_vlm_batch(model_folder: str, model_name: str, image_requests: List[Dict[str, Any]],
                            temperature: float, max_token: int, top_p: float, repetition_penalty: float, batch_size: int):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            processor, model = None, None
            try:
                print(f"  -> [{task_name}] Loading VLM model...")
                model_path = os.path.join(model_folder, model_name)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    device_map="cuda", offload_state_dict= False, trust_remote_code=True
                )
                
                all_responses = []
                for i in range(0, len(image_requests), batch_size):
                    mini_batch = image_requests[i:i + batch_size]
                    texts = [req['prompt'] for req in mini_batch]
                    images = [Image.open(BytesIO(req['content'])).convert("RGB") for req in mini_batch]
                    
                    prompts = []
                    for text in texts:
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

                    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=max_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True)
                    
                    responses = processor.batch_decode(outputs, skip_special_tokens=True)
                    all_responses.extend([resp.split("assistant\n")[-1].lstrip() for resp in responses])
                    del inputs, outputs
                return all_responses
            finally:
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  -> [{task_name}] VLM resources released.")

        print(f"⚙️  {task_name}: Starting VLM description generation...")
        if not batch_to_process:
            print(f"  -> {task_name}: Input batch is empty. Skipping.")
            return []

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            # 使用vllm_manager进行批量处理
            descriptions= query_vlm_batch_via_service(
                api_url= vllm_manager.get_next_endpoint("qwen2.5-vl-32b"), model_alias= "qwen2.5-vl-32b",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8))
        else:
            descriptions = _query_vlm_batch(
                model_folder=args.model_folder, model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests, temperature=args.temperature, max_token=getattr(args, "max_token", 1024),
                top_p=args.top_p, repetition_penalty=args.repetition_penalty, batch_size=getattr(args, "vlm_batch_size", 8)
            )
        
        final_results = [{"file_name": batch_to_process[i]["file_name"], "description": desc} for i, desc in enumerate(descriptions)]
        print(f"✅ {task_name}: Generated {len(final_results)} descriptions.")
        return final_results
    try:
        total = len(merged_image_features) if merged_image_features else 0
        idx3 = int(total * 0.6)
        print(f"[Task5d] total={total}, idx3={idx3}, batch=[{idx3}:], batch_size={total-idx3}")
        batch = merged_image_features[idx3:] if merged_image_features else []
        results = _vlm_processing_worker("Task 5d", args, batch, target_language)
        end_time = time.time()
        return {"dag_id": dag_id, "final_descriptions_d": results,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task5d_vlm_process failed: {str(e)}")
        return {"dag_id": dag_id, "final_descriptions_d": None,
                "start_time": start_time, "end_time": end_time}

def task5_merge_results(args, dag_id, final_descriptions_a, final_descriptions_b, final_descriptions_c, final_descriptions_d):
    """[Merge] 合并来自task5a, 5b, 5c, 5d的并行处理结果。"""
    import time
    start_time = time.time()
    try:
        print(f"✅ Task 5_merge: Merging parallel description results...")
        all_final_descriptions = (final_descriptions_a or []) + (final_descriptions_b or []) + \
                                 (final_descriptions_c or []) + (final_descriptions_d or [])
        print(f"✅ Task 5_merge: Merging complete. Total items: {len(all_final_descriptions)}.")
        end_time = time.time()
        return {"dag_id": dag_id, "final_descriptions": all_final_descriptions, "args": args,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task5_merge_results failed: {str(e)}")
        return {"dag_id": dag_id, "final_descriptions": None, "args": args,
                "start_time": start_time, "end_time": end_time}

def task6_output_final_answer(args, dag_id, final_descriptions):
    """Task 6: 格式化并输出最终答案。"""
    import time
    start_time = time.time()
    try:
        print("✅ Task 6: Formatting final output.")
        if final_descriptions is None:
            final_answer_text = "Workflow failed to produce final descriptions due to an upstream error."
        else:
            sorted_descriptions = sorted(final_descriptions, key=lambda x: x['file_name'])
            final_answer_text = '\n\n'.join([f"Image {d['file_name']}: {d['description']}" for d in sorted_descriptions])
        
        print(f"🏁 Final Answer for DAG {dag_id} generated.")
        end_time = time.time()
        return {"dag_id": dag_id, "final_answer": final_answer_text,
                "start_time": start_time, "end_time": end_time}
    except Exception as e:
        end_time = time.time()
        print(f"❌ task6_output_final_answer failed: {str(e)}")
        return {"dag_id": dag_id, "final_answer": f"Error during final output formatting: {e}",
                "start_time": start_time, "end_time": end_time}
