import os
import re
import gc
import json
import time
import pandas as pd
import ray
from typing import List, Dict, Any, Tuple
from io import BytesIO
import asyncio
import aiohttp
# 导入 AgentOS 调度器装饰器
from agentos.scheduler import cpu, gpu, io

# 导入图像处理和模型相关的库
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import easyocr
from scipy.stats import entropy

import base64

# ==============================================================================
#  1. 公共辅助函数 (Utilities)
# ==============================================================================

def estimate_tokens(text: str) -> int:
    """[一致] 估算中英混合文本的 token 数量。"""
    if not isinstance(text, str): return 0
    cjk_chars = sum(1 for char in text if '\u4E00' <= char <= '\u9FFF')
    non_cjk_text = re.sub(r'[\u4E00-\u9FFF]', ' ', text).replace("\n", " ")
    non_cjk_words_count = len(non_cjk_text.split())
    return cjk_chars + int(non_cjk_words_count * 1.3)

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
        base64_image = encode_image_bytes_base64(request['content'])
        
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

def encode_image_bytes_base64(img_bytes: bytes) -> str:
    """将图片字节流编码为base64字符串"""
    return base64.b64encode(img_bytes).decode('utf-8')

# query_vlm_batch (已修复版本)
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
import math

def query_vlm_batch(
    model_folder: str,
    model_name: str,
    image_requests: List[Dict[str, Any]],
    temperature: float = 0.6,
    max_token: int = 1024,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    batch_size: int = 8  # <--- 新增：定义微批次的大小，可以根据你的显存调整
) -> List[str]:
    """
    使用本地VLM模型批量处理图像描述任务 (最终修复版)。
    - 使用微批次(mini-batch)循环来防止OOM。
    - 修复padding_side为'left'。
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import os
    import torch
    import gc
    from rich.console import Console

    console = Console()
    model_path = os.path.join(model_folder, model_name)
    processor = None
    model = None

    try:
        console.print(f"🔯 [bold green]开始加载VLM模型... ({model_path})[/bold green]")
        # <--- 重要改动：添加 padding_side='left'
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype= torch.float16,
            low_cpu_mem_usage= True,
            device_map="cuda", offload_state_dict= False,
            trust_remote_code= True
        )
        console.print("✅ [bold green]VLM模型加载完毕。[/bold green]")
        
        all_final_responses = []
        num_requests = len(image_requests)
        num_batches = math.ceil(num_requests / batch_size)

        console.print(f"⚙️ [bold]总计 {num_requests} 张图片, 将分成 {num_batches} 个批次处理 (每批次 {batch_size} 张)。[/bold]")

        # <--- 核心改动：外层循环，实现微批次处理
        for i in range(0, num_requests, batch_size):
            batch_requests = image_requests[i:i + batch_size]
            console.print(f"--- 正在处理批次 {i // batch_size + 1} / {num_batches} ---")
            
            texts_for_processing = []
            images_for_processing = []

            for request in batch_requests:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": request['prompt']}
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts_for_processing.append(text)
                
                image = Image.open(BytesIO(request['content'])).convert("RGB")
                images_for_processing.append(image)

            inputs = processor(
                text=texts_for_processing,
                images=images_for_processing,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            responses = processor.batch_decode(output_ids, skip_special_tokens=True)
            cleaned_responses = [resp.split("assistant\n")[-1].lstrip() for resp in responses]
            all_final_responses.extend(cleaned_responses)
        
        console.print("✅ [bold green]所有批次处理完成。[/bold green]")
        return all_final_responses

    except Exception as e:
        console.print(f"[bold red]VLM批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        return [f"Error during VLM batch processing: {e}" for _ in image_requests]
    finally:
        if model is not None: del model
        if processor is not None: del processor
        gc.collect()
        torch.cuda.empty_cache()
        console.print("🗑️ [yellow]模型和GPU资源已释放。[/yellow]")

# --- [新增] 用于时间预测的特征提取函数 ---
def extract_vision_features(image_bytes: bytes, ocr_results: List[Dict] = None) -> Dict[str, float]:
    """为单个图像提取VLM预测所需的特征。"""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None: return {}
    
    h, w, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = {
        "image_height": h, "image_width": w, "image_area": h * w,
        "image_aspect_ratio": h / w if w > 0 else 1,
    }
    
    # 熵
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    prob_dist = hist / (hist.sum() + 1e-6)
    features["image_entropy"] = float(entropy(prob_dist))
    
    # 边缘密度
    edges = cv2.Canny(gray_img, 100, 200)
    features["edge_density"] = np.sum(edges > 0) / (h * w) if (h * w) > 0 else 0.0
    
    # 亮度
    features["avg_brightness"] = float(np.mean(gray_img))
    features["brightness_variance"] = float(np.var(gray_img))

    # OCR 相关特征
    text_area = 0
    if ocr_results:
        for res in ocr_results:
            box = np.array(res[0]).astype(np.int32)
            text_area += cv2.contourArea(box)
    features["text_area_ratio"] = text_area / (h * w) if (h * w) > 0 else 0.0
    features["text_block_count"] = len(ocr_results) if ocr_results else 0
            
    return features
# --- 结束新增 ---

# ==============================================================================
#  2. 重构后的 Task 函数 (已集成预测特征)
# ==============================================================================

@io(mem= 1024)
def task1_start_receive_task(context):
    """[一致] Task 1: 接收任务，确认核心参数。"""
    try:
        print("✅ Task 1: Starting... Verifying initial context.")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        question = ray.get(context.get.remote("question"))
        supplementary_files = ray.get(context.get.remote("supplementary_files"))
        
        target_language = "English"
        if "german" in question.lower() or "deutsch" in question.lower(): target_language = "German"
        elif "chinese" in question.lower() or "中文" in question.lower(): target_language = "Chinese"
        context.put.remote("target_language", target_language)

        print(f"  -> Received dag_id: {dag_id}, question: '{question[:50]}...', lang: {target_language}")
        return json.dumps({
            "dag_id": dag_id, "status": "success", "message": "Initial context verified.",
            "curr_task_feat": {"question_length": len(question), "num_files": len(supplementary_files)},
            "start_time": start_time, "end_time": time.time()
        })
    except Exception as e:
        dag_id = "unknown"; start_time = time.time()
        try: dag_id = ray.get(context.get.remote("dag_id"))
        except: pass
        print(f"❌ task1_start_receive_task failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 1 Error: {e}", "start_time": start_time, "end_time": time.time()})

@cpu(cpu_num= 1, mem= 1024)
def task2_read_and_enhance_images(context):
    """[修改] Task 2: 读取图像并增强，为后继任务准备简单特征。"""
    try:
        print("✅ Task 2: Reading and enhancing images...")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        supplementary_files = ray.get(context.get.remote("supplementary_files"))
        image_files = {k: v for k, v in supplementary_files.items() if k.lower().endswith(('.png', '.jpg', '.jpeg'))}
        enhanced_images = []

        for file_name, image_content in image_files.items():
            img_array = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            enhanced_img = img.copy()
            if (blur_score < 100):
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
            enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(enhanced_img_rgb)
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            enhanced_images.append({"file_name": file_name, "content": img_byte_arr.getvalue()})

        context.put.remote("enhanced_images", enhanced_images)
        print(f"✅ Task 2: Finished. Enhanced {len(enhanced_images)} images.")
        
        # 为两个并行的后继任务准备相同的简单特征
        return json.dumps({
            "dag_id": dag_id, "status": "success",
            "start_time": start_time, 
            "end_time": time.time()
        })
    except Exception as e:
        print(f"❌ task2_read_and_enhance_images failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 2 Error: {e}", "start_time": time.time(), "end_time": time.time()})

@gpu(gpu_mem= 8000)
def task3a_extract_blip_captions(context):
    """[一致] Task 3a (并行): 使用BLIP生成描述。"""
    try:
        print("✅ Task 3a: Generating BLIP captions...")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        enhanced_images = ray.get(context.get.remote("enhanced_images"))
        model_folder= ray.get(context.get.remote("model_folder"))
        model_path = os.path.join(model_folder, "blip-image-captioning-large")
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
        blip_captions = []
        for image_info in enhanced_images:
            raw_image = Image.open(BytesIO(image_info["content"])).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**inputs, max_new_tokens=75)
            caption = processor.decode(out[0], skip_special_tokens=True)
            blip_captions.append({"file_name": image_info["file_name"], "caption": caption})
        context.put.remote("blip_captions", blip_captions)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
        print("✅ Task 3a: BLIP captioning complete.")
        return json.dumps({
            "dag_id": dag_id, 
            "status": "success", 
            # "curr_task_feat": {"num_captions": len(blip_captions)}, 
            "start_time": start_time, 
            "end_time": time.time()
        })
    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ task3a_extract_blip_captions failed: {str(e)}")
        return json.dumps({
            "dag_id": dag_id, 
            "status": "failed", 
            "result": f"Task 3a Error: {e}", 
            "start_time": time.time(), 
            "end_time": time.time()
        })

@gpu(gpu_mem=4000)
def task3b_extract_ocr_text(context):
    """[修改] Task 3b (并行): 提取OCR文本并保存原始结果用于特征计算。"""
    try:
        print("✅ Task 3b: Extracting OCR text...")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        enhanced_images = ray.get(context.get.remote("enhanced_images"))
        target_language = ray.get(context.get.remote("target_language"))
        lang_map = {"chinese": "ch_sim", "german": "de"}
        reader_langs = [lang_map.get(target_language.lower(), 'en'), 'en']
        reader = easyocr.Reader(list(set(reader_langs)), gpu=True)
        ocr_results_list = []
        for image_info in enhanced_images:
            # reader.readtext 返回一个列表，每个元素是 (bbox, text, confidence)
            results = reader.readtext(image_info["content"])
            text = " ".join([res[1] for res in results if res[2] > 0.4])
            ocr_results_list.append({"file_name": image_info["file_name"], "text": text, "raw_results": results})
        context.put.remote("ocr_results_list", ocr_results_list)
        del reader; gc.collect(); torch.cuda.empty_cache()
        print("✅ Task 3b: OCR extraction complete.")
        return json.dumps({"dag_id": dag_id, "status": "success", "curr_task_feat": {"num_ocr_results": len(ocr_results_list)}, "start_time": start_time, "end_time": time.time()})
    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ task3b_extract_ocr_text failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 3b Error: {e}", "start_time": time.time(), "end_time": time.time()})

@cpu(cpu_num= 1, mem= 1024)
def task4_merge_image_features(context):
    """[MODIFIED FOR PREDICTION] Task 4 (合并点): 合并特征并为后继VLM任务准备预测特征。"""
    try:
        print("✅ Task 4: Merging features and preparing for prediction...")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        blip_captions = ray.get(context.get.remote("blip_captions"))
        ocr_results_list = ray.get(context.get.remote("ocr_results_list"))
        enhanced_images = ray.get(context.get.remote("enhanced_images"))

        captions_map = {item['file_name']: item['caption'] for item in blip_captions}
        ocr_map = {item['file_name']: {"text": item['text'], "raw": item['raw_results']} for item in ocr_results_list}
        
        merged_features_list = []
        all_vision_features = []
        prompt_lengths = []
        prompt_token_counts = []

        for image_info in enhanced_images:
            file_name = image_info['file_name']
            ocr_info = ocr_map.get(file_name, {"text": "", "raw": []})
            caption = captions_map.get(file_name, "")
            ocr_text = ocr_info["text"]

            # 构造VLM prompt（与后续任务一致）
            prompt = (
                "#Background#\n"
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
                "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
                "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
                f"#Question#\n"
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in English.\n"
                f"Base visual analysis suggests: '{caption}'.\n"
                + (f"Text found in the image: '{ocr_text}'.\n" if ocr_text else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in English."
            )
            prompt_lengths.append(len(prompt))
            prompt_token_counts.append(estimate_tokens(prompt))

            # 提取单个图像的详细视觉特征
            vision_features = extract_vision_features(image_info["content"], ocr_info["raw"])
            all_vision_features.append(vision_features)
            
            merged_features_list.append({
                "file_name": file_name, "content": image_info["content"],
                "caption": caption, "ocr_text": ocr_text
            })

        # 获取批处理大小
        vlm_batch_size = ray.get(context.get.remote("vlm_batch_size")) if hasattr(context, "get") else 1

        # --- 新增：动态计算批次索引 ---
        total = len(merged_features_list)
        idx1 = int(total * 0.2)
        idx2 = int(total * 0.4)
        idx3 = int(total * 0.6)
        batch_ranges = [
            (0, idx1),      # 0 ~ 20%
            (idx1, idx2),   # 20% ~ 40%
            (idx2, idx3),   # 40% ~ 60%
            (idx3, total)   # 60% ~ 100%
        ]
        batch_keys = [
            "task5a_vlm_process",
            "task5b_vlm_process",
            "task5c_vlm_process",
            "task5d_vlm_process"
        ]
        succ_task_feat = {}
        for idx, (start, end) in enumerate(batch_ranges):
            batch_feats = all_vision_features[start:end]
            batch_prompts = prompt_lengths[start:end]
            batch_tokens = prompt_token_counts[start:end]
            if batch_feats:
                df = pd.DataFrame(batch_feats)
                aggregated = df.mean().to_dict()
                aggregated["prompt_length"] = float(np.mean(batch_prompts)) if batch_prompts else 0.0
                aggregated["prompt_token_count"] = float(np.mean(batch_tokens)) if batch_tokens else 0.0
                aggregated["batch_size"] = vlm_batch_size
                aggregated["reason"] = 0
                succ_task_feat[batch_keys[idx]] = aggregated
            else:
                succ_task_feat[batch_keys[idx]] = {"batch_size": vlm_batch_size, "reason": 0}

        # 新增：将批次范围存入context，供后续任务使用
        context.put.remote("vlm_batch_ranges", batch_ranges)
        context.put.remote("aggregated_vision_features", succ_task_feat)
        context.put.remote("merged_image_features", merged_features_list)
        print(f"✅ Task 4: Feature merging complete. Aggregated features prepared for successor.")
        return json.dumps({
            "dag_id": dag_id, "status": "success",
            "succ_task_feat": {
                "task5a_vlm_process": succ_task_feat.get("task5a_vlm_process", {}),
                "task5b_vlm_process": succ_task_feat.get("task5b_vlm_process", {}),
                "task5c_vlm_process": succ_task_feat.get("task5c_vlm_process", {}),
                "task5d_vlm_process": succ_task_feat.get("task5d_vlm_process", {})
            },
            "start_time": start_time, "end_time": time.time()
        })
    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ task4_merge_image_features failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 4 Error: {e}", "start_time": time.time(), "end_time": time.time()})

@gpu(gpu_mem= 80000, model_name= "qwen2.5-vl-32b", backend="huggingface")
def task5a_vlm_process(context):
    """[Parallel Worker A] 使用VLM为前20%图片生成描述。"""
    try:
        backend= task5a_vlm_process._task_decorator['backend']
        print(f"✅ Task 5a: Starting description generation for items 0-19...")
        start_time = time.time()
        
        # --- 1. 从上下文中获取所需的所有参数 ---
        dag_id = ray.get(context.get.remote("dag_id"))
        model_folder = ray.get(context.get.remote("model_folder"))
        target_language = ray.get(context.get.remote("target_language"))
        temperature = ray.get(context.get.remote("temperature"))
        max_tokens = ray.get(context.get.remote("max_tokens"))
        top_p = ray.get(context.get.remote("top_p"))
        repetition_penalty = ray.get(context.get.remote("repetition_penalty"))
        vlm_batch_size= ray.get(context.get.remote("vlm_batch_size"))
        # --- 2. 获取完整数据并进行切片 ---
        merged_features = ray.get(context.get.remote("merged_image_features"))
        batch_ranges = ray.get(context.get.remote("vlm_batch_ranges"))
        start, end = batch_ranges[0]
        batch_to_process = merged_features[start:end]
        print(f"Task 5a: Processing items {start} to {end-1} (共{len(batch_to_process)}张, 占前20%)")

        # --- 3. 准备并调用VLM模型 ---
        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                "#Background#\n"
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
                "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
                "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
                f"#Question#\n"
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({
                "prompt": prompt,
                "content": image_info["content"]
            })
        if backend == "vllm":
            descriptions= query_vlm_batch_via_service(
                api_url= ray.get(context.get.remote("task5a_vlm_process_request_api_url")),
                model_alias="qwen2.5-vl-32b",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        else:
            descriptions = query_vlm_batch(
                model_folder=model_folder,
                model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        
        # --- 4. 整理结果并存入上下文 ---
        final_descriptions_a = []
        for i, desc in enumerate(descriptions):
            final_descriptions_a.append({
                "file_name": batch_to_process[i]["file_name"],
                "description": desc
            })
        
        # 将结果存入特定的键 "final_descriptions_a"
        context.put.remote("final_descriptions_a", final_descriptions_a)
        feature_5a = ray.get(context.get.remote("aggregated_vision_features")).get("task5a_vlm_process", {})
        if "text_length" in feature_5a:
            feature_5a["prompt_length"] = feature_5a.pop("text_length")
        if "token_count" in feature_5a:
            feature_5a["prompt_token_count"] = feature_5a.pop("token_count")
        print(f"✅ Task 5a: Generated {len(final_descriptions_a)} descriptions.")
        print(f"✅ Task 5a: Processing complete.")
        return json.dumps({"curr_task_feat": feature_5a,
                           "dag_id": dag_id, "status": "success", "task": "5a", "items_processed": len(final_descriptions_a), "start_time": start_time, "end_time": time.time()})

    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ 5a_generate_final_descriptions failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 5a Error: {e}", "start_time": time.time(), "end_time": time.time()})

@gpu(gpu_mem= 80000, model_name= "qwen2.5-vl-32b", backend="huggingface")
def task5b_vlm_process(context):
    """[Parallel Worker B] 使用VLM为第20%-40%图片生成描述。"""
    try:
        backend= task5b_vlm_process._task_decorator['backend']
        print(f"✅ Task 5b: Starting description generation for items 20-39...")
        start_time = time.time()
        
        dag_id = ray.get(context.get.remote("dag_id"))
        model_folder = ray.get(context.get.remote("model_folder"))
        target_language = ray.get(context.get.remote("target_language"))
        temperature = ray.get(context.get.remote("temperature"))
        max_tokens = ray.get(context.get.remote("max_tokens"))
        top_p = ray.get(context.get.remote("top_p"))
        vlm_batch_size= ray.get(context.get.remote("vlm_batch_size"))
        repetition_penalty = ray.get(context.get.remote("repetition_penalty"))
        
        merged_features = ray.get(context.get.remote("merged_image_features"))
        batch_ranges = ray.get(context.get.remote("vlm_batch_ranges"))
        start, end = batch_ranges[1]
        batch_to_process = merged_features[start:end]
        print(f"Task 5b: Processing items {start} to {end-1} (共{len(batch_to_process)}张, 占20%-40%)")

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                "#Background#\n"
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
                "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
                "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
                f"#Question#\n"
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            descriptions= query_vlm_batch_via_service(
                api_url= ray.get(context.get.remote("task5b_vlm_process_request_api_url")),
                model_alias="qwen2.5-vl-32b",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        else:
            descriptions = query_vlm_batch(
                model_folder=model_folder,
                model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
            
        final_descriptions_b = []
        for i, desc in enumerate(descriptions):
            final_descriptions_b.append({"file_name": batch_to_process[i]["file_name"], "description": desc})
        
        # --- 修改点: 上下文存储键 ---
        context.put.remote("final_descriptions_b", final_descriptions_b)
        feature_5b = ray.get(context.get.remote("aggregated_vision_features")).get("task5b_vlm_process", {})
        if "text_length" in feature_5b:
            feature_5b["prompt_length"] = feature_5b.pop("text_length")
        if "token_count" in feature_5b:
            feature_5b["prompt_token_count"] = feature_5b.pop("token_count")
        print(f"✅ Task 5b: Processing complete.")
        return json.dumps({
            "curr_task_feat": feature_5b,
            "dag_id": dag_id, "status": "success", "task": "5b", "items_processed": len(final_descriptions_b), "start_time": start_time, "end_time": time.time()})

    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ 5b_generate_final_descriptions failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 5b Error: {e}", "start_time": time.time(), "end_time": time.time()})

@gpu(gpu_mem= 80000, model_name= "qwen2.5-vl-32b", backend="huggingface")
def task5c_vlm_process(context):
    """[Parallel Worker C] 使用VLM为第40%-60%图片生成描述。"""
    try:
        backend= task5c_vlm_process._task_decorator['backend']
        print(f"✅ Task 5c: Starting description generation for items 40-59...")
        start_time = time.time()
        
        dag_id = ray.get(context.get.remote("dag_id"))
        model_folder = ray.get(context.get.remote("model_folder"))
        target_language = ray.get(context.get.remote("target_language"))
        temperature = ray.get(context.get.remote("temperature"))
        max_tokens = ray.get(context.get.remote("max_tokens"))
        top_p = ray.get(context.get.remote("top_p"))
        repetition_penalty = ray.get(context.get.remote("repetition_penalty"))
        vlm_batch_size= ray.get(context.get.remote("vlm_batch_size"))
        merged_features = ray.get(context.get.remote("merged_image_features"))
        batch_ranges = ray.get(context.get.remote("vlm_batch_ranges"))
        start, end = batch_ranges[2]
        batch_to_process = merged_features[start:end]
        print(f"Task 5c: Processing items {start} to {end-1} (共{len(batch_to_process)}张, 占40%-60%)")

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                "#Background#\n"
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
                "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
                "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
                f"#Question#\n"
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            descriptions= query_vlm_batch_via_service(
                api_url= ray.get(context.get.remote("task5c_vlm_process_request_api_url")),
                model_alias="qwen2.5-vl-32b",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        else:
            descriptions = query_vlm_batch(
                model_folder=model_folder,
                model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        
        final_descriptions_c = []
        for i, desc in enumerate(descriptions):
            final_descriptions_c.append({"file_name": batch_to_process[i]["file_name"], "description": desc})
        
        # --- 修改点: 上下文存储键 ---
        context.put.remote("final_descriptions_c", final_descriptions_c)
        feature_5c = ray.get(context.get.remote("aggregated_vision_features")).get("task5c_vlm_process", {})
        if "text_length" in feature_5c:
            feature_5c["prompt_length"] = feature_5c.pop("text_length")
        if "token_count" in feature_5c:
            feature_5c["prompt_token_count"] = feature_5c.pop("token_count")
        print(f"✅ Task 5c: Processing complete.")
        return json.dumps({
            "curr_task_feat": feature_5c,
            "dag_id": dag_id, "status": "success", "task": "5c", "items_processed": len(final_descriptions_c), "start_time": start_time, "end_time": time.time()})

    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ 5c_generate_final_descriptions failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 5c Error: {e}", "start_time": time.time(), "end_time": time.time()})

@gpu(gpu_mem= 80000, model_name= "qwen2.5-vl-32b", backend="huggingface")
def task5d_vlm_process(context):
    """[Parallel Worker D] 使用VLM为最后40%图片生成描述。"""
    try:
        backend= task5d_vlm_process._task_decorator['backend']
        print(f"✅ Task 5d: Starting description generation for items 60-99...")
        start_time = time.time()
        
        dag_id = ray.get(context.get.remote("dag_id"))
        model_folder = ray.get(context.get.remote("model_folder"))
        target_language = ray.get(context.get.remote("target_language"))
        temperature = ray.get(context.get.remote("temperature"))
        max_tokens = ray.get(context.get.remote("max_tokens"))
        top_p = ray.get(context.get.remote("top_p"))
        repetition_penalty = ray.get(context.get.remote("repetition_penalty"))
        vlm_batch_size= ray.get(context.get.remote("vlm_batch_size"))
        merged_features = ray.get(context.get.remote("merged_image_features"))
        batch_ranges = ray.get(context.get.remote("vlm_batch_ranges"))
        start, end = batch_ranges[3]
        batch_to_process = merged_features[start:end]
        print(f"Task 5d: Processing items {start} to {end-1} (共{len(batch_to_process)}张, 占60%-100%)")

        image_requests = []
        for image_info in batch_to_process:
            prompt = (
                "#Background#\n"
                "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
                "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
                "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
                "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
                "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
                f"#Question#\n"
                f"You are an expert image analyst. Provide a detailed, fluent description of the image in {target_language}.\n"
                f"Base visual analysis suggests: '{image_info['caption']}'.\n"
                + (f"Text found in the image: '{image_info['ocr_text']}'.\n" if image_info['ocr_text'] else "") +
                f"Combine all this information into a comprehensive description. Output only the final description in {target_language}."
            )
            image_requests.append({"prompt": prompt, "content": image_info["content"]})
        if backend == "vllm":
            descriptions= query_vlm_batch_via_service(
                api_url= ray.get(context.get.remote("task5d_vlm_process_request_api_url")),
                model_alias="qwen2.5-vl-32b",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        else:
            descriptions = query_vlm_batch(
                model_folder=model_folder,
                model_name="Qwen/Qwen2.5-VL-32B-Instruct",
                image_requests=image_requests,
                temperature=temperature,
                max_token=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size= vlm_batch_size
            )
        
        final_descriptions_d = []
        for i, desc in enumerate(descriptions):
            final_descriptions_d.append({"file_name": batch_to_process[i]["file_name"], "description": desc})
        
        # --- 修改点: 上下文存储键 ---
        context.put.remote("final_descriptions_d", final_descriptions_d)
        feature_5d = ray.get(context.get.remote("aggregated_vision_features")).get("task5d_vlm_process", {})
        if "text_length" in feature_5d:
            feature_5d["prompt_length"] = feature_5d.pop("text_length")
        if "token_count" in feature_5d:
            feature_5d["prompt_token_count"] = feature_5d.pop("token_count")
        print(f"✅ Task 5d: Processing complete.")
        return json.dumps({
            "curr_task_feat": feature_5d,
            "dag_id": dag_id, "status": "success", "task": "5d", "items_processed": len(final_descriptions_d), "start_time": start_time, "end_time": time.time()})

    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ 5d_generate_final_descriptions failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 5d Error: {e}", "start_time": time.time(), "end_time": time.time()})

@io(mem= 1024)
def task5_merge_results(context):
    """[Merge] 合并来自task5a, 5b, 5c, 5d的并行处理结果。"""
    try:
        print(f"✅ Task 5_merge: Merging parallel description results...")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))

        # 从上下文中获取所有并行任务的结果
        results_a = ray.get(context.get.remote("final_descriptions_a"))
        results_b = ray.get(context.get.remote("final_descriptions_b"))
        results_c = ray.get(context.get.remote("final_descriptions_c"))
        results_d = ray.get(context.get.remote("final_descriptions_d"))
        # 按顺序合并列表
        all_final_descriptions = results_a + results_b + results_c + results_d
        # 将最终合并的完整列表放入 "final_descriptions"，供task6使用
        context.put.remote("final_descriptions", all_final_descriptions)
        
        print(f"✅ Task 5_merge: Merging complete. Total items: {len(all_final_descriptions)}.")
        return json.dumps({"dag_id": dag_id, "status": "success", "task": "5_merge", "total_items": len(all_final_descriptions), "start_time": start_time, "end_time": time.time()})

    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ 5_merge failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 5_merge Error: {e}", "start_time": time.time(), "end_time": time.time()})

@cpu(cpu_num= 1, mem= 1024)
def task6_output_final_answer(context):
    """[一致] Task 6: 格式化并输出最终答案。"""
    try:
        print("✅ Task 6: Formatting final output.")
        start_time = time.time()
        dag_id = ray.get(context.get.remote("dag_id"))
        final_descriptions = ray.get(context.get.remote("final_descriptions"))
        sorted_descriptions = sorted(final_descriptions, key=lambda x: x['file_name'])
        final_answer_text = '\n\n'.join([f"Image {d['file_name']}: {d['description']}" for d in sorted_descriptions])
        print(f"🏁 Final Answer for DAG {dag_id}:\n{final_answer_text[:500]}...")
        return json.dumps({
            "dag_id": dag_id, "status": "success", "final_answer": final_answer_text,
            "curr_task_feat": {"final_answer_length": len(final_answer_text), "num_answers": len(final_descriptions)},
            "start_time": start_time, "end_time": time.time()
        })
    except Exception as e:
        dag_id = ray.get(context.get.remote("dag_id"))
        print(f"❌ task6_output_final_answer failed: {str(e)}")
        return json.dumps({"dag_id": dag_id, "status": "failed", "result": f"Task 6 Error: {e}", "start_time": time.time(), "end_time": time.time()})
