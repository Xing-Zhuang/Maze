import os
import cv2
import ray
import oss2
import json
import time
import torch
import gc
import base64
import easyocr
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Dict
from agentos.scheduler import gpu, io, cpu

def calculate_image_entropy(image):
    """计算图像熵作为复杂度指标"""
    hist= np.histogram(image, bins= 256, range= (0, 256))[0]
    hist= hist/ hist.sum()
    entropy= -np.sum(hist* np.log2(hist+ 1e-10))
    return entropy

import re
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

def encode_image_bytes_base64(image_bytes: bytes)-> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def query_vlm_vllm(api_url: str, model_alias: str, prompt: str, img_bytes: bytes, temperature:float= 0.6, max_token:int= 1024, top_p:float= 0.9, repetition_penalty:float= 1.1) -> tuple[dict, str]:
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
        features = {"text_length": len(prompt), "token_count": estimate_tokens(prompt)}
        return features, content
    except requests.exceptions.RequestException as e:
        error_msg = f"vLLM VLM request failed: {str(e)}"
        print(f"[bold red]{error_msg}")
        features = {"text_length": len(prompt), "token_count": estimate_tokens(prompt)}
        return features, f"[bold red]{error_msg}"

def query_vlm_online(api_url: str, api_key: str, model_name: str, prompt: str, img_bytes: bytes, tokenizer_path: str, temperature: float, max_tokens: int, timeout: int= 3600)-> str:
    """Call SiliconFlow API to get LLM response"""
    import requests
    from rich.console import Console
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
    # 将messages转换成字符串格式
    conversation= ""
    for message in messages:
        conversation+= f"{message['role']}: {message['content'][0]['text']}"
    
    console= Console()
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    payload= {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "text"}
    }
    try:
        response= requests.post(
            api_url,
            json= payload,
            headers= headers,
            timeout= timeout
        )
        response.raise_for_status()
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, response.json()['choices'][0]['message']['content'].lstrip()
    except Exception as e:
        console.print(f"[bold red]API request failed: {str(e)}")
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, f"[bold red]API request failed: {str(e)}"

def query_vlm(model_folder: str, model_name: str, prompt: str, img_bytes: bytes, temperature:float= 0.6, max_token:int= 1024, top_p:float= 0.9, repetition_penalty:float= 1.1)-> str:
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
    # 将messages转换成字符串格式
    conversation= ""
    for message in messages:
        conversation+= f"{message['role']}: {message['content'][1]['text']}"
    console= Console()

    try:
        model_path= os.path.join(model_folder, model_name)
        processor = AutoProcessor.from_pretrained(model_path, device_map= "cuda")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda", offload_state_dict= False,
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
            max_new_tokens= max_token,
            temperature= temperature,
            top_p= top_p,
            repetition_penalty= repetition_penalty)
        response= processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, response.lstrip()
    except Exception as e:
        console.print(f"[bold red]API request failed: {str(e)}")
        return {"text_length": len(conversation), "token_count": estimate_tokens(conversation)}, f"[bold red]API request failed: {str(e)}"
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if os.path.exists(temp_image_file):
            os.remove(temp_image_file)
            print(f"Temporary image file deleted: {temp_image_file}")

@cpu(cpu_num= 1, mem= 1024)
def task1_obtain_content(context):
    """
    Reads read.jsonl, finds the task based on task_id, reads the corresponding image file,
    and stores task info and file content in context.
    """
    import os
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote('dag_id'))
        question = ray.get(context.get.remote("question"))
        supplementary_files = ray.get(context.get.remote("supplementary_files"))
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
        context.put.remote("file_content", content)

        # --- MODIFICATION START ---
        # 流程调整：我们将分步计算图像和文本特征，然后合并它们。

        print("开始提取图像特征...")
        image = Image.open(BytesIO(content)).convert('RGB')
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = image_np

        edges = cv2.Canny(gray_img, 100, 200)
        edge_density = np.sum(edges > 0) / (image.height * image.width)

        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_area = 0
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
        for cnt in valid_contours:
            text_area += cv2.contourArea(cnt)

        image_features = {
            "image_height": image.height,
            "image_width": image.width,
            "image_area": image.height * image.width,
            "image_aspect_ratio": image.width / image.height if image.height > 0 else 0,
            "image_entropy": calculate_image_entropy(image_np),
            "edge_density": edge_density,
            "text_area_ratio": text_area / (image.height * image.width) if image.height * image.width > 0 else 0,
            "text_block_count": len(valid_contours),
            "avg_brightness": np.mean(gray_img),
            "brightness_variance": np.var(gray_img)
        }
        print(f"图像特征提取完成: {image_features}")
        # 步骤 2: 计算文本特征 (恢复之前的逻辑)
        print("开始提取文本特征...")
        prompt = (
            "#Background#\n"
            "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
            "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
            "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
            f"#Question#\n{question}\n"
        )
        text_features = {
            "prompt_length": len(prompt),
            "prompt_token_count": estimate_tokens(prompt)
        }
        print(f"文本特征提取完成: {text_features}")

        # 步骤 3: 合并图像与文本特征
        succ_task_feat = image_features.copy()  # 先复制图像特征
        succ_task_feat.update(text_features)   # 再并入文本特征
        context.put.remote("task2_vlm_process_feature", succ_task_feat)
        print(f"合并后的最终特征: {succ_task_feat}")

        return json.dumps({ # 为了保存到本地
            "dag_id": dag_id,
            "succ_task_feat": {
                "task2_vlm_process": succ_task_feat
            },
            "curr_task_feat": None,
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"task1_obtain_content 发生错误: {str(e)}")
        raise e

@gpu(gpu_mem= 70000, model_name= "qwen2.5-vl-32b", backend="huggingface")
def task2_vlm_process(context):
    """
    Processes the file content using LLM based on the question.
    """
    try:
        backend= task2_vlm_process._task_decorator["backend"]
        start_time= time.time()  # ADD: 记录开始时间
        dag_id= ray.get(context.get.remote('dag_id'))        
        question = ray.get(context.get.remote("question"))
        image_bytes= ray.get(context.get.remote("file_content"))
        vlm_process_feature= ray.get(context.get.remote('task2_vlm_process_feature'))
        # === 在此处集成下采样逻辑 ===
        print(f"原始图片大小: {len(image_bytes) / 1024:.1f} KB")
        # 调用下采样函数，它会根据需要返回原始或缩小后的图片字节
        image_bytes= downsample_image_if_needed(image_bytes, max_dimension= 1024, max_size_mb= 1.5)

        use_online_model= ray.get(context.get.remote("use_online_model"))
        model_folder= ray.get(context.get.remote("model_folder"))
        tokenizer_path= os.path.join(model_folder, "Qwen/Qwen3-32B")
        temperature= ray.get(context.get.remote("temperature"))
        max_tokens= ray.get(context.get.remote("max_tokens"))
        print(f"😀 max_tokens setting: {max_tokens}")
        top_p= ray.get(context.get.remote("top_p"))
        repetition_penalty= ray.get(context.get.remote("repetition_penalty"))
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
        )
        answer= None
        if use_online_model:
            _, answer= query_vlm_online(
                api_url= ray.get(context.get.remote("api_url")),
                api_key= ray.get(context.get.remote("api_key")),
                model_name= "Qwen/Qwen2.5-VL-32B-Instruct", 
                prompt= prompt, 
                img_bytes= image_bytes,
                tokenizer_path= tokenizer_path,
                temperature= temperature,
                max_tokens= max_tokens)
        elif backend == "vllm":
            _, answer= query_vlm_vllm(
                api_url= ray.get(context.get.remote("task2_vlm_process_request_api_url")),
                model_alias= "qwen2.5-vl-32b",
                prompt= prompt,
                img_bytes= image_bytes,
                temperature= temperature,
                max_token= max_tokens,
                top_p= top_p,
                repetition_penalty= repetition_penalty)
        else:
            print(f"🔯 本地VLM模型准备加载ing...")
            _, answer= query_vlm(model_folder= model_folder, model_name= "Qwen/Qwen2.5-VL-32B-Instruct", prompt= prompt, img_bytes= image_bytes, temperature= temperature, max_token= max_tokens, top_p= top_p, repetition_penalty= repetition_penalty)

        context.put.remote("vlm_answer", answer)
        return json.dumps({
            "task_id": dag_id,
            "curr_task_feat": vlm_process_feature,
            "start_time": start_time,
            "end_time": time.time()
        })
    except Exception as e:
        print(f"task2_vlm_process 发生错误: {str(e)}")
        raise e
    
@io(mem= 1024)
def task3_output_final_answer(context):
    """
    Outputs the final answer from the LLM.
    """
    try:
        start_time= time.time()
        dag_id= ray.get(context.get.remote('dag_id'))    
        answer = ray.get(context.get.remote("vlm_answer"))
        # 使用通用输出格式
        return json.dumps({
            "dag_id": dag_id,
            "final_answer": answer,
            "start_time": start_time,
            "end_time": time.time()           
        })    
    except Exception as e:
        print(f"task3_output_final_answer 发生错误: {str(e)}")
        raise e