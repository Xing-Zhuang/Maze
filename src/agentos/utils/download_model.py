import os
import easyocr
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用高速下载
os.environ["HF_HUB_OFFLINE"] = "0"  # 确保不使用离线模式
from pathlib import Path
from typing import List
import requests # 新增：用于下载测试图片
from PIL import Image # 新增：用于处理图片
from io import BytesIO # 新增：用于处理图片数据流
import torch
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import transformers.utils.hub as hub
hub.HF_ENDPOINT = "https://hf-mirror.com"
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration
)

# 设置可见的gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
def test_sentiment_analysis_inference(model_path: Path) -> bool:
    """测试多语言情感分析模型的推理功能。"""
    try:
        # 加载模型
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=model_path, 
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 测试文本 (包含多种语言)
        test_texts = [
            "This is absolutely wonderful!",  # 英语
            "Je suis très content avec ce produit.",  # 法语
            "Ich bin sehr enttäuscht von dieser Erfahrung.",  # 德语
            "Estoy bastante satisfecho con el servicio.",  # 西班牙语
            "这个产品真的很糟糕",  # 中文
        ]
        
        # 进行情感分析
        results = sentiment_analyzer(test_texts)
        
        # 打印结果
        for text, result in zip(test_texts, results):
            print(f"📝 文本: {text}")
            print(f"  情感标签: {result['label']}, 置信度: {result['score']:.4f}")
            print("-" * 50)
        
        # 清理内存
        del sentiment_analyzer
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ 情感分析模型推理失败: {e}")
        return False

def test_blip_inference(model_path: Path) -> bool:
    """测试 BLIP 模型的图像描述功能。"""
    try:
        # 加载模型和处理器
        processor = BlipProcessor.from_pretrained(model_path)
        # 使用 device_map="auto" 以便自动分配设备
        model = BlipForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 准备一张测试图片 (从网络加载)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url)
        raw_image = Image.open(BytesIO(response.content)).convert('RGB')

        # 准备输入
        # 将输入移动到模型所在的设备
        inputs = processor(raw_image, return_tensors="pt").to(model.device, torch.float16)

        # 生成描述
        out = model.generate(**inputs, max_new_tokens=75)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"🖼️ BLIP 推理结果: {caption}")
        # 清理内存
        del model, processor, inputs, out
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ BLIP 推理失败: {e}")
        return False

def test_deepseek_inference(model_path: Path) -> bool:
    """测试DeepSeek模型的文本生成功能。"""
    try:
        # 使用 AutoTokenizer 和 AutoModelForCausalLM 加载模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map= "auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # bfloat16 对新一代GPU更友好
            device_map="auto",
        )
        # 准备输入
        messages = [
            {"role": "user", "content": "请写一个关于月亮和星星的童话故事。"}
        ]
        # 使用 apply_chat_template 是处理对话模型的最佳实践
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # 生成文本
        outputs = model.generate(
            **inputs,
            max_new_tokens= 16384,
            do_sample= True,
            top_p= 0.9,
            temperature= 0.6,
            repetition_penalty=1.1
        )
        # 解码并打印结果
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 DeepSeek 推理结果: {response}")
        return True
    except Exception as e:
        print(f"❌ DeepSeek 推理失败: {e}")
        return False

def test_whisper_inference(model_path):
    try:
        processor = WhisperProcessor.from_pretrained(model_path, device_map= "auto")
        model = WhisperForConditionalGeneration.from_pretrained(model_path, device_map= "auto")
        model.generation_config.forced_decoder_ids = None

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = ds[0]["audio"]

        input_features = processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt"
        ).input_features
        input_features = input_features.to("cuda")
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print("🔊 Whisper 识别结果：", transcription)
        return True
    except Exception as e:
        print("❌ Whisper 推理失败：", e)
        return False

def test_qwen3_inference(model_path, cache_dir= "/mnt/7T/xz/"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map= "auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir= cache_dir,
            device_map= "auto"
        )

        messages = [
            {"role": "user", "content": "你好，请用一句话介绍一下你自己。"},
        ]
        conversation = ""
        for m in messages:
            conversation += f"{m['role']}: {m['content']}\n"

        input_ids = tokenizer.encode(conversation + tokenizer.eos_token, return_tensors="pt").to("cuda")
        output = model.generate(input_ids, pad_token_id= tokenizer.eos_token_id)
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("🧠 Qwen3 推理结果：", response)

        del model
        del tokenizer
        return True
    except Exception as e:
        print("❌ Qwen3 推理失败：", e)
        return False

def test_qwen_vl_inference(model_path):
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import Qwen2_5_VLForConditionalGeneration  # 确保路径正确

        processor = AutoProcessor.from_pretrained(model_path, device_map= "auto")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "请描述这张图片"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        output_ids = model.generate(**inputs, max_new_tokens=64)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("🖼️ Qwen-VL 图文推理结果：", response)
        return True
    except Exception as e:
        print("❌ Qwen-VL 推理失败：", e)
        return False

def test_t5_inference(model_path: Path) -> bool:
    """测试 T5 模型的文本摘要功能。"""
    try:
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 准备测试文本
        text = """The Tower of London is a historic castle on the north bank of 
                the River Thames in central London. It was founded towards the 
                end of 1066 as part of the Norman Conquest. The Tower has served 
                variously as an armory, a treasury, a menagerie, the home of the 
                Royal Mint, a public records office, and the home of the Crown 
                Jewels of England."""
        
        # 生成输入
        inputs = tokenizer(
            "summarize: " + text,  # T5 需要任务前缀
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(model.device)
        
        # 生成摘要
        outputs = model.generate(
            **inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
        
        # 解码结果
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 T5 摘要结果: {summary}")
        
        # 清理内存
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ T5 推理失败: {e}")
        return False

def download_model(model_name: str, local_model_folder: str= "./"):
    if model_name == "Qwen/Qwen3-32B":
        local_path = os.path.join(local_model_folder, "Qwen/Qwen3-32B")
        if Path(local_path).exists() and test_qwen3_inference(local_path):
            print(f"✅ 已存在可用 Qwen3 模型：{local_path}")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map= "auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map= "auto"
        )
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        print(f"✅ Qwen3 模型已保存到：{local_path}")
        test_qwen3_inference(local_path)
    
    elif model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
        local_path = os.path.join(local_model_folder, "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        if Path(local_path).exists() and test_deepseek_inference(local_path):
            print(f"✅ 已存在可用 DeepSeek r1 模型：{local_path}")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map= "auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map= "auto"
        )
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        print(f"✅ DeepSeek r1 模型已保存到：{local_path}")
        test_deepseek_inference(local_path)

    elif model_name == "Qwen/Qwen2.5-VL-32B-Instruct":
        local_path = os.path.join(local_model_folder, "Qwen/Qwen2.5-VL-32B-Instruct")
        if Path(local_path).exists() and test_qwen_vl_inference(local_path):
            print(f"✅ 已存在可用 Qwen-VL 模型：{local_path}")
            return
        processor = AutoProcessor.from_pretrained(model_name, device_map= "auto")
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map= "auto")
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
        processor.save_pretrained(local_path)
        print(f"✅ Qwen-VL 模型已保存到：{local_path}")
        test_qwen_vl_inference(local_path)

    elif model_name == "openai/whisper-medium":
        local_path = os.path.join(local_model_folder, "whisper_models/whisper-medium")        
        if Path(local_path).exists() and test_whisper_inference(local_path):
            print(f"✅ 已存在可用 Whisper 模型：{local_path}")
            return

        processor = WhisperProcessor.from_pretrained(model_name, device_map= "auto")
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map= "auto")
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
        print(f"✅ Whisper 模型已保存到：{local_path}")
        test_whisper_inference(local_path)

    elif model_name == "Salesforce/blip-image-captioning-large":
        # 定义一个更简洁的本地文件夹名
        local_path = os.path.join(local_model_folder, "blip-image-captioning-large")
        if Path(local_path).exists() and test_blip_inference(local_path):
            print(f"✅ 已存在可用 BLIP 模型：{local_path}")
            return

        print(f"📥 正在下载 BLIP 模型: {model_name}...")
        processor = BlipProcessor.from_pretrained(
            model_name,
            force_download=True,  # 强制重新下载
            resume_download=False,  # 不恢复下载
            local_files_only=False,  # 不使用本地缓存
            use_fast=False,  # 明确指定不使用 fast tokenizer
        )
        model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" # 自动选择设备
        )
        
        # 创建目录并保存
        Path(local_path).mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        print(f"✅ BLIP 模型已保存到：{local_path}")
        # 下载后立即测试
        test_blip_inference(local_path)

    elif model_name== "t5-base":
        model_name = "t5-base"
        languages= ["en"]
        local_path = os.path.join(local_model_folder, "t5-base")
        
        # 检查模型是否已存在且可用
        if Path(local_path).exists() and test_t5_inference(local_path):
            print(f"✅ 已存在可用 T5-base 模型: {local_path}")
            return
        
        print(f"📥 正在下载 T5-base 模型...")
        
        # 下载模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 保存到本地
        Path(local_path).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        print(f"✅ T5-base 模型已保存到: {local_path}")
        # 测试模型
        test_t5_inference(local_path)
    elif model_name== "easyocr":
        model_name = "easyocr"
        languages= ["en"]
        local_path = os.path.join(local_model_folder, "easyocr")
        # 检查模型是否已存在且可用
        if Path(local_path).exists() and test_easyocr_inference(local_path):
            print(f"✅ 已存在可用 easyocr 模型: {local_path}")
            return
        
        print(f"📥 正在下载 easyocr 模型...")

        # 测试模型是否已存在且可用
        if Path(local_path).exists():
            print("🔄 检测到已有模型文件，正在验证...")
            if test_easyocr_inference(local_path):
                print(f"✅ 现有模型验证通过，无需重新下载")
                return 

        # 下载模型
        print("⏳ 正在下载模型文件（首次运行可能需要较长时间）...")
        start_time = time.time()
        
        # 此调用会自动下载模型
        reader = easyocr.Reader(
            lang_list= languages,
            gpu=True,
            download_enabled=True,
            model_storage_directory=local_path,
            detector=True,
            recognizer=True
        )
        
        download_time = time.time() - start_time
        print(f"✅ 模型下载完成，耗时 {download_time:.2f}s")
        
        # 立即测试
        print("\n🧪 开始模型测试...")
        test_result = test_easyocr_inference(local_path, languages)
        # 清理资源
        del reader
        torch.cuda.empty_cache()        
        if not test_result:
            raise RuntimeError("模型测试失败")
    elif model_name== "nlptown/bert-base-multilingual-uncased-sentiment":
        local_path = os.path.join(local_model_folder, model_name)
        # 检查模型是否已存在且可用
        if Path(local_path).exists() and test_sentiment_analysis_inference(local_path):
            print(f"✅ 已存在可用 {model_name} 模型: {local_path}")
            return
        
        print(f"📥 正在下载 easyocr 模型...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=model_name,
            device="cuda"
        )
        
        # 保存到本地
        Path(local_path).mkdir(parents=True, exist_ok=True)
        sentiment_analyzer.model.save_pretrained(local_path)
        sentiment_analyzer.tokenizer.save_pretrained(local_path)
        
        print(f"✅ 多语言情感分析模型已保存到: {local_path}")

        # 测试模型是否已存在且可用
        if Path(local_path).exists():
            print("🔄 检测到已有模型文件，正在验证...")
            if test_sentiment_analysis_inference(local_path):
                print(f"✅ 现有模型验证通过，无需重新下载")
                return 
    else:
        print(f"❌ 未知模型名称：{model_name}")


if __name__ == "__main__":
    # 可根据需要修改这里
    # download_model("Qwen/Qwen3-32B")
    # download_model("Qwen/Qwen2.5-VL-32B-Instruct")
    # download_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    # download_model("openai/whisper-medium")
    # download_model("Salesforce/blip-image-captioning-large")
    # download_model("t5-base")
    # download_model("easyocr")
    download_model("nlptown/bert-base-multilingual-uncased-sentiment")