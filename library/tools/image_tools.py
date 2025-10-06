import os
import easyocr
from typing import List
from io import BytesIO
from maze.library.tools.definitions import tool
from maze.library.tools.definitions import tool, TYPE_FILEPATH

@tool(
    name="ocr_image_batch",
    description="(批处理版) 对一批图片（二进制格式）执行OCR文字识别。模型在函数内部初始化，以确保在远程GPU节点上加载。",
    task_type='gpu',      # 明确这是一个GPU密集型任务
    gpu_mem=4096,         # 继承并明确资源需求
    model_name='easyocr', # 标注任务依赖的模型/库
    input_parameters={
        "type": "object",
        "properties": {
            "image_bytes_list": {
                "type": "array",
                "description": "一个包含多张页面图片二进制内容的列表。",
                "items": {"type": "string", "format": "binary"}
            },
            "start_page_number": {
                "type": "integer",
                "description": "该批次的起始页码（用于日志记录）。"
            }
        },
        "required": ["image_bytes_list", "start_page_number"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "ocr_texts": {
                "type": "array",
                "description": "一个列表，包含每张图片识别出的文本字符串。",
                "items": {"type": "string"}
            }
        }
    }
)
def ocr_image_batch(image_bytes_list: List[bytes], start_page_number: int, model_cache_dir: str= None) -> List[str]:
    """
    (批处理版) 对一批图片（二进制格式）执行OCR文字识别。
    模型在函数内部初始化，以确保在远程GPU节点上加载。
    
    :param image_bytes_list: 一个包含多张页面图片二进制内容的列表。
    :param start_page_number: 该批次的起始页码（用于日志记录）。
    :return: 一个列表，包含每张图片识别出的文本字符串。
    """

    # --- 函数核心逻辑保持不变 ---
    print(f"  - Batch starting at page {start_page_number}: Initializing EasyOCR reader on worker node...")
    model_cache_dir= os.path.join(model_cache_dir, "easyocr")
    ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True, model_storage_directory= model_cache_dir)
    
    print(f"  - Batch starting at page {start_page_number}: Processing {len(image_bytes_list)} images...")
    
    batch_results = []
    for i, image_bytes in enumerate(image_bytes_list):
        current_page = start_page_number + i
        try:
            ocr_results = ocr_reader.readtext(image_bytes)
            ocr_text = "\n".join([result[1] for result in ocr_results])
            batch_results.append(f"--- Page {current_page} (OCR) ---\n{ocr_text}\n\n")
        except Exception as e:
            print(f"  - Page {current_page}: OCR processing failed: {e}")
            batch_results.append(f"--- Page {current_page} (OCR FAILED) ---\n\n")
            
    print(f"  - Batch starting at page {start_page_number}: OCR finished.")
    return batch_results