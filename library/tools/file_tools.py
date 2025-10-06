import os
import io
import re
import fitz
import easyocr
import pdfplumber
from io import BytesIO
from typing import Tuple
import concurrent.futures
from typing import List, Dict, Optional, Any
from maze.library.tools.llm_tools import _query_llm_online
from maze.library.tools.definitions import tool, TYPE_FILEPATH
from maze.library.tools.image_tools import ocr_image_batch

@tool(
    name="load_pdf",
    description="将本地的PDF文件加载到内存中，作为二进制内容返回。",
    task_type='io',
    mem=512,
    input_parameters={
        "properties": {
            "pdf_path": {
                "type": TYPE_FILEPATH,
                "description": "需要加载的PDF文件在本地的完整路径。"
            }
        },
        "required": ["pdf_path"]
    },
    output_parameters={
        "properties": {
            "pdf_content": {
                "type": "string",
                "description": "PDF文件的二进制内容。"
            }
        }
    }
)
def load_pdf(pdf_path: str) -> bytes:
    """
    从指定路径加载PDF文件并返回其二进制内容。
    """
    print(f"Executing load_pdf(pdf_path='{pdf_path}')")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"文件未找到: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        print(f"  - Successfully loaded {len(content)} bytes from file.")
        return content
    except Exception as e:
        print(f"  - Error reading file: {e}")
        raise

@tool(
    name= "extract_text_and_tables_from_native_pdf",
    description= "从原生（非扫描）PDF文件的二进制内容中快速提取所有文本和结构化表格，并将它们格式化为单个字符串返回。此工具不适用于扫描件或图片型PDF。",
    task_type= 'cpu',
    mem= 1024,
    input_parameters= {
        "type": "object",
        "properties": {
            "pdf_content": {
                "type": "string",
                "format": "binary",
                "description": "需要处理的PDF文件的二进制内容。"
            }
        },
        "required": ["pdf_content"]
    },
    output_parameters= {
        "type": "object",
        "properties": {
            "extracted_text": {
                "type": "string",
                "description": "从PDF中提取并格式化后的所有文本和表格内容。"
            }
        }
    }
)
def extract_text_and_tables_from_native_pdf(pdf_content: bytes) -> str:
    """
    使用pdfplumber从PDF二进制内容中提取文本和表格。
    此方法直接解析PDF文本对象，速度快，但对扫描件无效。
    """
    print(f"Executing extract_text_and_tables_from_native_pdf(...)")
    if not pdf_content:
        raise ValueError("输入的PDF内容不能为空。")

    try:
        pdf_pages = []
        # 使用 io.BytesIO 直接在内存中打开二进制内容，避免读写磁盘
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            print(f"  - Processing {len(pdf.pages)} pages from PDF content in memory.")
            for i, page in enumerate(pdf.pages):
                page_content = []
                
                # 1. 提取页面文本
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_content.append(f"--- Page {i+1} Text ---\n{page_text}")
                
                # 2. 提取页面中的表格
                tables = page.extract_tables()
                if tables:
                    table_texts = []
                    for j, table in enumerate(tables):
                        table_text = f"\n--- Table {j+1} on Page {i+1} ---\n"
                        for row in table:
                            # 清理None值，并确保所有单元格都是字符串
                            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                            table_text += " | ".join(cleaned_row) + "\n"
                        table_texts.append(table_text)
                    
                    if table_texts:
                        page_content.append("".join(table_texts))

                # 将当前页的文本和表格内容合并
                if page_content:
                    pdf_pages.append("\n".join(page_content))
        
        processed_content = "\n\n".join(pdf_pages)
        print(f"  - Successfully extracted and formatted content.")
        return processed_content

    except Exception as e:
        print(f"  - Error processing PDF with pdfplumber: {e}")
        # 抛出异常，以便上层调用者可以捕获和处理
        raise

@tool(
    name="extract_text_from_pdf_range",
    description="通过将PDF页面转换为图片并使用EasyOCR，从指定页码范围提取文本。",
    task_type='gpu',
    mem=2048,
    gpu_mem=4096,
    input_parameters={
        "type": "object",
        "properties": {
            "pdf_content": {
                "type": "string",
                "description": "PDF文件的二进制内容。"
            },
            "page_range": {
                "type": "array",
                "description": "一个包含起始和结束物理页码的元组 (例如, [3, 5])，页码从1开始。"
            }
        },
        "required": ["pdf_content", "page_range"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "extracted_text": {
                "type": "string",
                "description": "从指定页面范围内OCR识别并合并后的文本。"
            }
        }
    }
)
def extract_text_from_pdf_range(pdf_content: bytes, page_range: Tuple[int, int], model_cache_dir: str = None) -> str:
    """
    对PDF内容的指定页面范围进行OCR文本提取。
    """
    start_page, end_page = page_range
    print(f"Executing extract_text_from_pdf_range(page_range=[{start_page}, {end_page}]) using EasyOCR.")

    if not pdf_content:
        print("  - Error: Received empty PDF content.")
        return ""
    
    try:
        # 1. 初始化 EasyOCR Reader
        print("  - Initializing EasyOCR reader (this may take a moment)...")
        model_cache_dir= os.path.join(model_cache_dir, "easyocr")
        ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True, model_storage_directory= model_cache_dir)
        
        # 2. 从内存中打开PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        # 3. 页码验证
        num_pages = len(doc)
        if start_page < 1 or end_page > num_pages or start_page > end_page:
            raise ValueError(
                f"无效的页码范围: [{start_page}, {end_page}]。文件总页数: {num_pages}。"
            )
        
        all_text_parts = []
        # 4. 遍历页面，执行OCR
        # PyMuPDF页码是0-indexed, 所以需要转换
        for page_num in range(start_page - 1, end_page):
            current_page_for_log = page_num + 1
            print(f"  - Processing Page {current_page_for_log} with OCR...")
            page = doc.load_page(page_num)
            
            # 将页面渲染为高质量图片
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            # 执行OCR
            ocr_results = ocr_reader.readtext(img_bytes)
            ocr_text = "\n".join([result[1] for result in ocr_results])
            all_text_parts.append(f"--- Page {current_page_for_log} (OCR) ---\n{ocr_text}\n\n")

        # 5. 清理资源
        doc.close()
        
        result = "".join(all_text_parts)
        print(f"  - Successfully processed all pages with OCR.")
        return result
        
    except Exception as e:
        print(f"  - Error processing PDF with OCR: {e}")
        raise


@tool(
    name="calculate_page_offset",
    description="通过比较目录中起始页码最早的章节和用户输入的第一章物理页码，来计算页码偏移量。",
    task_type='cpu',
    mem=256,
    input_parameters={
        "type": "object",
        "properties": {
            "logical_toc_with_ranges": {
                "type": "object",
                "description": "由LLM解析出的、包含起始和结束页码的结构化目录。"
            },
            "physical_page_of_chapter_1": {
                "type": "integer",
                "description": "第一章内容实际开始的物理页码（从1开始）。"
            }
        },
        "required": ["logical_toc_with_ranges", "physical_page_of_chapter_1"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "page_offset": {
                "type": "integer",
                "description": "计算出的页码偏移量（物理页码 = 逻辑页码 + 偏移量）。"
            }
        }
    }
)
def calculate_page_offset(logical_toc_with_ranges: dict, physical_page_of_chapter_1: int) -> int:
    """
    根据带有范围的逻辑目录计算页码偏移量。
    """
    print(f"Executing calculate_page_offset(physical_page_of_chapter_1={physical_page_of_chapter_1})")
    if not logical_toc_with_ranges:
        raise ValueError("输入的目录 (logical_toc_with_ranges) 为空，无法计算偏移量。")
    try:
        # 通过寻找 "start" 值最小的条目来确定第一章
        first_chapter_title = min(
            logical_toc_with_ranges, 
            key=lambda k: logical_toc_with_ranges[k]['start']
        )
        logical_page_of_chapter_1 = logical_toc_with_ranges[first_chapter_title]['start']
        page_offset = physical_page_of_chapter_1 - logical_page_of_chapter_1
        print(f"  - Identified first chapter as '{first_chapter_title}' (Logical start page: {logical_page_of_chapter_1})")
        print(f"  - Calculated page offset: {page_offset}")
        return page_offset
    except KeyError:
        raise ValueError("目录条目格式错误，缺少 'start' 键。")
    except Exception as e:
        print(f"  - An unexpected error occurred: {e}")
        raise


@tool(
    name="count_lines",
    description="计算上传的第一个文件的行数。这个函数演示了如何接收由框架自动注入的文件内容。",
    task_type='cpu', # 文件解码和分割是CPU密集型操作
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "supplementary_files": {
                "type": "object",
                "description": "由框架注入的文件字典，键为文件名，值为文件的二进制内容。",
                "additionalProperties": {"type": "string", "format": "binary"}
            }
        },
        "required": ["supplementary_files"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "line_count": {
                "type": "integer",
                "description": "文件中的总行数"
            }
        }
    }
)
def count_lines(supplementary_files: dict) -> int:
    """计算上传的第一个文件的行数。"""
    print("Executing count_lines...")
    if not supplementary_files:
        print("  - No files received.")
        return 0
    
    first_filename = next(iter(supplementary_files))
    content_bytes = supplementary_files[first_filename]
    
    content_str = content_bytes.decode('utf-8')
    line_count = len(content_str.strip().split('\n'))
    
    print(f"  - Counted {line_count} lines in file '{first_filename}'.")
    return line_count

def _sanitize_filename(name):
    # 移除非法字符
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # 替换空格
    name = name.replace(" ", "_")
    return name

@tool(
    name="split_pdf_by_chapters",
    description="将PDF文件按章节切割，并将每个章节块作为独立的PDF文件保存到指定目录。",
    task_type='cpu',
    mem=1024,
    input_parameters={
        "type": "object",
        "properties": {
            "pdf_content": { "type": "string", "format": "binary" },
            "logical_toc_with_ranges": { "type": "object" },
            "page_offset": { "type": "integer" },
            "physical_page_of_chapter_1": { "type": "integer" },
            "output_directory": {
                "type": "string",
                "description": "用于保存切分后PDF文件的目录路径。"
            }
        },
        "required": ["pdf_content", "logical_toc_with_ranges", "page_offset", "physical_page_of_chapter_1", "output_directory"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "pdf_chunk_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "一个包含所有已保存的PDF文件块路径的列表。"
            }
        }
    }
)
def split_pdf_by_chapters(
    pdf_content: bytes, 
    logical_toc_with_ranges: dict, 
    page_offset: int, 
    physical_page_of_chapter_1: int,
    output_directory: str
) -> List[str]:
    print(f"Executing split_pdf_by_chapters (save to disk version)...")
    os.makedirs(output_directory, exist_ok=True) # 确保输出目录存在
    
    try:
        main_doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(main_doc)
        saved_paths = []

        # 1. 处理“前言”块
        if physical_page_of_chapter_1 > 1:
            preface_path = os.path.join(output_directory, "00_Preface.pdf")
            print(f"  - Saving 'Preface' chunk to '{preface_path}'")
            preface_doc = fitz.open()
            preface_doc.insert_pdf(main_doc, from_page=0, to_page=physical_page_of_chapter_1 - 2)
            preface_doc.save(preface_path)
            preface_doc.close()
            saved_paths.append(preface_path)

        # 2. 遍历并保存所有章节块
        for i, (title, ranges) in enumerate(logical_toc_with_ranges.items(), 1):
            # ... (页码计算和保护逻辑和之前一样) ...
            physical_start = ranges['start'] + page_offset
            physical_end = min(ranges['end'] + page_offset, total_pages)
            
            if not (1 <= physical_start <= total_pages and physical_start <= physical_end):
                continue
            
            # 创建安全的文件名
            safe_title = _sanitize_filename(title)
            filename = f"{i:02d}_{safe_title}.pdf"
            output_path = os.path.join(output_directory, filename)

            print(f"  - Saving chunk for '{title}' to '{output_path}'")
            chapter_doc = fitz.open()
            chapter_doc.insert_pdf(main_doc, from_page=physical_start - 1, to_page=physical_end - 1)
            chapter_doc.save(output_path)
            chapter_doc.close()
            saved_paths.append(output_path)

        main_doc.close()
        print(f"  - Successfully saved {len(saved_paths)} PDF chunks to '{output_directory}'.")
        return saved_paths

    except Exception as e:
        print(f"  - Error splitting and saving PDF: {e}")
        raise

@tool(
    name="scatter_chapter_in_memory",
    description="将一个章节的PDF二进制内容，在内存中按指定的页数切割成多个更小的PDF块（二进制列表）。",
    task_type='cpu',
    mem=1024,  # 为内存中的PDF操作预留足够空间
    input_parameters={
        "type": "object",
        "properties": {
            "chapter_pdf_content": {
                "type": "string",
                "format": "binary",
                "description": "单个章节的PDF二进制内容。"
            },
            "pages_per_chunk": {
                "type": "integer",
                "description": "每个更小的PDF块所包含的页数。"
            }
        },
        "required": ["chapter_pdf_content", "pages_per_chunk"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "page_chunk_contents": {
                "type": "array",
                "items": {"type": "string", "format": "binary"},
                "description": "一个列表，其中每一项都是一个小PDF块的二进制内容。"
            }
        }
    }
)
def scatter_chapter_in_memory(chapter_pdf_content: bytes, pages_per_chunk: int = 5) -> List[bytes]:
    """
    在内存中将一个PDF字节流按页数分割成多个PDF字节流。
    """
    print(f"Executing scatter_chapter_in_memory(pages_per_chunk={pages_per_chunk})...")
    if not chapter_pdf_content:
        print("  - Warning: Received empty PDF content. Returning empty list.")
        return []

    try:
        main_doc = fitz.open(stream=chapter_pdf_content, filetype="pdf")
        total_pages = len(main_doc)
        chunk_contents = []

        print(f"  - Splitting a {total_pages}-page PDF into chunks of up to {pages_per_chunk} pages.")

        # 按步长遍历所有页面
        for start_page_idx in range(0, total_pages, pages_per_chunk):
            end_page_idx = min(start_page_idx + pages_per_chunk - 1, total_pages - 1)
            
            # 创建一个新的、空白的内存PDF文档
            new_chunk_doc = fitz.open()
            # 将页面范围从主文档复制到新文档
            new_chunk_doc.insert_pdf(main_doc, from_page=start_page_idx, to_page=end_page_idx)
            
            # 将新文档序列化为字节并存入列表
            chunk_contents.append(new_chunk_doc.tobytes())
            new_chunk_doc.close()

        main_doc.close()
        print(f"  - Successfully created {len(chunk_contents)} PDF chunks in memory.")
        return chunk_contents

    except Exception as e:
        print(f"  - Error while scattering PDF in memory: {e}")
        raise

@tool(
    name="ocr_memory_chunk",
    description="对单一一小块PDF的二进制内容执行OCR，并返回所有页面的文本。",
    task_type='gpu',   # 明确这是一个GPU密集型任务
    mem=2048,
    gpu_mem=4096,      # 为OCR模型和图像处理预留显存
    input_parameters={
        "type": "object",
        "properties": {
            "pdf_chunk_content": {
                "type": "string",
                "format": "binary",
                "description": "由 scatter 工具生成的单个小PDF块的二进制内容。"
            }
        },
        "required": ["pdf_chunk_content"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "all_text_parts": {
                "type": "list",
                "description": "从该PDF块中识别出的每页文本的内容。"
            }
        }
    }
)
def ocr_memory_chunk(pdf_chunk_content: bytes) -> str:
    """
    接收一个PDF块的二进制内容，对其所有页面进行OCR，并返回合并后的文本。
    """
    if not pdf_chunk_content:
        print("  - Warning: Received empty PDF chunk for OCR. Returning empty string.")
        return ""
        
    # 注意：Reader在每个任务中独立初始化，这是分布式执行所必需的
    print(f"  - Executing ocr_memory_chunk on a new chunk...")
    print(f"    - Initializing EasyOCR reader on worker...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    
    try:
        doc = fitz.open(stream=pdf_chunk_content, filetype="pdf")
        all_text_parts = []
        
        print(f"    - Processing {len(doc)} pages in this chunk...")
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            ocr_results = reader.readtext(img_bytes)
            page_text = "\n".join([result[1] for result in ocr_results])
            all_text_parts.append(page_text)
        
        doc.close()
        
        full_text = "\n\n".join(all_text_parts) # 用双换行符分隔不同页面的文本
        print(f"    - Successfully OCR'd chunk, extracted {len(full_text)} characters.")
        return all_text_parts

    except Exception as e:
        print(f"  - Error during OCR processing of a chunk: {e}")
        raise

@tool(
    name="gather_ocr_results",
    description="将多个并行OCR任务返回的页面文本列表（一个嵌套列表），聚合成一个单一的、扁平化的页面文本列表。",
    task_type='cpu',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "ocr_texts": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "description": "一个嵌套列表，其每个内部列表都是一个OCR任务返回的页面文本列表。"
            }
        },
        "required": ["ocr_texts"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "flat_page_texts_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "一个扁平化的列表，其中每一项都是章节中某一页的完整文本。"
            }
        }
    }
)
def gather_ocr_results(ocr_texts: List[List[str]]) -> List[str]:
    """
    将一个嵌套的页面文本列表，扁平化成一个单一的页面文本列表。
    """
    print(f"Executing gather_ocr_results for {len(ocr_texts)} nested lists...")
    if not ocr_texts:
        return []

    # 使用列表推导式高效地将嵌套列表扁平化
    flat_list = [
        page_text
        for chunk_pages_list in ocr_texts
        for page_text in chunk_pages_list
    ]
    
    print(f"  - Successfully flattened nested lists into a single list with {len(flat_list)} pages.")
    return flat_list

@tool(
    name="split_text_for_summary",
    description="将一个扁平化的页面文本列表，按指定的页面数，重新组合成一个文本块列表，为并行摘要做准备。",
    task_type='cpu',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "flat_page_texts_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "由 gather_ocr_results 生成的、包含所有页面文本的扁平化列表。"
            },
            "pages_per_summary_chunk": {
                "type": "integer",
                "description": "每个摘要块应包含的页面数量。"
            }
        },
        "required": ["flat_page_texts_list", "pages_per_summary_chunk"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "summary_text_chunks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "一个列表，其中每一项都是由N页文本拼接而成的大文本块。"
            }
        }
    }
)
def split_text_for_summary(flat_page_texts_list: List[str], pages_per_summary_chunk: int = 10) -> List[str]:
    """
    将页面文本列表，按指定的页面数，重新组合成摘要块列表。
    """
    print(f"Executing split_text_for_summary to group {len(flat_page_texts_list)} pages into chunks of {pages_per_summary_chunk} pages each...")
    if not flat_page_texts_list:
        return []

    summary_chunks = []
    # 按指定的页数步长，遍历页面列表
    for i in range(0, len(flat_page_texts_list), pages_per_summary_chunk):
        # 获取当前块的页面文本
        page_group = flat_page_texts_list[i:i + pages_per_summary_chunk]
        # 将这些页面的文本合并成一个大的文本块
        chunk_text = "\n\n".join(page_group)
        summary_chunks.append(chunk_text)
    
    print(f"  - Successfully grouped pages into {len(summary_chunks)} summary chunks.")
    return summary_chunks



@tool(
    name="save_summary_to_md",
    description="将最终的章节摘要文本保存到一个以章节名命名的Markdown文件中。",
    task_type='io',
    mem=256,
    input_parameters={
        "type": "object",
        "properties": {
            "summary_text": {
                "type": "string",
                "description": "由 summarize_long_text 工具生成的最终摘要内容。"
            },
            "output_directory": {
                "type": "string",
                "description": "用于保存摘要 .md 文件的目录。"
            },
            "chapter_title": {
                "type": "string",
                "description": "章节的原始标题，用于生成文件名。"
            }
        },
        "required": ["summary_text", "output_directory", "chapter_title"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "summary_file_path": {
                "type": "string",
                "description": "最终保存的 Markdown 文件的完整路径。"
            }
        }
    }
)
def save_summary_to_md(summary_text: str, output_directory: str, chapter_title: str) -> str:
    """
    将摘要内容保存为Markdown文件。
    """
    print(f"Executing save_summary_to_md for chapter '{chapter_title}'...")
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 从章节标题创建安全的文件名
    safe_filename = _sanitize_filename(chapter_title) + ".md"
    file_path = os.path.join(output_directory, safe_filename)
    
    # 将标题作为一级标题写入文件内容
    content_to_save = f"# {chapter_title}\n\n{summary_text}"
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_to_save)
        
        print(f"  - Successfully saved summary to '{file_path}'")
        return file_path
    except Exception as e:
        print(f"  - Error saving summary file: {e}")
        raise

@tool(
    name="scan_chapters_directory",
    description="扫描指定目录，为所有PDF文件生成一个包含路径、标题和页数的信息列表。",
    task_type='io',
    mem=256,
    input_parameters={
        "type": "object", "properties": {"directory_path": {"type": "string"}}, "required": ["directory_path"]
    },
    output_parameters={
        "type": "object", "properties": {"chapters_info": {"type": "array", "items": {"type": "object"}}}
    }
)
def scan_chapters_directory(directory_path: str) -> List[dict]:
    """扫描目录，获取所有章节PDF的信息。"""
    chapters = []
    print(f"Executing scan_chapters_directory on '{directory_path}'...")
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    pdf_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')])

    for filename in pdf_files:
        path = os.path.join(directory_path, filename)
        title, _ = os.path.splitext(filename)
        try:
            with fitz.open(path) as doc:
                pages = len(doc)
            chapters.append({'path': path, 'title': title, 'pages': pages})
        except Exception as e:
            print(f"  - Warning: Could not process file {filename}: {e}")

    print(f"  - Found {len(chapters)} chapter PDF files.")
    return chapters

@tool(
    name="assemble_final_report",
    description="将所有章节的Markdown摘要文件，按顺序汇编成一份完整的报告。",
    task_type='io',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "summary_md_paths": {"type": "array", "items": {"type": "string"}},
            "book_title": {"type": "string"},
            "output_directory": {"type": "string"}
        },
        "required": ["summary_md_paths", "book_title", "output_directory"]
    },
    output_parameters={
        "type": "object", "properties": {"final_report_path": {"type": "string"}}
    }
)
def assemble_final_report(
    summary_md_paths: List[str],
    book_title: str,
    output_directory: str
) -> str:
    """
    将多个Markdown文件合并成一个最终报告。
    """
    print(f"Executing assemble_final_report to combine {len(summary_md_paths)} markdown files...")
    os.makedirs(output_directory, exist_ok=True)

    final_report_content = [f"# {book_title} - 完整摘要报告\n\n"]

    # 按顺序读取并拼接文件
    for md_path in sorted(summary_md_paths):
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                final_report_content.append(f.read())
        except Exception as e:
            print(f"  - Warning: Could not read file {md_path}: {e}")

    final_report_text = "\n\n---\n\n".join(final_report_content)

    # 保存最终报告
    safe_book_title = re.sub(r'[\\/*?:"<>|]', "", book_title).replace(" ", "_")
    final_report_path = os.path.join(output_directory, f"{safe_book_title}_完整报告.md")

    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write(final_report_text)

    print(f"  - Successfully assembled and saved final report to '{final_report_path}'")
    return final_report_path

@tool(
    name="load_markdown_files",
    description="扫描指定目录，加载所有章节摘要.md文件，并返回一个包含标题和内容的结构化列表。",
    task_type='io',
    mem=256,
    input_parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "存放所有章节摘要 .md 文件的目录路径。"
            }
        },
        "required": ["directory_path"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "chapter_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"}
                    }
                },
                "description": "一个字典列表，每一项包含一个章节的标题和完整的摘要内容。"
            }
        }
    }
)
def load_markdown_files(directory_path: str) -> List[Dict[str, str]]:
    """
    从指定目录加载所有 .md 文件，并返回其内容列表。
    """
    print(f"Executing load_markdown_files from directory '{directory_path}'...")
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"指定的目录不存在: {directory_path}")

    summaries = []
    try:
        # 获取所有.md文件并按文件名排序，以确保章节顺序正确
        md_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.md')])
        
        for filename in md_files:
            file_path = os.path.join(directory_path, filename)
            # 从文件名中提取标题 (去掉 .md 后缀)
            title, _ = os.path.splitext(filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            summaries.append({'title': title, 'content': content})
        
        print(f"  - Successfully loaded {len(summaries)} markdown files.")
        return summaries
    except Exception as e:
        print(f"  - Error while loading markdown files: {e}")
        raise

@tool(
    name="assemble_final_report",
    description="将全书摘要和所有筛选后的章节详细摘要，按顺序汇编成一份最终的、完整的报告文件。",
    task_type='io',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "global_summary": {"type": "string"},
            "all_chapter_summaries": {
                "type": "array",
                "items": {"type": "object"}
            },
            "book_title": {"type": "string"},
            "output_directory": {"type": "string"}
        },
        "required": ["global_summary", "all_chapter_summaries", "book_title", "output_directory"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "final_report_path": {"type": "string"}
        }
    }
)
def assemble_final_report(
    global_summary: str,
    all_chapter_summaries: List[Dict[str, str]],
    book_title: str,
    output_directory: str
) -> str:
    """
    将多个Markdown文件合并成一个最终报告。
    """
    print(f"Executing assemble_final_report to create the final report for '{book_title}'...")
    os.makedirs(output_directory, exist_ok=True)
    
    # 定义需要从最终报告中排除的非正文章节的关键词
    filter_keywords = ['preface', '目录', '前言', '后记', 'appendix']
    
    # 1. 添加主标题和全书摘要
    final_report_content = [
        f"# {book_title} - 完整摘要报告\n",
        "## 全书摘要\n",
        global_summary
    ]
    
    print(f"  - Assembling report with global summary and {len(all_chapter_summaries)} chapter summaries...")
    
    # 2. 筛选并拼接正式章节的详细摘要
    for chapter in all_chapter_summaries:
        title = chapter.get('title', '').lower()
        content = chapter.get('content', '')
        
        # 检查标题是否包含任何需要过滤的关键词
        if any(keyword in title for keyword in filter_keywords):
            print(f"  - Skipping non-essential chapter: '{chapter.get('title')}'")
            continue
        
        final_report_content.append(f"\n\n---\n\n{content}")

    final_report_text = "".join(final_report_content)
    
    # 3. 保存最终报告
    # (假设 _sanitize_filename 辅助函数已存在于此文件中)
    safe_book_title = _sanitize_filename(book_title)
    final_report_path = os.path.join(output_directory, f"{safe_book_title}_完整摘要报告.md")
    
    try:
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write(final_report_text)
        print(f"  - Successfully assembled and saved final report to '{final_report_path}'")
        return final_report_path
    except Exception as e:
        print(f"  - Error saving the final report: {e}")
        raise