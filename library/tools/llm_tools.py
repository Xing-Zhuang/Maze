import os
import re
import json
import fitz
import requests
import tiktoken
import concurrent.futures
from typing import List, Dict
from maze.library.tools.definitions import tool

def _count_token(text: str, model_name: str= "gpt-4o")-> int:
    if text== "":
        return 0
    tokenizer= tiktoken.encoding_for_model(model_name)
    return len(tokenizer.encode(text))

def _query_llm_online(
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """一个内部辅助函数，用于调用在线LLM服务。"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # 确保URL是完整的
    api_url = os.path.join(base_url, "chat/completions")
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].lstrip()
    except requests.exceptions.RequestException as e:
        print(f"[LLM Error] API request failed: {str(e)}")
        return f"API request failed: {str(e)}"

def _fault_tolerance4token(
    content: str,
    token_number: int,
    max_tokens: int,
    free_ratio: float= 0.35
)-> str:
    _max_tokens= max_tokens* (1- free_ratio)
    if token_number> _max_tokens:
        return content[0: -(token_number- _max_tokens)]
    else:
        return content

@tool(
    name="answer_question_by_analyze_file_content_using_llm",
    description="使用LLM分析文件并回答问题",
    task_type='gpu',
    mem=1024,
    gpu_mem=20000,
    input_parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "原始文本"
            },
            "question": {
                "type": "string",
                "description": "问题"
            },
            "api_key": { "type": "string" },
            "base_url": { "type": "string" },
            "model": { "type": "string" },
            "temperature": {"type": "number"},
            "max_tokens": {"type": "number"}
        },
        "required": ["content", "question"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "根据文本，回答问题的内容"
            }
        }
    }
)
def answer_question_by_analyze_file_content_using_llm(
    content: str,
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: float
)-> str:
    prompt_templete= (
        "#Background#\n"
        "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].\n"
        "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
        "If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
        "If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.\n"
        "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
        "#Question#\n{question}\n"
        "#Extracted text from file#\n"
        "{content}\n"
    )
    prompt= prompt_templete.format(question= question, content= content)
    prompt_count= _count_token(prompt, model_name= "gpt-4o")
    content= _fault_tolerance4token(content, prompt_count, max_tokens)
    prompt = prompt_templete.format(question= question, content= content)
    answer= _query_llm_online(
        messages= [{"role": "user", "content": prompt}], 
        api_key= api_key, 
        base_url= base_url, 
        model= model,
        temperature= temperature,
        max_tokens= max_tokens)
    return answer

@tool(
    name="answer_question_by_fuse_llm_output",
    description="融合两个LLM的输出以回答问题",
    task_type='gpu',
    mem=1024,
    gpu_mem=2048,
    input_parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "问题"
            },
            "answer1": {
                "type": "string",
                "description": "LLM1的回答"
            },
            "answer2": {
                "type": "string",
                "description": "LLM2的回答"
            },
            "api_key": { "type": "string" },
            "base_url": { "type": "string" },
            "model": { "type": "string" },
            "temperature": {"type": "number"},
            "max_tokens": {"type": "number"}
        },
        "required": ["content", "question"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "根据文本，回答问题的内容"
            }
        }
    }
)
def answer_question_by_fuse_llm_output(
    question: str,
    answer1: str,
    answer2: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: float
)-> str:
    prompt_templete= (
            "You are a senior editor and a world-class reasoning expert. Your job is to synthesize the answers from two different AI assistants to produce one final, superior answer for the given question.\n\n"
            "--- Original Question ---\n{question}\n\n"
            "--- Answer from Assistant 1 (Qwen3) ---\n{answer1}\n\n"
            "--- Answer from Assistant 2 (DeepSeek) ---\n{answer2}\n\n"
            "--- Your Task ---\n"
            "Analyze both answers. Identify the strengths and weaknesses of each. Then, combine their best elements, correct any errors, and provide a single, comprehensive, and accurate final answer. Adhere to the final answer format requested in the original prompt.\n\n"
            "Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]."
        )
    prompt= prompt_templete.format(question= question, answer1= answer1, answer2= answer2)
    answer= _query_llm_online(
        messages= [{"role": "user", "content": prompt}], 
        api_key= api_key,
        base_url= base_url,
        model= model,
        temperature= temperature,
        max_tokens= max_tokens)
    return answer

@tool(
    name="parse_toc_with_llm",
    description="使用LLM解析目录文本，并结合PDF总页数，生成包含起始和结束页码的结构化章节地图。",
    task_type='gpu',
    mem=1024,
    gpu_mem=2048,
    input_parameters={
        "type": "object",
        "properties": {
            "toc_text": {
                "type": "string",
                "description": "从PDF目录页面提取的原始文本。"
            },
            "pdf_content": {
                "type": "string",
                "format": "binary",
                "description": "完整的PDF二进制内容，用于获取总页数。"
            },
            "api_key": { "type": "string" },
            "base_url": { "type": "string" },
            "model": { "type": "string" },
            "temperature": {"type": "number"},
            "max_tokens": {"type": "number"}
        },
        "required": ["toc_text", "pdf_content", "api_key", "base_url", "model"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "logical_toc_with_ranges": {
                "type": "object",
                "description": "一个字典，键是章节标题，值是包含 'start' 和 'end' 页码的字典。"
            }
        }
    }
)
def parse_toc_with_llm(toc_text: str, pdf_content: bytes, api_key: str, base_url: str, model: str, temperature: float, max_tokens: float) -> dict:
    """
    通过LLM将目录文本转换为包含起始和结束页码的结构化字典。
    """
    print(f"Executing parse_toc_with_llm (v2 with ranges)...")

    # 在工具内部直接获取总页数
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        print(f"  - PDF total pages: {total_pages}")
    except Exception as e:
        raise ValueError(f"无法解析PDF内容以获取总页数: {e}")

    system_prompt = f"""
You are an expert assistant specializing in parsing document table of contents (ToC).
Your task is to convert the user-provided raw ToC text into a structured JSON object that defines the start and end page for each chapter. The document has a total of {total_pages} pages.

Rules:
1. The output must be a single JSON object.
2. The keys of the object are the full, cleaned chapter titles.
3. The values of the object are another dictionary with two keys: "start" and "end".
4. The "start" page is the page number listed in the ToC. All page numbers must be integers.
5. The "end" page for a chapter is the page number right before the "start" page of the *next* chapter.
6. For the *very last* chapter in the ToC, its "end" page is the total number of pages in the document, which is {total_pages}.
7. Clean the chapter titles: remove dot leaders (e.g., '.......') and extra whitespace.
8. Only include entries that are actual chapters with page numbers. Ignore preface sections with Roman numerals (e.g., 'i', 'v').

Example Input Text (assuming total pages is 300):
'''
Contents
Chapter 1: The Beginning ......... 1
Chapter 2: The Adventure ....... 15
Appendix ......................... 250
'''

Example Output JSON:
{{
  "Chapter 1: The Beginning": {{
    "start": 1,
    "end": 14
  }},
  "Chapter 2: The Adventure": {{
    "start": 15,
    "end": 249
  }},
  "Appendix": {{
    "start": 250,
    "end": 300
  }}
}}

Your final output must be ONLY the JSON object, with no other explanations.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the table of contents text:\n\n{toc_text}"}
    ]
    
    try:
        response_str = _query_llm_online(
            messages=messages, api_key=api_key, base_url=base_url, model=model,
            temperature=temperature, max_tokens=max_tokens
        )
        
        if "```json" in response_str:
            response_str = response_str.split("```json\n")[1].split("```")[0]
        
        logical_toc_with_ranges = json.loads(response_str)
        print(f"  - Successfully parsed ToC into {len(logical_toc_with_ranges)} ranged entries.")
        return logical_toc_with_ranges

    except json.JSONDecodeError:
        print(f"  - Error: LLM returned invalid JSON. Response: {response_str}")
        raise ValueError("LLM did not return a valid JSON object for the ToC ranges.")
    except Exception as e:
        print(f"  - Error querying LLM or processing response: {e}")
        raise

@tool(
    name="summarize_text_chunk",
    description="使用一个结构化的prompt，对一个由多页文本组成的文本块进行深入的初步总结。",
    task_type='gpu',
    mem=1024,
    gpu_mem=4096,
    input_parameters={
        "type": "object",
        "properties": {
            "text_chunk": {"type": "string", "description": "由 split_text_for_summary 生成的文本块。"},
            "book_title": {"type": "string", "description": "书籍的标题。"},
            "chapter_title": {"type": "string", "description": "当前章节的标题。"},
            "api_key": { "type": "string" },
            "base_url": { "type": "string" },
            "model": { "type": "string" },
            "temperature": {"type": float},
            "max_tokens": {"type": float}
        },
        "required": ["text_chunk", "book_title", "chapter_title"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "chunk_summary": {"type": "string", "description": "对该文本块的结构化初步摘要。"}
        }
    }
)
def summarize_text_chunk(text_chunk: str, book_title: str, chapter_title: str, api_key: str, base_url: str, model: str, temperature: float, max_tokens: float) -> str:
    """
    对单个文本块，应用高质量的结构化prompt进行总结。
    """
    print(f"Executing summarize_text_chunk for a part of chapter {chapter_title}...")

    # 使用您提供的、高质量的prompt模板
    prompt = f"""
你是一名北京大学的马克思主义研究学者，你现在需要总结《{book_title}》中《{chapter_title}》章节的片段。
请依据以下提供的“核心内容浓缩”作为主要信息来源，严格按照6点结构输出，确保内容全面、逻辑清晰、语言简洁：

--- 片段开始 ---
{text_chunk}
--- 片段结束 ---

1. **片段定位**
   - 说明片段说了哪些重要内容，之间什么联系

2. **核心概念**
   - 列出一些关键概念（含定义、特点、易混淆点）
   - 说明概念之间的关系（对立/统一/因果/并列等）

3. **重点内容**
   - 说明该片段中，哪些内容是重要的
   - 为什么重要
   - 内容之间是否有什么逻辑联系
   
4. **核心论点与证据**
   - 论点1 + 作者使用的证据（案例/数据/引用， 尽量写清楚时间、地点和人物）
   - 论点2 + 作者使用的证据（案例/数据/引用， 尽量写清楚时间、地点和人物）

5. **分析与思考**
   - 如果出现"分析与思考"部分，请你帮我提取对应的问题，你无需回答，仅提取文字就可以。
   
6. **小结**
   - 总结片段内容（联系片段中的重点，串联起片段中的概念进行总结）

7. **考试提醒**
   - 如果你是出卷人，你可能会如何出论述题，给出论述题和解答。

要求：
   - 避免冗余，保留关键细节
   - 你的资料是写给博士生用的，他们只想考试通过
   - 已知最后考试会有两个论述题，老师让学生关注19届6中全会、20大报告、23和24政府报告、十四五、十五五规划的内容
   - 核心概念不少于5个
   - 核心论点与证据不少于2个
   - 论述题，你给出1-2个就可以，论述题一定要围绕片段内容
   - 论述题解答逻辑如下
       对于非材料题：(1) 分析题干，提取关键信息，定位知识点； (2) 第一段总起，解释相关名词，并引出后面的论述；（3）分点阐述，结合所学知识和相关热点进行论述；（4）最后一段总结，可以写发展趋势，展望未来也可总结利弊等；（5）注意阐述的维度和思路清晰明了
       对于材料题：（1）通读材料和题干，提取关键信息，找到相关知识点；（2）第一段总起，阐明材料和题目相关内容；（3）分点阐述，结合材料分析，结合当下热点话题以及书上相关知识点，作进一步阐述，写出自己的观点；（4）最后一段总结（同非材料题）
   - 保持客观中立，必要时引用原文关键表述
   - 保持中立，实事求是，不要自己编造内容
   - 总结的字数不低于1500字，不超过3000字
"""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        summary = _query_llm_online(messages=messages, api_key= api_key, base_url= base_url, model= model,
            temperature= temperature, max_tokens= max_tokens)
        print(f"  - Successfully generated a structured summary for the chunk.")
        return summary
    except Exception as e:
        print(f"  - Error during structured summary generation for a chunk: {e}")
        raise

@tool(
    name="combine_chunk_summaries",
    description="聚合多个初步摘要，进行精确截断，然后使用结构化prompt生成最终的、完整的章节摘要。",
    task_type='gpu',
    mem=2048,
    gpu_mem=4096,
    input_parameters={
        "type": "object",
        "properties": {
            "chunk_summaries": {"type": "array", "items": {"type": "string"}},
            "book_title": {"type": "string"},
            "chapter_title": {"type": "string"},
            "api_key": {"type": "string"},
            "base_url": {"type": "string"},
            "model": {"type": "string"},
            "temperature": {"type": "number"},
            "max_tokens": {"type": "integer"}
        },
        "required": ["chunk_summaries", "book_title", "chapter_title", "api_key", "base_url", "model", "temperature", "max_tokens"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "final_summary_text": {"type": "string"}
        }
    }
)
def combine_chunk_summaries(
    chunk_summaries: List[str],
    book_title: str,
    chapter_title: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    聚合所有初步摘要，进行最终总结。
    """
    print(f"Executing combine_chunk_summaries for chapter '{chapter_title}'...")
    
    # 1. 聚合初步摘要
    combined_text = "\n\n---\n\n".join(chunk_summaries)
    print(f"  - Combined {len(chunk_summaries)} chunk summaries into a text of {len(combined_text)} chars.")

    # 2. 精确计算截断长度
    # (使用占位符来构建不含动态内容的prompt模板)
    prompt_template= f"""
你是一名北京大学的马克思主义研究学者，你会收到《{book_title}》中《{chapter_title}》章节的多个片段内容的总结。
你需要多个片段内容的总结进行二次加工，得到该章节的总结内容。
请依据以下提供的“片段总结”作为主要信息来源，严格按照6点结构输出，确保内容全面、逻辑清晰、语言简洁：
--- 片段总结开始 ---
{combined_text}
--- 片段总结结束 ---
1. **章节定位**
   - 说明章节说了哪些重要内容，之间什么联系

2. **核心概念**
   - 列出一些关键概念（含定义、特点、易混淆点，不少于10个）
   - 说明概念之间的关系（对立/统一/因果/并列等）

3. **重点内容**
   - 说明该章节中，哪些内容是重要的
   - 为什么重要
   - 内容之间是否有什么逻辑联系
   
4. **核心论点与证据**
   - 论点1 + 作者使用的证据（案例/数据/引用， 尽量写清楚时间、地点和人物）
   - 论点2 + 作者使用的证据（案例/数据/引用， 尽量写清楚时间、地点和人物）

5. **论述题**
   - 如果你是出卷人，你可能会如何出论述题，给出论述题和解答。

6. **分析与思考**
   - 如果出现"分析与思考"部分，请你帮我结合上下文进行回答，字数不限，尽量简洁，明了。
      
7. **小结**
   - 总结章节内容（联系章节中的重点，串联起章节中的概念进行总结）

要求：
   - 避免冗余，保留关键细节，最后输出的时候，不要有无关信息，比如“以下是根据您提供的多个片段总结，对《中国马克思主义与当代》第x章“xxx”章节的二次加工总结。内容严格依据您提供的片段总结，并按照6点结构输出，确保内容全面、逻辑清晰、语言简洁。”之类的
   - 你的资料是写给博士生用的，他们只想考试通过
   - 已知最后考试会有两个论述题，老师让学生关注19届6中全会、20大报告、23和24政府报告、十四五、十五五规划的内容
   - 核心概念不少于10个
   - 核心论点与证据不少于4个
   - 在你收到的总结中提取一些你认为最重要的论述题，且不少于2个，论述题一定要围绕本章内容，一定要有答案且答案要遵循下面论述题解答逻辑
   - 论述题解答逻辑如下
       对于非材料题：(1) 分析题干，提取关键信息，定位知识点； (2) 第一段总起，解释相关名词，并引出后面的论述；（3）分点阐述，结合所学知识和相关热点进行论述；（4）最后一段总结，可以写发展趋势，展望未来也可总结利弊等；（5）注意阐述的维度和思路清晰明了
       对于材料题：（1）通读材料和题干，提取关键信息，找到相关知识点；（2）第一段总起，阐明材料和题目相关内容；（3）分点阐述，结合材料分析，结合当下热点话题以及书上相关知识点，作进一步阐述，写出自己的观点；（4）最后一段总结（同非材料题）
   - 保持客观中立，必要时引用原文关键表述
   - 保持中立，实事求是，不要自己编造内容
   - 最终以markdown输出，要求大标题为2级，后面以此类推，重点内容可以加粗等方式表示，格式要求干净、整洁且尽量好看
   - 总结的字数不低于1500字，不超过3000字
"""
    # 填充模板但保留核心内容为空，计算固定部分长度
    filled_prompt_template = prompt_template.format(
        book_title=book_title, 
        chapter_title=chapter_title, 
        condensed_text=""  # 核心内容暂时为空
    )
    prompt_template_len = len(filled_prompt_template)
    
    # 核心修改：更科学的长度计算
    # 预留30-40%的token用于生成输出内容
    output_token_reserve = int(max_tokens * 0.35)
    input_available_tokens = max_tokens - output_token_reserve
    
    # 字符与token的转换系数（中文约2-3字符/token，英文约4-5字符/token）
    # 考虑混合文本，取3.5作为折中
    chars_per_token = 3.5
    
    # 计算可用于核心内容的最大字符数
    # 总可用字符 = 可用token * 转换系数 - 模板固定长度
    max_available_chars = int(input_available_tokens * chars_per_token) - prompt_template_len
    
    # 增加5%的安全余量
    max_content_len = int(max_available_chars * 0.95)
    
    # 确保不会出现负数（极端情况下的保护）
    max_content_len = max(max_content_len, 0)
    
    if len(combined_text) > max_content_len:
        print(f"  - Warning: Combined text ({len(combined_text)} chars) exceeds limit ({max_content_len}). Truncating.")
        combined_text = combined_text[:max_content_len]

    # 3. 填充最终的Prompt
    final_prompt = prompt_template.format(condensed_text=combined_text)
    messages = [{"role": "user", "content": final_prompt}]
    
    # 4. 调用LLM进行最终总结
    final_summary = _query_llm_online(
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print("  - Final structured summary generated.")
    return final_summary


@tool(
    name="get_chapter_core_idea",
    description="接收一篇详细的章节摘要，并使用LLM将其浓缩为几句话的核心思想。",
    task_type='gpu',
    mem=1024,
    gpu_mem=4096,
    input_parameters={
        "type": "object",
        "properties": {
            "chapter_info": {
                "type": "object",
                "description": "一个包含'title'和'content'的字典。"
            },
            "book_title": {"type": "string"},
            "api_key": {"type": "string"},
            "base_url": {"type": "string"},
            "model": {"type": "string"},
            "temperature": {"type": "number"},
            "max_tokens": {"type": "integer"}
        },
        "required": ["chapter_info", "book_title", "api_key", "base_url", "model", "temperature", "max_tokens"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "core_idea_summary": {"type": "string"}
        }
    }
)
def get_chapter_core_idea(
    chapter_info: Dict[str, str],
    book_title: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    对单个章节的详细摘要进行再浓缩，提炼出核心思想。
    """
    chapter_title = chapter_info['title']
    chapter_summary_content = chapter_info['content']
    print(f"Executing get_chapter_core_idea for chapter '{chapter_title}'...")

    prompt = f"""
你是一名北京大学的马克思主义研究学者，你的任务是为《{book_title}》一书的《{chapter_title}》这一章提炼核心思想。
你将收到一份关于该章节的详细摘要，请基于这份摘要，用5到10句话，高度概括出本章最核心、最精炼的观点和主线。

要求：
- 语言精炼，说明章节内在脉络，重点，直击要点。
- 总结必须能够独立成文，让读者一眼就能看懂本章的精华所在。
- 不要超过500字。

--- 章节详细摘要 ---
{chapter_summary_content}
--- 章节详细摘要结束 ---

核心思想总结：
"""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        core_idea = _query_llm_online(
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens # 这里的max_tokens可以设小一些，例如512
        )
        print(f"  - Successfully extracted core idea for chapter '{chapter_title}'.")
        return core_idea
    except Exception as e:
        print(f"  - Error while extracting core idea for chapter '{chapter_title}': {e}")
        raise

@tool(
    name="create_global_summary",
    description="接收一个包含所有章节核心思想的列表，调用LLM生成一份总揽全局的、高度概括的全书摘要。",
    task_type='gpu',
    mem=1024,
    gpu_mem=4096,
    input_parameters={
        "type": "object",
        "properties": {
            "core_idea_summaries": {"type": "array", "items": {"type": "string"}},
            "book_title": {"type": "string"},
            "api_key": {"type": "string"},
            "base_url": {"type": "string"},
            "model": {"type": "string"},
            "temperature": {"type": "number"},
            "max_tokens": {"type": "integer"}
        },
        "required": ["core_idea_summaries", "book_title", "api_key", "base_url", "model", "temperature", "max_tokens"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "global_summary_text": {"type": "string"}
        }
    }
)
def create_global_summary(
    core_idea_summaries: List[str],
    book_title: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    将所有章节的核心思想融合成一份全书摘要。
    """
    print(f"Executing create_global_summary for book '{book_title}'...")

    # 将核心思想列表格式化，以便LLM更好地理解
    combined_ideas = "\n".join(f"- 第 {i+1} 章核心思想: {idea}" for i, idea in enumerate(core_idea_summaries))

    prompt = f"""
你是一名北京大学的马克思主义研究学者，你的任务是为《{book_title}》这本书撰写一篇引人入胜的、高度概括的序言或简介。
你已经收到了这本书每个章节的核心思想摘要，现在需要你将这些碎片化的思想，编织成一段连贯、流畅、能体现全书逻辑脉络和核心价值的文字。

--- 各章节核心思想 ---
{combined_ideas}
--- 各章节核心思想结束 ---

请撰写一份全书摘要，要求：
1. 最终以markdown输出，要求大标题为1级，重点内容可以加粗等方式表示，格式要求干净、整洁且尽量好看
2. 你只需要搞个大标题，并隔行直接写序言
3. 篇幅适中: 总字数控制在1000字左右。
4. 你的资料是写给博士生用的，他们只想考试通过
5. 已知最后考试会有两个论述题，老师让学生关注19届6中全会、20大报告、23和24政府报告、十四五、十五五规划的内容，所以，你也可以在序言或简介这说明那几章是重点、考点和难点。
6. 高度概括: 不要罗列细节，而是抓住贯穿全书的主线和最重要的论点。
7. 逻辑清晰: 清晰地阐述这本书从头到尾的论证思路或叙事结构。
8. 保持中立，实事求是，不要自己编造内容
9. 不要有无关信息，比如“以下是根据您提供的多个片段总结....”，请直接markdown格式输出总结内容
"""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        global_summary = _query_llm_online(
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature, # 可以用稍高的温度增加一点创造性
            max_tokens=max_tokens
        )
        print(f"  - Successfully generated the global summary for the book.")
        return global_summary
    except Exception as e:
        print(f"  - Error while creating the global summary: {e}")
        raise