import os
from maze.agent.dag_agent import DAGAgent
from maze.library.tools.file_tools import (
    load_pdf,
    extract_text_from_pdf_range,
    calculate_page_offset,
    split_pdf_by_chapters
)
from maze.library.tools.llm_tools import parse_toc_with_llm
from maze.agent.config import config # 导入全局配置

def split_pdf_by_chapters_workflow(pdf_path: str, physical_page_of_chapter_1: int):
    """
    构建并执行第一阶段的PDF章节切分工作流。

    Args:
        pdf_path (str): 要处理的PDF文件的路径。
        physical_page_of_chapter_1 (int): 正文第一章所在的物理页码（从1开始）。
    
    Returns:
        list or None: 成功则返回包含PDF块的列表，失败则返回None。
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        return None

    # --- 1. 初始化 Agent ---
    print("--- [Step 1: Initializing Workflow Agent] ---")
    agent = DAGAgent(name="PDF Chapter Splitting Workflow")

    # --- 2. 定义工作流（添加并连接任务） ---
    print("--- [Step 2: Defining Workflow Tasks] ---")

    task_a_id = agent.add_task(
        func=load_pdf,
        task_name="load_source_pdf",
        inputs={"pdf_path": pdf_path}
    )

    toc_range = (1, physical_page_of_chapter_1 - 1)
    task_b_id = agent.add_task(
        func=extract_text_from_pdf_range,
        task_name="extract_toc_text_with_ocr",
        inputs={
            "pdf_content": f"{task_a_id}.output.pdf_content",
            "page_range": toc_range
        }
    )

    task_c_id = agent.add_task(
        func=parse_toc_with_llm,
        task_name="parse_toc_to_json",
        inputs={
            "toc_text": f"{task_b_id}.output.extracted_text",
            "pdf_content": f"{task_a_id}.output.pdf_content"
        }
    )

    task_d_id = agent.add_task(
        func=calculate_page_offset,
        task_name="calculate_page_offset",
        inputs={
            "logical_toc_with_ranges": f"{task_c_id}.output.logical_toc_with_ranges",
            "physical_page_of_chapter_1": physical_page_of_chapter_1
        }
    )

    task_e_id = agent.add_task(
        func=split_pdf_by_chapters,
        task_name="split_pdf_into_chunks",
        inputs={
            "pdf_content": f"{task_a_id}.output.pdf_content",
            "logical_toc_with_ranges": f"{task_c_id}.output.logical_toc_with_ranges",
            "page_offset": f"{task_d_id}.output.page_offset",
            "physical_page_of_chapter_1": physical_page_of_chapter_1
        }
    )
    print("--- [Step 2: Workflow definition complete] ---")

    # --- 3. 可视化工作流 (在ipynb中会自动显示) ---
    print("\n--- [Step 3: Visualizing Workflow DAG] ---")
    return agent
