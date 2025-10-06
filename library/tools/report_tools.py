import os
import shutil
from typing import List, Dict
from maze.library.tools.definitions import tool, TYPE_FILEPATH, TYPE_FOLDERPATH

@tool(
    name="create_report_folder",
    description="在一个目录中创建多个报告文件，然后将其打包成一个zip压缩包。",
    task_type='cpu', # 这是一个CPU密集型任务
    mem=2048,
    input_parameters={
        "type": "object",
        "properties": {
            "data": {
                "type": "object", 
                "description": "用于生成报告的源数据"
            },
            "output_dir": {
                "type": TYPE_FOLDERPATH, 
                "description": "由框架提供的、用于存放报告文件的临时输出目录路径"
            }
        }
    },
    output_parameters={
        "type": "object",
        "properties": {
            "archive_path": {
                "type": TYPE_FILEPATH, 
                "description": "最终生成的zip压缩包的路径"
            }
        }
    }
)
def create_report_folder(data: dict, output_dir: str) -> str:
    """
    这个函数是一个完全通用的Python函数。
    它接收一个普通的字符串路径，向其中写入文件，然后打包并返回一个压缩包的路径。
    它完全不知道自己运行在分布式环境中。
    """
    # 1. 在框架提供的目录中自由创建文件
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Summary for data: {str(data)}")
    with open(os.path.join(output_dir, "raw_data.json"), "w", encoding="utf-8") as f:
        f.write(str(data))
    
    print(f"在目录 '{output_dir}' 中成功创建了多个报告文件。")

    # 2. 将整个目录打包成一个zip文件
    # 我们将压缩包放在输出目录的上一级，并命名为 report_archive
    archive_base_path = os.path.join(os.path.dirname(output_dir), "report_archive")
    archive_path = shutil.make_archive(archive_base_path, 'zip', output_dir)
    
    print(f"目录已成功打包到: {archive_path}")

    # 3. 返回这个zip压缩包的路径
    return archive_path

@tool(
    name="analyze_folder",
    description="分析一个文件夹内的所有文件，并统计文件数量和总大小。",
    task_type='io', # 这是一个IO密集型任务
    mem=1024,
    input_parameters={
        "type": "object",
        "properties": {
            "input_dir": {
                "type": TYPE_FOLDERPATH, 
                "description": "包含待分析文件的文件夹路径"
            }
        }
    },
    output_parameters={
        "type": "object",
        "properties": {
            "analysis_result": {
                "type": "object", 
                "description": "包含文件数量和总大小的分析结果字典"
            }
        }
    }
)
def analyze_folder(input_dir: str) -> dict:
    """
    这个函数也是一个完全通用的Python函数。
    它接收一个普通的目录路径，并对其内容进行分析。
    """
    print(f"开始分析目录 '{input_dir}' 中的所有文件...")
    
    file_count = 0
    total_size = 0
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            file_count += 1
            total_size += os.path.getsize(file_path)
            
    return {"file_count": file_count, "total_size_bytes": total_size}

@tool(
    name="filter_chapter_summaries",
    description="过滤摘要结果列表，只保留正式章节的摘要，去除前言、结尾等非正文部分。",
    task_type='cpu',
    mem=512,
    input_parameters={
        "type": "object",
        "properties": {
            "all_summary_results": {
                "type": "array",
                "description": "包含所有部分（前言、章节、结尾）摘要的列表。每个元素是一个包含'title'和'summary'的字典。",
                "items": {
                    "type": ["object", "null"], # 允许列表中的元素为None
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"}
                    }
                }
            }
        },
        "required": ["all_summary_results"]
    },
    output_parameters={
        "type": "object",
        "properties": {
            "filtered_summaries": {
                "type": "array",
                "description": "一个只包含正式章节摘要的字典列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "required": ["title", "summary"]
                }
            }
        }
    }
)
def filter_chapter_summaries(all_summary_results: List[Dict]) -> List[Dict]:
    """
    根据标题过滤摘要列表，只保留正式章节。
    """
    print(f"开始过滤 {len(all_summary_results)} 条摘要结果...")
    
    filtered_list = []
    # 定义非章节标题的关键词
    non_chapter_keywords = ["前言", "引言", "序", "结尾", "后记", "附录"]
    
    for result in all_summary_results:
        # 跳过由“固定大小并行槽”策略产生的None元素
        if result is None:
            continue
            
        title = result.get("title", "")
        
        # 检查标题是否包含任何非章节关键词
        is_non_chapter = any(keyword in title for keyword in non_chapter_keywords)
        
        if not is_non_chapter:
            filtered_list.append(result)
            print(f"  - [保留] 正式章节: '{title}'")
        else:
            print(f"  - [过滤] 非正文章节: '{title}'")
            
    print(f"过滤完成，保留了 {len(filtered_list)} 个正式章节的摘要。")
    return filtered_list