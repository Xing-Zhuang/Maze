#!/usr/bin/env python3
# test_openagi_specific.py

import os
import sys
import argparse

# 添加项目路径到 Python 路径
sys.path.append('/home/hustlbw/AgentOS/src')

from agentos.utils.query_loader import OpenAGILoader

def check_task_structure(task_path: str, task_id: str, category: str):
    """检查任务文件夹结构"""
    print(f"\n📁 检查任务结构: {task_path}")
    
    if not os.path.exists(task_path):
        print(f"❌ 任务文件夹不存在: {task_path}")
        return False
    
    inputs_path = os.path.join(task_path, "inputs")
    outputs_path = os.path.join(task_path, "outputs")
    
    print(f"📂 inputs 文件夹: {'✅' if os.path.exists(inputs_path) else '❌'}")
    print(f"📂 outputs 文件夹: {'✅' if os.path.exists(outputs_path) else '❌'}")
    
    # 检查 inputs 文件夹内容
    if os.path.exists(inputs_path):
        input_files = os.listdir(inputs_path)
        print(f"📄 inputs 文件: {input_files}")
        
        # 检查问题文件
        question_files = [f for f in input_files if f in ['question.txt', 'questions.txt']]
        print(f"❓ 问题文件: {question_files}")
    
    # 检查 outputs 文件夹内容
    if os.path.exists(outputs_path):
        output_files = os.listdir(outputs_path)
        print(f"📄 outputs 文件: {output_files}")
        
        # 检查答案文件
        answer_files = [f for f in output_files if f in ['answers.txt', 'labels.txt']]
        print(f"💡 答案文件: {answer_files}")
    
    return True

def test_openagi_loader(task_id: str, category: str):
    """测试 OpenAGILoader 对特定任务的处理"""
    
    print(f"\n🧪 测试 OpenAGILoader - 任务 {task_id} ({category})")
    print("=" * 60)
    
    # 设置参数
    args = argparse.Namespace()
    args.proj_path = "/home/hustlbw/AgentOS"
    args.data_path = "data"
    args.dag_path = "dag"
    
    # 检查任务文件夹结构
    task_path = os.path.join(args.proj_path, args.data_path, "openagi", category, task_id)
    if not check_task_structure(task_path, task_id, category):
        return
    
    # 获取补充文件列表
    inputs_path = os.path.join(task_path, "inputs")
    supplementary_files = []
    if os.path.exists(inputs_path):
        supplementary_files = [f for f in os.listdir(inputs_path) if os.path.isfile(os.path.join(inputs_path, f))]
    
    print(f"\n📎 补充文件列表: {supplementary_files}")
    
    try:
        # 创建 OpenAGILoader 实例
        print(f"\n🔧 创建 OpenAGILoader 实例...")
        loader = OpenAGILoader(
            args=args,
            dag_id=task_id,
            dag_type=category,
            dag_source="openagi",
            supplementary_files=supplementary_files
        )
        
        print(f"\n📋 加载结果:")
        print(f"  🔤 任务ID: {loader.dag_id}")
        print(f"  📂 任务类型: {loader.dag_type}")
        print(f"  📚 数据源: {loader.dag_source}")
        
        print(f"\n❓ 问题内容:")
        print(f"  {loader.question}")
        
        print(f"\n💡 答案内容:")
        if loader.answer:
            # 如果答案太长，只显示前200个字符
            answer_preview = loader.answer[:200] + "..." if len(loader.answer) > 200 else loader.answer
            print(f"  {answer_preview}")
        else:
            print(f"  (无答案或答案为空)")
        
        print(f"\n📎 补充文件路径:")
        supplementary_paths = loader.get_supplementary_files()
        if supplementary_paths:
            for filename, filepath in supplementary_paths.items():
                file_exists = "✅" if os.path.exists(filepath) else "❌"
                print(f"  {file_exists} {filename}: {filepath}")
        else:
            print(f"  (无补充文件)")
        
        print(f"\n✅ 测试成功完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 OpenAGI Loader 测试程序")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        ("175", "document_qa"),
        ("108", "text_processing_multilingual")
    ]
    
    for task_id, category in test_cases:
        test_openagi_loader(task_id, category)
    
    print(f"\n🎉 所有测试完成!")

if __name__ == "__main__":
    main()