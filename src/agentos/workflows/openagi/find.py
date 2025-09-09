# find_path.py
import ray
import sys

print("Connecting to the existing Ray cluster...")
# 连接到您已经通过 "ray start --head" 启动的集群
ray.init(address='auto')
print("Connection successful.")

@ray.remote
def get_executed_file_path():
    """
    这个函数将在 Ray 的一个工作节点上运行，
    就像您那个失败的 task 一样。
    """
    try:
        # 我们导入那个一直出问题的模块
        import agentos.workflows.openagi.document_qa.task
        
        # 我们让 Python 告诉我们这个模块的文件路径
        return agentos.workflows.openagi.document_qa.task.__file__
    except Exception as e:
        return f"Error importing module: {e}"

print("Running remote task to find the file path...")
# 运行远程函数并获取结果
actual_path = ray.get(get_executed_file_path.remote())
print("Task completed.")

print("\n" + "="*80)
print(">>> 诊断结果 / DIAGNOSTIC RESULT <<<")
print(f"\nRay 工作节点实际执行的文件是:\n{actual_path}")
print("\n" + "="*80)

print("\n请比较上面的路径和您正在编辑的路径:")
print(f"/home/hustlbw/gujing/AgentOS/src/agentos/workflows/openagi/document_qa/task.py")
print("\n如果两个路径不一致，就找到了问题的根源！")

ray.shutdown()
