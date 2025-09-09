import sys
import asyncio
import argparse
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
# MODIFIED: 导入我们定义好的具体 Agent 类
from worker_agent import *
from autogen_core import try_get_known_serializers_for_type
# NEW: 将所有可供启动的 Agent 类放入一个字典，方便动态查找
# 以后有新的 Agent 类，只需在这里添加即可
AGENT_CLASS_MAP = {
    "gaia_file": GAIA_File_Process_Agent,
    "gaia_reason": GAIA_Reason_Agent,
    "gaia_speech": GAIA_Speech_Agent,
    "gaia_vision": GAIA_Vision_Agent,
    "tbench_airline_book": tbench_airline_book_Agent,
    "tbench_airline_cancel": tbench_airline_cancel_Agent,
    "tbench_retail_cancel": tbench_retail_cancel_Agent,
    "tbench_retail_return": tbench_retail_return_Agent,
    "tbench_retail_modify": tbench_retail_modify_Agent,
    "tbench_retail_cancel_modify": tbench_retail_cancel_modify_Agent,
    # 新增 openagi 系列
    "openagi_document_qa": openagi_document_qa_Agent,
    "openagi_image_captioning_complex": openagi_image_captioning_complex_Agent,
    "openagi_multimodal_vqa_complex": openagi_multimodal_vqa_complex_Agent,
    "openagi_text_processing_multilingual": openagi_text_processing_multilingual_Agent
}

# NOTE:
# 该脚本运行在各个计算结点
# 通过计算结点启动Worker，再注册到主节点上
async def main():
    # python run_worker.py --host 127.0.0.1:5003 --name agent1 --workflow_type gaia_file
    parser = argparse.ArgumentParser(description="启动一个 AutoGen Worker Agent")
    parser.add_argument("--host", type=str, required=True, help="要连接的 Host 地址，例如: 127.0.0.1:5003")
    parser.add_argument("--name", type=str, required=True, help="要注册的 Agent 的唯一名称，例如: gaia_file_agent_1")
    parser.add_argument("--workflow_type", type=str, required=True, choices=AGENT_CLASS_MAP.keys(), help="workflow的类型，即dispatch_task中的dag_source_dag_type")
    parser.add_argument("--grpc_max_len", default= 100* 1024* 1024,
                        help="Sampling temperature for model output (0.0-1.0, lower = more deterministic)")
    args = parser.parse_args()
    agent_name = args.name
    host_address = args.host
    workflow_type = args.workflow_type
    
    # 从字典中动态获取要实例化的类
    AgentClass = AGENT_CLASS_MAP[workflow_type]

    print(f"🚀 启动 Worker: Agent名='{agent_name}', 类='{workflow_type}'")
    print(f"   -> 连接主节点: {host_address}")
    extra_grpc_config = [
        ("grpc.max_send_message_length", args.grpc_max_len),
        ("grpc.max_receive_message_length", args.grpc_max_len),
    ]
    # 启动 gRPC Worker Runtime
    worker = GrpcWorkerAgentRuntime(host_address=host_address, extra_grpc_config= extra_grpc_config)
    await worker.start()
    print("   -> 正在注册消息序列化器...")
    worker.add_message_serializer(try_get_known_serializers_for_type(DAGMessage))
    worker.add_message_serializer(try_get_known_serializers_for_type(AckMessage))
    print("   -> 序列化器注册完毕。")
    # MODIFIED: 更新注册逻辑，使用 lambda 表达式来传递初始化所需的参数
    # lambda 在这里创建了一个延迟执行的函数，当 Host 需要创建实例时，这个函数才会被调用
    await AgentClass.register(
        worker, 
        agent_name, 
        lambda: AgentClass(id= agent_name, workflow_type= workflow_type)
    )

    print(f"✅ Agent '{agent_name}' 已成功注册到 Host，并开始等待任务。")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())