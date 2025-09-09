import argparse
import agentscope
from agentscope.server import RpcAgentServerLauncher
from worker_agent import * # 导入具体的 Agent 类

# NEW: 将所有可供启动的 Worker Agent 类放入一个字典，方便动态查找
# 以后有新的工人Agent类，只需在这里添加即可
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
if __name__ == "__main__":
    # NEW: 使用 argparse 来处理命令行参数，更清晰、更健壮
    parser = argparse.ArgumentParser(description="启动一个 AgentScope Worker Agent。")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Worker Agent 服务监听的主机地址。")
    parser.add_argument("--port", type=int, required=True, help="Worker Agent 服务监听的端口。")
    parser.add_argument("--name", type=str, required=True, help="要注册的 Agent 的唯一名称，例如: agent1")
    parser.add_argument("--workflow_types", type=str, required=True, default="gaia_file,gaia_speech", help="此 Agent 负责处理的工作流类型，例如: gaia_file")
    args = parser.parse_args()

    # 从字典中动态获取要实例化的类
    workflow_types_to_run = [s.strip() for s in args.workflow_types.split(',')]
    agent_classes_to_run = []
    for wt_type in workflow_types_to_run:
        agent_class = AGENT_CLASS_MAP.get(wt_type)
        if agent_class:
            agent_classes_to_run.append(agent_class)
        else:
            print(f"⚠️ 警告: 在 AGENT_CLASS_MAP 中未找到类型 '{wt_type}' 对应的 Agent 类，将跳过。")
    
    if not agent_classes_to_run:
        print("❌ 错误: 没有找到任何有效的 Agent 类型可以注册，程序退出。")
    else:
        print(f"🚀 准备启动 Worker:")
        print(f"   - Agent 名称: {args.name}")
        print(f"   - Agent 类型: {args.workflow_types}")
        print(f"   - 监听地址:   {args.host}:{args.port}")

    agentscope.init()
    # MODIFIED: 将构建好的类列表传递给 custom_agent_classes
    server= RpcAgentServerLauncher(
        host= args.host,
        port= args.port,
        custom_agent_classes= agent_classes_to_run,
        max_expire_time= 3600, 
        max_timeout_seconds= 3600
    )
    server.launch(in_subprocess=False)
    print(f"✅ AgentScope RPC 服务已在 {args.host}:{args.port} 上启动。")
    print("   -> 按下 Ctrl+C 来停止服务。")
    server.wait_until_terminate()