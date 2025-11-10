"""
测试同一个workflow多次运行并获取每次结果
"""
from datetime import datetime
from maze import MaClient, task


@task(
    inputs=["run_number"],
    outputs=["result", "timestamp"]
)
def process_data(params):
    """
    处理数据任务，返回运行编号和时间戳
    """
    run_number = params.get("run_number")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    result = f"运行 #{run_number} 完成"
    
    return {
        "result": result,
        "timestamp": timestamp
    }


@task(
    inputs=["prev_result"],
    outputs=["final_result"]
)
def format_output(params):
    """
    格式化输出任务
    """
    prev_result = params.get("prev_result")
    final_result = f"[最终输出] {prev_result}"
    
    return {
        "final_result": final_result
    }


def test_multiple_runs():
    """
    测试同一个workflow多次运行
    
    注意：同一个workflow的输入参数在定义时就固定了，
    多次运行会使用相同的输入参数，但每次运行有独立的run_id
    """
    print("=" * 60)
    print("测试：同一个workflow多次运行")
    print("=" * 60)
    
    # 创建客户端和workflow
    client = MaClient()
    workflow = client.create_workflow()
    
    # 定义workflow结构（只定义一次，在循环外）
    task1 = workflow.add_task(process_data, inputs={"run_number": "固定输入"})
    task2 = workflow.add_task(format_output, inputs={"prev_result": task1.outputs["result"]})
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ 添加了 2 个任务")
    print(f"✓ 输入参数: run_number='固定输入'")
    print(f"  （注意：同一workflow的输入在定义时固定，多次运行使用相同输入）\n")
    
    # 运行workflow 3次（相同的workflow定义）
    num_runs = 3
    all_results = []
    
    for i in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"第 {i} 次运行")
        print(f"{'='*60}")
        
        # 运行workflow（不重新添加任务）
        run_id = workflow.run()
        print(f"Run ID: {run_id[:8]}...")
        
        # 获取结果（使用get_results，不打印原始消息）
        messages = workflow.get_results(run_id, verbose=False)
        
        # 解析结果
        run_results = {
            "run_number": i,
            "run_id": run_id,
            "task_results": {},
            "messages_count": len(messages)
        }
        
        for msg in messages:
            msg_type = msg.get("type")
            msg_data = msg.get("data", {})
            
            if msg_type == "finish_task":
                task_id = msg_data.get("task_id")
                result = msg_data.get("result")
                if task_id and result:
                    run_results["task_results"][task_id] = result
                    
                    # 打印任务结果
                    for key, value in result.items():
                        print(f"  {key}: {value}")
            
            elif msg_type == "finish_workflow":
                print(f"✓ Workflow完成")
        
        all_results.append(run_results)
        
        # 验证每次运行都有结果
        assert len(run_results["task_results"]) == 2, f"第{i}次运行应该有2个任务结果，实际有{len(run_results['task_results'])}个"
    
    # 总结
    print(f"\n{'='*60}")
    print("运行总结")
    print(f"{'='*60}")
    print(f"总运行次数: {num_runs}")
    print(f"Workflow ID: {workflow.workflow_id[:8]}...")
    
    for i, result in enumerate(all_results, 1):
        print(f"\n第 {i} 次运行:")
        print(f"  Run ID: {result['run_id'][:8]}...")
        print(f"  消息数: {result['messages_count']}")
        print(f"  任务结果数: {len(result['task_results'])}")
    
    # 验证每次运行的run_id都不同
    run_ids = [r["run_id"] for r in all_results]
    assert len(set(run_ids)) == num_runs, "每次运行的run_id应该不同"
    
    print(f"\n{'='*60}")
    print("🎉 测试通过！")
    print("✓ 同一个workflow可以多次运行")
    print("✓ 每次运行有独立的run_id")
    print("✓ 每次运行都能正确获取结果")
    print(f"{'='*60}")


def test_concurrent_workflow_runs():
    """
    测试快速连续运行workflow
    """
    print("\n\n" + "=" * 60)
    print("测试：快速连续运行workflow")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 添加简单任务
    @task(inputs=["value"], outputs=["result"])
    def simple_task(params):
        value = params.get("value")
        return {"result": f"处理了: {value}"}
    
    task1 = workflow.add_task(simple_task, inputs={"value": "test"})
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    
    # 快速启动5次运行
    run_ids = []
    for i in range(1, 6):
        run_id = workflow.run()
        run_ids.append(run_id)
        print(f"  启动第 {i} 次运行: {run_id[:8]}...")
    
    # 收集所有结果
    print("\n收集结果:")
    for i, run_id in enumerate(run_ids, 1):
        messages = workflow.get_results(run_id, verbose=False)
        
        # 统计消息类型
        msg_types = [msg.get("type") for msg in messages]
        has_finish = "finish_workflow" in msg_types
        
        print(f"  第 {i} 次运行: {run_id[:8]}... - {'✓ 完成' if has_finish else '✗ 未完成'}")
    
    print(f"\n{'='*60}")
    print("🎉 快速连续运行测试通过！")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_multiple_runs()
    test_concurrent_workflow_runs()


