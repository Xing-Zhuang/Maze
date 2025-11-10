"""
测试工作流结果缓存功能

这个测试演示了如何使用客户端缓存来避免服务端"只能消费一次"的限制
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maze import MaClient, task


@task(inputs=["x"], outputs=["y"])
def multiply_task(params):
    """简单的乘法任务"""
    x = params.get("x")
    result = x * 2
    return {"y": result}


@task(inputs=["a", "b"], outputs=["sum"])
def add_task(params):
    """简单的加法任务"""
    a = params.get("a")
    b = params.get("b")
    return {"sum": a + b}


def test_basic_cache():
    """
    测试1: 基本缓存功能
    """
    print("\n" + "=" * 60)
    print("测试1: 基本缓存功能")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    task1 = workflow.add_task(multiply_task, inputs={"x": 10})
    task2 = workflow.add_task(add_task, inputs={"a": task1.outputs["y"], "b": 5})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    
    # 运行工作流
    run_id = workflow.run()
    print(f"✓ 运行 ID: {run_id[:8]}...")
    
    # 第一次获取结果（从服务器）
    print("\n📥 第一次获取结果（从服务器）...")
    messages1 = workflow.get_results(run_id, verbose=False)
    print(f"✓ 获取到 {len(messages1)} 条消息")
    
    # 第二次获取结果（从缓存）
    print("\n📥 第二次获取结果（从缓存）...")
    messages2 = workflow.get_results(run_id, verbose=False)
    print(f"✓ 获取到 {len(messages2)} 条消息")
    
    # 验证两次结果相同
    assert messages1 == messages2, "两次获取的结果应该相同"
    print("✓ 验证通过：两次结果完全一致")
    
    print("\n" + "=" * 60)
    print("✓ 基本缓存测试完成")
    print("=" * 60)


def test_get_task_result():
    """
    测试2: 查询特定任务结果
    """
    print("\n\n" + "=" * 60)
    print("测试2: 查询特定任务结果")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    task1 = workflow.add_task(multiply_task, inputs={"x": 100})
    task2 = workflow.add_task(add_task, inputs={"a": task1.outputs["y"], "b": 50})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ Task1 ID: {task1.task_id[:8]}...")
    print(f"✓ Task2 ID: {task2.task_id[:8]}...")
    
    # 运行并获取结果
    run_id = workflow.run()
    workflow.get_results(run_id, verbose=False)
    
    # 查询特定任务的结果
    print("\n🔍 查询 Task1 结果...")
    task1_result = workflow.get_task_result(run_id, task1.task_id)
    print(f"✓ Task1 结果: {task1_result}")
    assert task1_result is not None, "应该找到 Task1 的结果"
    assert task1_result["status"] == "success", "Task1 应该成功"
    assert task1_result["result"]["y"] == 200, "Task1 结果应该是 200"
    
    print("\n🔍 查询 Task2 结果...")
    task2_result = workflow.get_task_result(run_id, task2.task_id)
    print(f"✓ Task2 结果: {task2_result}")
    assert task2_result is not None, "应该找到 Task2 的结果"
    assert task2_result["status"] == "success", "Task2 应该成功"
    assert task2_result["result"]["sum"] == 250, "Task2 结果应该是 250"
    
    # 测试使用短 ID 查询
    print("\n🔍 使用短 ID 查询 Task1...")
    task1_short_result = workflow.get_task_result(run_id, task1.task_id[:8])
    print(f"✓ 使用短 ID 查询成功: {task1_short_result['result']}")
    assert task1_short_result == task1_result, "使用短 ID 应该得到相同结果"
    
    print("\n" + "=" * 60)
    print("✓ 查询特定任务结果测试完成")
    print("=" * 60)


def test_multiple_runs_cache():
    """
    测试3: 多次运行的缓存
    """
    print("\n\n" + "=" * 60)
    print("测试3: 多次运行的缓存")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    task1 = workflow.add_task(multiply_task, inputs={"x": 5})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    
    # 运行三次
    run_ids = []
    for i in range(3):
        run_id = workflow.run()
        run_ids.append(run_id)
        print(f"✓ 第 {i+1} 次运行: {run_id[:8]}...")
        workflow.get_results(run_id, verbose=False)
    
    # 列出所有缓存的运行
    print("\n📋 列出所有缓存的运行...")
    cached_runs = workflow.list_cached_runs()
    print(f"✓ 缓存了 {len(cached_runs)} 次运行的结果")
    for idx, cached_run_id in enumerate(cached_runs):
        print(f"  {idx+1}. {cached_run_id[:16]}...")
    
    assert len(cached_runs) == 3, "应该缓存了 3 次运行"
    
    # 验证可以分别查询每次运行的结果
    print("\n🔍 查询每次运行的结果...")
    for idx, run_id in enumerate(run_ids):
        result = workflow.get_task_result(run_id, task1.task_id)
        print(f"  第 {idx+1} 次运行结果: {result['result']}")
        assert result is not None, f"应该能找到第 {idx+1} 次运行的结果"
    
    print("\n" + "=" * 60)
    print("✓ 多次运行缓存测试完成")
    print("=" * 60)


def test_cache_management():
    """
    测试4: 缓存管理（清除缓存）
    """
    print("\n\n" + "=" * 60)
    print("测试4: 缓存管理")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    task1 = workflow.add_task(multiply_task, inputs={"x": 3})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    
    # 运行两次
    run_id1 = workflow.run()
    run_id2 = workflow.run()
    
    workflow.get_results(run_id1, verbose=False)
    workflow.get_results(run_id2, verbose=False)
    
    cached_runs = workflow.list_cached_runs()
    print(f"✓ 缓存了 {len(cached_runs)} 次运行")
    assert len(cached_runs) == 2, "应该缓存了 2 次运行"
    
    # 清除特定运行的缓存
    print(f"\n🗑️  清除第一次运行的缓存: {run_id1[:8]}...")
    workflow.clear_cache(run_id1)
    cached_runs = workflow.list_cached_runs()
    print(f"✓ 剩余缓存: {len(cached_runs)} 次运行")
    assert len(cached_runs) == 1, "应该剩余 1 次运行"
    assert run_id1 not in cached_runs, "run_id1 应该被清除"
    assert run_id2 in cached_runs, "run_id2 应该还在"
    
    # 清除所有缓存
    print("\n🗑️  清除所有缓存...")
    workflow.clear_cache()
    cached_runs = workflow.list_cached_runs()
    print(f"✓ 剩余缓存: {len(cached_runs)} 次运行")
    assert len(cached_runs) == 0, "所有缓存应该被清除"
    
    print("\n" + "=" * 60)
    print("✓ 缓存管理测试完成")
    print("=" * 60)


def test_cache_with_show_results():
    """
    测试5: show_results 也会缓存结果
    """
    print("\n\n" + "=" * 60)
    print("测试5: show_results 也会缓存")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    task1 = workflow.add_task(multiply_task, inputs={"x": 7})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    
    run_id = workflow.run()
    
    # 使用 show_results（应该也会缓存）
    print("\n📥 使用 show_results 获取结果...")
    results = workflow.show_results(run_id)
    
    # 验证已经缓存
    cached_runs = workflow.list_cached_runs()
    print(f"\n✓ 缓存了 {len(cached_runs)} 次运行")
    assert run_id in cached_runs, "show_results 应该也缓存结果"
    
    # 现在可以使用 get_task_result 查询
    print("\n🔍 从缓存中查询任务结果...")
    task_result = workflow.get_task_result(run_id, task1.task_id)
    print(f"✓ 查询成功: {task_result['result']}")
    assert task_result is not None, "应该能从缓存查询到结果"
    
    print("\n" + "=" * 60)
    print("✓ show_results 缓存测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_cache()
    test_get_task_result()
    test_multiple_runs_cache()
    test_cache_management()
    test_cache_with_show_results()
    
    print("\n\n" + "=" * 70)
    print("🎉 所有缓存功能测试完成！")
    print("=" * 70)
    print("\n💡 使用方式总结:")
    print("  1. workflow.get_results(run_id)         - 自动缓存结果")
    print("  2. workflow.show_results(run_id)        - 自动缓存结果")
    print("  3. workflow.get_task_result(run_id, task_id) - 查询特定任务")
    print("  4. workflow.list_cached_runs()          - 列出所有缓存")
    print("  5. workflow.clear_cache(run_id)         - 清除缓存")
    print("=" * 70)
    print("\n⭐ 优势:")
    print("  • 多次查询同一 run_id 不会重复连接服务器")
    print("  • 支持按 task_id 查询特定任务结果")
    print("=" * 70)

