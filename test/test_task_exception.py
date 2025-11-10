"""
测试任务异常处理
"""
from maze import MaClient, task


@task(
    inputs=["value"],
    outputs=["result"]
)
def task_division(params):
    """
    执行除法操作，可能会抛出异常
    """
    value = params.get("value")
    # 这里会抛出 ZeroDivisionError
    result = 100 / value
    return {"result": result}


@task(
    inputs=["text"],
    outputs=["processed"]
)
def task_type_error(params):
    """
    类型错误测试
    """
    text = params.get("text")
    # 这里会抛出 TypeError（如果text不是字符串）
    result = text.upper()
    return {"processed": result}


@task(
    inputs=["index"],
    outputs=["item"]
)
def task_index_error(params):
    """
    索引错误测试
    """
    index = params.get("index")
    items = ["a", "b", "c"]
    # 可能抛出 IndexError
    item = items[index]
    return {"item": item}


@task(
    inputs=["filename"],
    outputs=["content"]
)
def task_file_error(params):
    """
    文件错误测试
    """
    filename = params.get("filename")
    # 可能抛出 FileNotFoundError
    with open(filename, 'r') as f:
        content = f.read()
    return {"content": content}


@task(
    inputs=["value"],
    outputs=["result"]
)
def task_normal(params):
    """
    正常任务，不会抛出异常
    """
    value = params.get("value")
    return {"result": f"正常处理: {value}"}


def test_division_by_zero():
    """
    测试除零错误
    """
    print("=" * 60)
    print("测试1：除零错误 (ZeroDivisionError)")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 添加会抛出异常的任务
    task1 = workflow.add_task(task_division, inputs={"value": 0})  # 除以0
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    print("✓ 添加了除以0的任务\n")
    
    # 运行workflow
    run_id = workflow.run()
    print(f"Run ID: {run_id[:8]}...\n")
    
    # 获取结果
    messages = workflow.get_results(run_id, verbose=False)
    
    # 分析消息
    has_exception = False
    exception_message = None
    
    print("消息分析:")
    for msg in messages:
        msg_type = msg.get("type")
        msg_data = msg.get("data", {})
        
        print(f"  - 消息类型: {msg_type}")
        
        if msg_type == "task_exception":
            has_exception = True
            task_id = msg_data.get("task_id", "")[:8]
            exception_message = msg_data.get("result", "")
            
            print(f"    任务ID: {task_id}...")
            print(f"    异常信息: {exception_message[:100]}...")
    
    # 验证
    assert has_exception, "应该捕获到异常"
    assert "ZeroDivisionError" in str(exception_message) or "division" in str(exception_message).lower(), "应该是除零错误"
    
    print(f"\n{'='*60}")
    print("✓ 测试通过：成功捕获除零错误")
    print(f"{'='*60}")


def test_type_error():
    """
    测试类型错误
    """
    print("\n\n" + "=" * 60)
    print("测试2：类型错误 (TypeError)")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 添加会抛出类型错误的任务
    task1 = workflow.add_task(task_type_error, inputs={"text": 123})  # 传入数字而不是字符串
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    print("✓ 添加了类型错误的任务\n")
    
    # 运行workflow
    run_id = workflow.run()
    print(f"Run ID: {run_id[:8]}...\n")
    
    # 获取结果
    messages = workflow.get_results(run_id, verbose=False)
    
    # 分析消息
    has_exception = False
    
    print("消息分析:")
    for msg in messages:
        msg_type = msg.get("type")
        
        print(f"  - 消息类型: {msg_type}")
        
        if msg_type == "task_exception":
            has_exception = True
            msg_data = msg.get("data", {})
            task_id = msg_data.get("task_id", "")[:8]
            print(f"    任务ID: {task_id}...")
            print(f"    ✓ 捕获到异常")
    
    # 验证
    assert has_exception, "应该捕获到异常"
    
    print(f"\n{'='*60}")
    print("✓ 测试通过：成功捕获类型错误")
    print(f"{'='*60}")


def test_mixed_success_and_failure():
    """
    测试混合场景：部分任务成功，部分任务失败
    """
    print("\n\n" + "=" * 60)
    print("测试3：混合场景（正常任务 + 异常任务）")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 添加一个正常任务
    task1 = workflow.add_task(task_normal, inputs={"value": "test123"})
    
    # 添加一个会抛出异常的任务（独立的，不依赖task1）
    task2 = workflow.add_task(task_index_error, inputs={"index": 10})  # 索引越界
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    print("✓ 添加了1个正常任务和1个异常任务\n")
    
    # 运行workflow
    run_id = workflow.run()
    print(f"Run ID: {run_id[:8]}...\n")
    
    # 获取结果
    messages = workflow.get_results(run_id, verbose=False)
    
    # 统计
    success_count = 0
    exception_count = 0
    
    print("任务执行情况:")
    for msg in messages:
        msg_type = msg.get("type")
        msg_data = msg.get("data", {})
        
        if msg_type == "finish_task":
            success_count += 1
            task_id = msg_data.get("task_id", "")[:8]
            result = msg_data.get("result", {})
            print(f"  ✓ 任务成功: {task_id}... - 结果: {result}")
        
        elif msg_type == "task_exception":
            exception_count += 1
            task_id = msg_data.get("task_id", "")[:8]
            print(f"  ✗ 任务异常: {task_id}...")
    
    print(f"\n统计:")
    print(f"  成功任务: {success_count}")
    print(f"  异常任务: {exception_count}")
    
    # 验证
    assert success_count > 0, "应该有成功的任务"
    assert exception_count > 0, "应该有失败的任务"
    
    print(f"\n{'='*60}")
    print("✓ 测试通过：正确处理混合场景")
    print(f"{'='*60}")


def test_exception_message_format():
    """
    测试异常消息格式
    """
    print("\n\n" + "=" * 60)
    print("测试4：异常消息格式验证")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 添加会抛出异常的任务
    task1 = workflow.add_task(task_file_error, inputs={"filename": "nonexistent_file.txt"})
    
    print(f"\n✓ Workflow创建成功: {workflow.workflow_id[:8]}...")
    print("✓ 添加了文件不存在的任务\n")
    
    # 运行workflow
    run_id = workflow.run()
    print(f"Run ID: {run_id[:8]}...\n")
    
    # 获取结果
    messages = workflow.get_results(run_id, verbose=False)
    
    # 验证消息格式
    exception_msg = None
    
    for msg in messages:
        if msg.get("type") == "task_exception":
            exception_msg = msg
            break
    
    assert exception_msg is not None, "应该有异常消息"
    
    # 验证消息结构
    print("异常消息结构验证:")
    
    assert "type" in exception_msg, "消息应该有type字段"
    print(f"  ✓ type: {exception_msg['type']}")
    
    assert "data" in exception_msg, "消息应该有data字段"
    print(f"  ✓ data: 存在")
    
    data = exception_msg["data"]
    assert "task_id" in data, "data应该有task_id字段"
    print(f"  ✓ task_id: {data['task_id'][:8]}...")
    
    assert "result" in data, "data应该有result字段（异常信息）"
    print(f"  ✓ result: {str(data['result'])[:80]}...")
    
    print(f"\n{'='*60}")
    print("✓ 测试通过：异常消息格式正确")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_division_by_zero()
    test_type_error()
    test_mixed_success_and_failure()
    test_exception_message_format()
    
    print("\n\n" + "=" * 60)
    print("🎉 所有异常处理测试通过！")
    print("=" * 60)


