"""
测试工作流可视化功能
"""
from maze import MaClient, task


@task(
    inputs=["raw_data"],
    outputs=["cleaned_data"],
    resources={"cpu": 2}
)
def data_cleaning(params):
    """数据清洗任务"""
    raw_data = params.get("raw_data")
    cleaned_data = raw_data.strip().lower()
    return {"cleaned_data": cleaned_data}


@task(
    inputs=["cleaned_data"],
    outputs=["analyzed_result"],
    resources={"cpu": 4, "cpu_mem": 2048}
)
def data_analysis(params):
    """数据分析任务"""
    cleaned_data = params.get("cleaned_data")
    analyzed_result = f"分析结果: {cleaned_data}"
    return {"analyzed_result": analyzed_result}


@task(
    inputs=["analyzed_result"],
    outputs=["report"],
    resources={"cpu": 1, "gpu_mem": 1024}
)
def generate_report(params):
    """生成报告任务"""
    analyzed_result = params.get("analyzed_result")
    report = f"[报告] {analyzed_result}"
    return {"report": report}


@task(
    inputs=["report"],
    outputs=["final_output"],
    resources={"cpu": 1}
)
def publish_result(params):
    """发布结果任务"""
    report = params.get("report")
    final_output = f"已发布: {report}"
    return {"final_output": final_output}


def test_linear_workflow_visualization():
    """
    测试线性工作流的可视化
    """
    print("=" * 60)
    print("测试1: 线性工作流可视化")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 创建线性工作流: task1 -> task2 -> task3 -> task4
    task1 = workflow.add_task(data_cleaning, inputs={"raw_data": "  TEST DATA  "})
    task2 = workflow.add_task(data_analysis, inputs={"cleaned_data": task1.outputs["cleaned_data"]})
    task3 = workflow.add_task(generate_report, inputs={"analyzed_result": task2.outputs["analyzed_result"]})
    task4 = workflow.add_task(publish_result, inputs={"report": task3.outputs["report"]})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ 添加了 4 个任务\n")
    
    # 方式1: ASCII 树形图
    print("\n" + "=" * 60)
    print("方式1: ASCII 树形图")
    print("=" * 60)
    workflow.print_graph()
    
    # 方式2: Mermaid 图
    print("\n" + "=" * 60)
    print("方式2: Mermaid 图代码")
    print("=" * 60)
    mermaid_code = workflow.get_graph_mermaid()
    print(mermaid_code)
    print("\n💡 可以将以上代码复制到 Mermaid 在线编辑器查看图形:")
    print("   https://mermaid.live/")
    
    # 方式3: 详细信息
    print("\n" + "=" * 60)
    print("方式3: 详细图结构信息")
    print("=" * 60)
    graph_info = workflow.get_graph_info()
    
    print(f"\n📊 统计信息:")
    print(f"  总任务数: {graph_info['stats']['total_tasks']}")
    print(f"  总边数: {graph_info['stats']['total_edges']}")
    print(f"  起始任务: {', '.join(graph_info['stats']['start_tasks'])}")
    print(f"  结束任务: {', '.join(graph_info['stats']['end_tasks'])}")
    
    print(f"\n📝 任务列表:")
    for i, node in enumerate(graph_info['nodes'], 1):
        print(f"\n  {i}. {node['name']} ({node['func_name']})")
        print(f"     ID: {node['task_id_short']}...")
        print(f"     Inputs: {', '.join(node['inputs'])}")
        print(f"     Outputs: {', '.join(node['outputs'])}")
        print(f"     Resources: {node['resources']}")
    
    print(f"\n🔗 依赖关系:")
    for i, edge in enumerate(graph_info['edges'], 1):
        print(f"  {i}. {edge['source_name']} → {edge['target_name']}")
    
    print("\n" + "=" * 60)
    print("✓ 线性工作流可视化测试完成")
    print("=" * 60)


@task(inputs=["input1"], outputs=["output1"])
def branch_task_a(params):
    """分支任务A"""
    return {"output1": "Branch A"}


@task(inputs=["input2"], outputs=["output2"])
def branch_task_b(params):
    """分支任务B"""
    return {"output2": "Branch B"}


@task(inputs=["merge_input1", "merge_input2"], outputs=["merged_output"])
def merge_task(params):
    """合并任务"""
    input1 = params.get("merge_input1")
    input2 = params.get("merge_input2")
    return {"merged_output": f"{input1} + {input2}"}


def test_branching_workflow_visualization():
    """
    测试分支和合并工作流的可视化
    """
    print("\n\n" + "=" * 60)
    print("测试2: 分支合并工作流可视化")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 创建分支合并工作流:
    #       root
    #      /    \
    #   task_a  task_b
    #      \    /
    #      merge
    
    root = workflow.add_task(data_cleaning, inputs={"raw_data": "root data"})
    task_a = workflow.add_task(branch_task_a, inputs={"input1": root.outputs["cleaned_data"]})
    task_b = workflow.add_task(branch_task_b, inputs={"input2": root.outputs["cleaned_data"]})
    merge = workflow.add_task(
        merge_task,
        inputs={
            "merge_input1": task_a.outputs["output1"],
            "merge_input2": task_b.outputs["output2"]
        }
    )
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ 添加了 4 个任务（分支合并结构）\n")
    
    # ASCII 可视化
    workflow.print_graph()
    
    # Mermaid 图
    print("\n" + "=" * 60)
    print("Mermaid 图代码")
    print("=" * 60)
    print(workflow.get_graph_mermaid())
    
    print("\n" + "=" * 60)
    print("✓ 分支合并工作流可视化测试完成")
    print("=" * 60)


def test_parallel_workflow_visualization():
    """
    测试并行工作流的可视化
    """
    print("\n\n" + "=" * 60)
    print("测试3: 并行工作流可视化")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 创建三个独立的并行任务链
    # Chain 1: A -> B
    # Chain 2: C -> D
    # Chain 3: E
    
    task_a = workflow.add_task(data_cleaning, inputs={"raw_data": "data A"})
    task_b = workflow.add_task(data_analysis, inputs={"cleaned_data": task_a.outputs["cleaned_data"]})
    
    task_c = workflow.add_task(branch_task_a, inputs={"input1": "data C"})
    task_d = workflow.add_task(branch_task_b, inputs={"input2": task_c.outputs["output1"]})
    
    task_e = workflow.add_task(generate_report, inputs={"analyzed_result": "data E"})
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ 添加了 5 个任务（3条并行链）\n")
    
    # ASCII 可视化
    workflow.print_graph()
    
    # 图信息
    graph_info = workflow.get_graph_info()
    print(f"\n📊 统计:")
    print(f"  起始任务: {', '.join(graph_info['stats']['start_tasks'])}")
    print(f"  结束任务: {', '.join(graph_info['stats']['end_tasks'])}")
    
    print("\n" + "=" * 60)
    print("✓ 并行工作流可视化测试完成")
    print("=" * 60)


def test_empty_workflow_visualization():
    """
    测试空工作流的可视化
    """
    print("\n\n" + "=" * 60)
    print("测试4: 空工作流可视化")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    print(f"\n✓ 空 Workflow 创建成功: {workflow.workflow_id[:8]}...")
    
    # ASCII 可视化
    print("\nASCII 可视化:")
    workflow.print_graph()
    
    # 图信息
    graph_info = workflow.get_graph_info()
    print(f"\n统计: {graph_info['stats']}")
    
    print("\n" + "=" * 60)
    print("✓ 空工作流可视化测试完成")
    print("=" * 60)


def test_draw_graph_image():
    """
    测试图片导出功能
    """
    print("\n\n" + "=" * 60)
    print("测试5: 导出工作流为图片")
    print("=" * 60)
    
    client = MaClient()
    workflow = client.create_workflow()
    
    # 创建一个有趣的工作流
    task1 = workflow.add_task(data_cleaning, inputs={"raw_data": "data"})
    task2 = workflow.add_task(data_analysis, inputs={"cleaned_data": task1.outputs["cleaned_data"]})
    task3 = workflow.add_task(branch_task_a, inputs={"input1": task2.outputs["analyzed_result"]})
    task4 = workflow.add_task(branch_task_b, inputs={"input2": task2.outputs["analyzed_result"]})
    task5 = workflow.add_task(
        merge_task,
        inputs={
            "merge_input1": task3.outputs["output1"],
            "merge_input2": task4.outputs["output2"]
        }
    )
    
    print(f"\n✓ Workflow 创建成功: {workflow.workflow_id[:8]}...")
    print(f"✓ 添加了 5 个任务\n")
    
    # 方式1: 自动选择（推荐）
    print("尝试导出图片...")
    try:
        output_file = workflow.draw_graph("workflow_auto.png", method="auto")
        print(f"✓ 图片已保存到: {output_file}")
    except ImportError as e:
        print(f"⚠️  无法导出图片: {e}")
        print("   请安装以下任一组合:")
        print("   - pip install graphviz  (推荐，需要系统安装 Graphviz)")
        print("   - pip install matplotlib networkx")
    
    # 方式2: 使用 Graphviz（如果可用）
    print("\n尝试使用 Graphviz 导出...")
    try:
        output_file = workflow.draw_graph("workflow_graphviz.png", method="graphviz", dpi=200)
        print(f"✓ Graphviz 图片已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️  Graphviz 不可用: {str(e)[:100]}")
    
    # 方式3: 使用 Matplotlib（如果可用）
    print("\n尝试使用 Matplotlib 导出...")
    try:
        output_file = workflow.draw_graph("workflow_matplotlib.png", method="matplotlib", 
                                          figsize=(14, 10), dpi=150)
        print(f"✓ Matplotlib 图片已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️  Matplotlib 不可用: {str(e)[:100]}")
    
    # 方式4: 导出为 PDF（Graphviz）
    print("\n尝试导出为 PDF...")
    try:
        output_file = workflow.draw_graph("workflow.pdf", method="graphviz")
        print(f"✓ PDF 已保存到: {output_file}")
    except Exception as e:
        print(f"⚠️  无法导出 PDF: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("✓ 图片导出测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_linear_workflow_visualization()
    test_branching_workflow_visualization()
    test_parallel_workflow_visualization()
    test_empty_workflow_visualization()
    test_draw_graph_image()
    
    print("\n\n" + "=" * 70)
    print("🎉 所有可视化测试完成！")
    print("=" * 70)
    print("\n💡 使用方式总结:")
    print("  1. workflow.print_graph()              - 打印ASCII树形图")
    print("  2. workflow.get_graph_ascii()          - 获取ASCII字符串")
    print("  3. workflow.get_graph_mermaid()        - 获取Mermaid图代码")
    print("  4. workflow.get_graph_info()           - 获取详细图结构信息")
    print("  5. workflow.draw_graph('file.png')     - 导出为图片 ⭐ 新功能")
    print("=" * 70)
    print("\n📦 图片导出依赖（任选其一）:")
    print("  方案1（推荐）: pip install graphviz")
    print("             + 系统安装 Graphviz (https://graphviz.org/download/)")
    print("  方案2: pip install matplotlib networkx")
    print("=" * 70)

